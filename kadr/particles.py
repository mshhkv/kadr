"""The `particles` layer: a picture breaking into grains that the wind carries off (mode blow), or grains flying in and
landing into the picture (mode gather — the same flight run backwards). render.py places the result like a picture.

The picture is cut into cells of `cell` content px. A front crosses it — starting at `start`, taking `sweep` s,
travelling `direction` degrees (0: left → right, 90: top → bottom) — and as it passes a cell (± `jitter` s) the cell
becomes a grain: a square of the cell's mean colour that flies with `wind` (content px/s), a random `spread` of its
own, `turbulence` (a smooth noise drift) and `gravity`, shrinking by `shrink` and fading out over `life` s. Cells the
front has not reached are the picture itself, pixel for pixel. A grain is drawn as a square with exact edge coverage
(corner impulses into a summed-area table), several times along its path within the shutter when it moves fast."""
import functools, math, os

import numpy as np
from PIL import Image

from .effects import fbm


@functools.lru_cache(8)
def _cells(path, mtime, crop, w, h, cell):
    """the picture's cells: colours (n, 4) premultiplied 0..255 / alpha 0..1, homes (n, 2), grid (ny, nx), cell size"""
    im = Image.open(path).convert('RGBA')
    if crop:
        x, y, cw, ch = crop
        im = im.crop((int(x), int(y), int(x + cw), int(y + ch)))
    nx, ny = max(1, int(round(w / cell))), max(1, int(round(h / cell)))
    small = np.asarray(im.convert('RGBa').resize((nx, ny), Image.BOX), np.float32)
    col = np.concatenate([small[..., :3], small[..., 3:] / 255], -1).reshape(-1, 4)
    cw_, ch_ = w / nx, h / ny
    ys, xs = np.mgrid[0:ny, 0:nx]
    homes = np.stack([(xs.ravel() + 0.5) * cw_, (ys.ravel() + 0.5) * ch_], -1)
    return col, homes, (ny, nx), (cw_, ch_)


@functools.lru_cache(2)
def _source(path, mtime, crop):
    """the picture's pixels, cropped, premultiplied RGBa"""
    im = Image.open(path).convert('RGBA')
    if crop:
        x, y, cw, ch = crop
        im = im.crop((int(x), int(y), int(x + cw), int(y + ch)))
    return im.convert('RGBa')


def _picture(path, mtime, crop, W, H, win):
    """the window win = (xa, ya, xb, yb) of the picture drawn at W × H canvas px, premultiplied float. Only the
    window is resampled: at a strong zoom the whole picture would be gigabytes (v1 cached four of them and the
    render's workers died of memory)"""
    im = _source(path, mtime, crop)
    xa, ya, xb, yb = win
    sx, sy = im.width / W, im.height / H
    a = np.asarray(im.resize((xb - xa, yb - ya), Image.LANCZOS, box=(xa * sx, ya * sy, xb * sx, yb * sy)), np.float32)
    return np.concatenate([a[..., :3], a[..., 3:] / 255], -1)


def _rand(seed, n, k):
    return np.random.default_rng(seed).random((k, n))


def _splat(H, W, x0, y0, x1, y1, col):
    """squares (canvas px, corners x0 y0 – x1 y1) of premultiplied colours col (n, 4) summed into an (H, W, 4)
    picture, each with its exact area coverage: ±corner impulses split bilinearly, then a 2D running sum. All 16
    deposits of all squares and their 4 channels go through one bincount"""
    idx, wts = [], []
    for (cx, cy, sgn) in ((x0, y0, 1.0), (x1, y0, -1.0), (x0, y1, -1.0), (x1, y1, 1.0)):
        ix, iy = np.floor(cx).astype(np.int64), np.floor(cy).astype(np.int64)
        fx, fy = cx - ix, cy - iy
        for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            X, Y = np.clip(ix + dx, 0, W + 1), np.clip(iy + dy, 0, H + 1)
            idx.append(Y * (W + 2) + X)
            wts.append(sgn * wgt)
    idx = np.concatenate(idx)
    wts = np.concatenate(wts)
    c4 = np.tile(col, (16, 1))                                 # the colour of each deposit (16 per square)
    full = (idx[:, None] * 4 + np.arange(4)[None, :]).ravel()
    out = np.bincount(full, weights=(wts[:, None] * c4).ravel(), minlength=(H + 2) * (W + 2) * 4)
    out = out.reshape(H + 2, W + 2, 4).cumsum(0).cumsum(1)
    return out[:H, :W]


def render(L, t, sc, motion_blur, fps, view=None):
    """the layer at t drawn in its own plane at `sc` px per content px → (x0, y0, w, h of the canvas in content px,
    premultiplied PIL RGBa picture of it) or None"""
    w, h = L['psize']
    cell = max(1.0, L['cell'])
    path = L['src']
    mt = os.path.getmtime(path)
    col, homes, (ny, nx), (cw_, ch_) = _cells(path, mt, L['crop'], w, h, cell)
    n = len(col)
    r = _rand(L['seed'], n, 6)
    # the front: each cell's place along `direction`, 0 where the front starts … 1 where it ends
    th = math.radians(L['direction'])
    d = homes[:, 0] * math.cos(th) + homes[:, 1] * math.sin(th)
    d = (d - d.min()) / max(1e-9, d.max() - d.min())
    tr = L['start'] + L['sweep'] * d + L['jitter'] * (r[0] - 0.5) * 2
    life = max(1e-3, L['life'])
    age = (t - tr) if L['mode'] == 'blow' else (tr - t)
    intact = age <= 0
    fly = (age > 0) & (age < life) & (col[:, 3] > 1e-3)
    # the canvas: the picture's box and wherever flying grains are, at most one picture size beyond it
    vx = L['wind'][0] + L['spread'] * (r[1] - 0.5) * 2
    vy = L['wind'][1] + L['spread'] * (r[2] - 0.5) * 2
    hs = (motion_blur['shutter'] / 360 / fps / 2) if motion_blur else 0.0

    def pos(a):
        hx, hy = homes[fly, 0], homes[fly, 1]
        tu = L['turbulence'] * a
        ox = (fbm(hx / 90 + 0.7 * a, hy / 90, L['seed']) - 0.5) * 2 * tu if L['turbulence'] else 0.0
        oy = (fbm(hx / 90, hy / 90 + 0.7 * a, L['seed'] + 5) - 0.5) * 2 * tu if L['turbulence'] else 0.0
        return hx + vx[fly] * a + ox, hy + vy[fly] * a + 0.5 * L['gravity'] * a * a + oy

    a0 = age[fly]
    K = 1
    if hs > 0 and fly.any():
        speed = float(np.max(np.hypot(vx[fly], vy[fly]))) + L['turbulence'] + abs(L['gravity']) * life
        K = int(min(8, max(1, math.ceil(speed * 2 * hs * sc / 1.5))))
    sub = [a0 + ((j + 0.5) / K * 2 - 1) * hs * (1 if L['mode'] == 'blow' else -1) for j in range(K)]
    ps = [pos(np.clip(a, 0, None)) for a in sub] if fly.any() else []
    margin = max(w, h)
    xs_all = [0.0, w] + [float(p[0].min()) - cell for p in ps] + [float(p[0].max()) + cell for p in ps]
    ys_all = [0.0, h] + [float(p[1].min()) - cell for p in ps] + [float(p[1].max()) + cell for p in ps]
    bx0, by0 = max(-margin, min(xs_all)), max(-margin, min(ys_all))
    bx1, by1 = min(w + margin, max(xs_all)), min(h + margin, max(ys_all))
    if view is not None:                                   # only what the frame can see (render._particle_view)
        bx0, by0, bx1, by1 = max(bx0, view[0]), max(by0, view[1]), min(bx1, view[2]), min(by1, view[3])
        if bx1 <= bx0 or by1 <= by0:
            return None
    # the canvas grid snapped to whole canvas px of the picture's own grid, so the intact part is the picture exactly
    ox_px, oy_px = math.floor(bx0 * sc), math.floor(by0 * sc)
    W, H = int(math.ceil(bx1 * sc)) - ox_px, int(math.ceil(by1 * sc)) - oy_px
    if W <= 0 or H <= 0 or W * H > 40e6:
        return None
    pw, ph = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    canvas = np.zeros((H, W, 4), np.float64)
    if intact.any():
        x0, y0 = -ox_px, -oy_px
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W, x0 + pw), min(H, y0 + ph)
        if xa < xb and ya < yb:                            # only the part on the canvas (it may be a small window of it)
            pic = _picture(path, mt, L['crop'], pw, ph, (xa - x0, ya - y0, xb - x0, yb - y0))
            gx = np.minimum(nx - 1, ((np.arange(xa - x0, xb - x0) + 0.5) / pw * nx).astype(np.int64))
            gy = np.minimum(ny - 1, ((np.arange(ya - y0, yb - y0) + 0.5) / ph * ny).astype(np.int64))
            keep = intact.reshape(ny, nx)[gy[:, None], gx[None, :]]
            canvas[ya:yb, xa:xb] = pic * keep[..., None]
    if ps:
        fade = (1 - a0 / life)
        size = cell * (1 - L['shrink'] * a0 / life)
        c = col[fly].astype(np.float64)
        pm = np.concatenate([c[:, :3] * fade[:, None] / K, c[:, 3:] * fade[:, None] / K], -1)   # colour already × alpha
        half = size * sc / 2
        for px, py in ps:
            X, Y = px * sc - ox_px, py * sc - oy_px
            on = (X + half > 0) & (X - half < W) & (Y + half > 0) & (Y - half < H)   # grains off the canvas add nothing
            if on.any():
                hs_ = half[on] if np.ndim(half) else half
                canvas += _splat(H, W, X[on] - hs_, Y[on] - hs_, X[on] + hs_, Y[on] + hs_, pm[on])
    over = np.maximum(canvas[..., 3:], 1.0)                # grains piled up: their mean colour, not a brighter sum
    canvas = canvas / over
    a = np.clip(canvas[..., 3:], 0, 1)
    rgb = np.clip(canvas[..., :3], 0, 255 * a)
    out = np.concatenate([rgb, a * 255], -1)
    im = Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8), 'RGBa')
    return ox_px / sc, oy_px / sc, W / sc, H / sc, im
