"""Built-in effects. Each is registered exactly as a plugin's would be (see plugins.py): it gets the layer's own picture —
colour (h, w, 3) 0..255 and alpha (h, w, 1) 0..1, after the layer's blur and filter, before its shadows — and returns a
new one. Sizes in params are layer px: multiply by fx.scale for screen px."""
import numpy as np
from .plugins import effect

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


@effect('glow', params={'radius': 12.0, 'strength': 0.8, 'color': None, 'space': 'frame'}, grow=lambda p, s: 3 * p['radius'] * s)
def glow(rgb, a, p, fx):
    """light around the layer: a blurred copy added under it (the layer's own colour, or `color`). space: layer — the
    halo squashed and turned with the layer (CSS: a filter on a scaled element), frame — round in the frame"""
    from .render import cov_reach
    sig = p['radius'] * fx.scale
    layer = p.get('space') == 'layer' and fx.J is not None
    C = fx.layer_cov(p['radius']) if layer else None
    rgb, a, n = fx.grow(rgb, a, int(np.ceil(max(cov_reach(C)) if layer else 3 * sig)) + 1)
    src = np.broadcast_to(np.array(p['color'], np.float32), rgb.shape) if p['color'] is not None else rgb
    x = np.concatenate([src * a, a], -1)
    halo = (fx.blur_layer(x, p['radius']) if layer else fx.blur(x, sig)) * p['strength']
    ha = np.clip(halo[..., 3:], 0, 1)
    out_a = a + ha * (1 - a)
    out = (rgb * a + halo[..., :3] * (1 - a)) / np.maximum(out_a, 1e-6)
    return np.clip(out, 0, 255), out_a, n


@effect('fill', params={'color': (255, 255, 255), 'amount': 1.0})
def fill(rgb, a, p, fx):
    """the layer in one colour (a silhouette), keeping its alpha"""
    return rgb + (np.array(p['color'], np.float32) - rgb) * p['amount'], a


@effect('tint', params={'black': (0, 0, 0), 'white': (255, 255, 255), 'amount': 1.0})
def tint(rgb, a, p, fx):
    """brightness mapped from `black` to `white` (After Effects Tint): duotone archive, a channel's colour cast"""
    y = (rgb @ LUMA)[..., None] / 255
    b, w = np.array(p['black'], np.float32), np.array(p['white'], np.float32)
    return rgb + (b + (w - b) * y - rgb) * p['amount'], a


@effect('levels', params={'black': 0.0, 'white': 255.0, 'gamma': 1.0, 'out_black': 0.0, 'out_white': 255.0})
def levels(rgb, a, p, fx):
    """input black/white points, gamma, output range — per channel, like Levels"""
    x = np.clip((rgb - p['black']) / max(1e-3, p['white'] - p['black']), 0, 1) ** (1 / max(1e-3, p['gamma']))
    return p['out_black'] + x * (p['out_white'] - p['out_black']), a


@effect('invert', params={'amount': 1.0})
def invert(rgb, a, p, fx):
    return rgb + (255 - 2 * rgb) * p['amount'], a


@effect('motion_tile', params={'tile': [1080.0, 1920.0], 'center': [540.0, 960.0], 'mirror': True})
def motion_tile(rgb, a, p, fx):
    """After Effects' Motion Tile with mirror edges, for a collapsing group (its picture is the whole frame): the
    `tile` × `center` rectangle (frame px, animatable — a zoom-out shrinks it) repeats around itself, every neighbour
    mirrored, so a picture pulled back from the camera is surrounded by its own reflections instead of emptiness"""
    k = fx.k
    H, W = a.shape[:2]
    tw, th = p['tile'][0] * k, p['tile'][1] * k
    cx, cy = p['center'][0] * k, p['center'][1] * k
    x0, y0 = int(round(cx - tw / 2)), int(round(cy - th / 2))
    x1, y1 = int(round(cx + tw / 2)), int(round(cy + th / 2))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return rgb, a
    pm = np.concatenate([rgb * a, a], -1)
    tile = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
    tile[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = pm[cy0:cy1, cx0:cx1]
    out = np.zeros_like(pm)
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            t_ = tile[:, ::-1] if (i and p['mirror']) else tile
            t_ = t_[::-1] if (j and p['mirror']) else t_
            ox, oy = x0 + i * (x1 - x0), y0 + j * (y1 - y0)
            ax0, ay0, ax1, ay1 = max(0, ox), max(0, oy), min(W, ox + t_.shape[1]), min(H, oy + t_.shape[0])
            if ax0 < ax1 and ay0 < ay1:
                out[ay0:ay1, ax0:ax1] = t_[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox]
    oa = out[..., 3:]
    return out[..., :3] / np.maximum(oa, 1e-6), oa


import functools

N_TAB = 1024                                                # the lattice repeats every 1024 cells (3000+ px at grain 3)


@functools.lru_cache(256)
def _lattice(seed):
    """the random value of every lattice point of one noise, as a table (wraps around)"""
    return np.random.default_rng(seed + 0x9E37).random((N_TAB, N_TAB), dtype=np.float32)


def value_noise(u, v, seed=0):
    """smooth value noise 0..1 on the lattice of whole numbers: a random value per lattice point (a table per seed),
    smoothstep between"""
    T = _lattice(int(seed))
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = (u - iu).astype(np.float32), (v - iv).astype(np.float32)
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    iu = iu.astype(np.int64) & (N_TAB - 1)
    iv = iv.astype(np.int64) & (N_TAB - 1)
    iu1, iv1 = (iu + 1) & (N_TAB - 1), (iv + 1) & (N_TAB - 1)
    a, b, c, d = T[iv, iu], T[iv, iu1], T[iv1, iu], T[iv1, iu1]
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def fbm(u, v, seed=0, octaves=3):
    """fractal value noise 0..1: octaves at doubling frequency and halving weight"""
    out, amp, tot = 0.0, 1.0, 0.0
    for o in range(octaves):
        out = out + amp * value_noise(u * (1 << o), v * (1 << o), seed * 7 + o)
        tot += amp
        amp *= 0.5
    return out / tot


def _bilinear1(a, x, y):
    """a (h, w) sampled at float px coords (centres at i + 0.5), zero outside"""
    h, w = a.shape
    px, py = x - 0.5, y - 0.5
    ix, iy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    fx, fy = px - ix, py - iy

    def at(i, j):
        ok = (i >= 0) & (i < w) & (j >= 0) & (j < h)
        return np.where(ok, a[np.clip(j, 0, h - 1), np.clip(i, 0, w - 1)], 0.0)
    return (at(ix, iy) * (1 - fx) + at(ix + 1, iy) * fx) * (1 - fy) + (at(ix, iy + 1) * (1 - fx) + at(ix + 1, iy + 1) * fx) * fy


_FIELDS = {}                                              # (seed, grain) → (i0, j0, step, the 4 noise fields)


def _print_fields(seed, g, u, v):
    """the print effect's four noise fields (ink density, edge shift x and y, wear) sampled in the layer's own px:
    computed once on a grid of grain/8 px over the part of the layer seen so far (grown when more is seen), then only
    looked up — the noise belongs to the print, not to the frame, so a page flying past costs a lookup per pixel"""
    step = g / 8
    i0n, j0n = int(np.floor(u.min() / step)) - 2, int(np.floor(v.min() / step)) - 2
    i1n, j1n = int(np.ceil(u.max() / step)) + 3, int(np.ceil(v.max() / step)) + 3
    key = (seed, g)
    hit = _FIELDS.get(key)
    if hit is None or not (hit[0] <= i0n and hit[1] <= j0n and i1n <= hit[0] + hit[3].shape[1] and j1n <= hit[1] + hit[3].shape[0]):
        if hit is not None:                                    # grow to hold both, with room to move
            i0n, j0n = min(i0n, hit[0]), min(j0n, hit[1])
            i1n, j1n = max(i1n, hit[0] + hit[3].shape[1]), max(j1n, hit[1] + hit[3].shape[0])
        mx, my = (i1n - i0n) // 4, (j1n - j0n) // 4
        i0n, j0n, i1n, j1n = i0n - mx, j0n - my, i1n + mx, j1n + my
        if (i1n - i0n) * (j1n - j0n) > 16e6:
            return None
        jj, ii = np.mgrid[j0n:j1n, i0n:i1n].astype(np.float64)
        x, y = ii * step, jj * step
        f = np.stack([fbm(x / g, y / g, seed), fbm(x / (g * 0.7), y / (g * 0.7), seed + 11),
                      fbm(x / (g * 0.7), y / (g * 0.7), seed + 23), fbm(x / (g * 2.5), y / (g * 2.5), seed + 37)], -1).astype(np.float32)
        if len(_FIELDS) > 48:
            _FIELDS.clear()
        hit = _FIELDS[key] = (i0n, j0n, step, f)
    i0, j0, st, f = hit
    px, py = u / st - i0, v / st - j0
    ix, iy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    fx, fy = (px - ix)[..., None].astype(np.float32), (py - iy)[..., None].astype(np.float32)
    H, W = f.shape[:2]
    ix, iy = np.clip(ix, 0, W - 2), np.clip(iy, 0, H - 2)
    return (f[iy, ix] * (1 - fx) + f[iy, ix + 1] * fx) * (1 - fy) + (f[iy + 1, ix] * (1 - fx) + f[iy + 1, ix + 1] * fx) * fy


@effect('print', params={'ink': 0.35, 'grain': 3.0, 'rough': 1.0, 'bleed': 0.5, 'wear': 0.0, 'seed': 1},
        grow=lambda p, s: (abs(p['rough']) + 3 * p['bleed']) * s + 2)
def print_(rgb, a, p, fx):
    """ink printed on the surface instead of a sharp layer on top of it: the edge roughened (`rough` layer px, the
    noise of the paper fibres), the ink spread a little (`bleed` px), its density uneven (`ink` 0..1 of it lost in
    blotches `grain` px across), `wear` — patches where the ink is gone. The noise lives in the layer's own px, so it
    sticks to the print while the layer moves, turns or tilts. With blend: multiply the surface shows through."""
    n = int(np.ceil((abs(p['rough']) + 3 * p['bleed']) * fx.scale)) + 2
    rgb, a, n = fx.grow(rgb, a, n)
    H, W = a.shape[:2]
    fx.origin = (fx.origin[0] - n, fx.origin[1] - n)
    uv = fx.content_xy(H, W)
    if uv is None:                                             # no placement known: the raster's own px over the scale
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        uv = ((xs + fx.origin[0]) / fx.scale, (ys + fx.origin[1]) / fx.scale)
    u, v = uv
    g = max(0.2, float(p['grain']))
    F = _print_fields(int(p['seed']), g, u, v)
    if F is None:
        return rgb, a, n
    al = a[..., 0].astype(np.float32)
    if p['rough']:
        r = p['rough'] * fx.scale
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        al = _bilinear1(al, xs + 0.5 + (F[..., 1] - 0.5) * 2 * r, ys + 0.5 + (F[..., 2] - 0.5) * 2 * r)
    if p['bleed'] > 0:
        al = fx.blur(al[..., None], p['bleed'] * fx.scale)[..., 0]
        al = np.clip((al - 0.5) * 1.15 + 0.5, 0, 1) if p['rough'] else al     # a spread edge, not a soft one
    if p['ink']:
        al = al * (1 - p['ink'] * F[..., 0] ** 1.5)
    if p['wear']:
        al = al * np.clip((1 - p['wear'] * 0.6 - F[..., 3]) * 6 + 1, 0, 1)
    return rgb, np.clip(al, 0, 1).astype(np.float32)[..., None], n


print_.kadr_flat = True          # works in the layer's own px: a text drawn flat (turned in 3D) gets it before placing
