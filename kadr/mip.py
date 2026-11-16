"""A picture seen at many sizes at once — a plane going off to the horizon, an endless tiled texture: a mip pyramid
and an anisotropic sampler. Each output pixel looks back through the inverse matrix, measures how many texels its
footprint covers along its long and short axes, and averages up to MAXN samples along the long axis, each from the
pyramid level that matches the short axis (trilinear; Mitchell's bicubic on the full picture where it is magnified).
The near sand stays sharp, the far sand neither shimmers nor turns to mush.

This numpy version is the reference: core/src/wgsl/mip.wgsl does exactly these steps per pixel on the GPU. Both the
renderer on the CPU (render.py) and the one on the GPU (gpu.py) build the pyramid here.

The picture's texels relate to content px by `tmap` = (sx, sy, ox, oy): texel = (content − (ox, oy)) · (sx, sy), in
continuous texel units (texel i covers i..i+1). `wrap`: 0 — the picture once (clamped at its edges; coverage of the
content box 0..cw × 0..ch makes the edge), 1 — repeated, 2 — repeated with mirrored neighbours."""
import functools, math

import numpy as np

MAXN = 8                      # samples along a footprint's long axis at most (a grazing plane: 8× longer than wide)
CLAMP, REPEAT, MIRROR = 0, 1, 2


def pyramid(pm):
    """premultiplied (h, w, 4) float32 → [level 0, 1, …] down to 1 × 1; each level the 2 × 2 box mean of the one
    before (an odd last column / row repeats itself)"""
    levels = [np.ascontiguousarray(pm, np.float32)]
    while levels[-1].shape[0] > 1 or levels[-1].shape[1] > 1:
        a = levels[-1]
        h, w = a.shape[:2]
        if w % 2:
            a = np.concatenate([a, a[:, -1:]], 1)
        if h % 2:
            a = np.concatenate([a, a[-1:]], 0)
        a = (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) * np.float32(0.25)
        levels.append(np.ascontiguousarray(a, np.float32))
    return levels


def packed(levels):
    """the levels one after another as one (n, 4) array — how the GPU holds them"""
    return np.ascontiguousarray(np.concatenate([l.reshape(-1, 4) for l in levels], 0)[None], np.float32)


@functools.lru_cache(8)
def tile_pyramid(path, mtime, crop, grade):
    """a tiled texture's pyramid: the picture (render.load_plain) resampled by Lanczos to power-of-two sides, so every
    level still tiles (an odd level would shift the seams) → (levels, packed levels, (w, h) of the picture, (w, h) of
    level 0)"""
    from PIL import Image
    from .render import load_plain
    im = load_plain(path, crop, grade)
    w, h = im.size
    pw, ph = 1 << max(0, math.ceil(math.log2(w) - 1e-9)), 1 << max(0, math.ceil(math.log2(h) - 1e-9))
    if (pw, ph) != (w, h):
        im = im.resize((pw, ph), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    lv = pyramid(np.concatenate([a[..., :3], a[..., 3:] / 255], -1))
    return lv, packed(lv), (w, h), (pw, ph)


def tile_map(L, size, size0):
    """texel = (content − tile_offset) · s: the picture shows tile_scale content px per picture px"""
    k = L['tile_scale']
    return (size0[0] / (size[0] * k), size0[1] / (size[1] * k), L['tile_offset'][0], L['tile_offset'][1])


def _wrap(i, n, wrap):
    if wrap == REPEAT:
        return np.mod(i, n)
    if wrap == MIRROR:
        m = np.mod(i, 2 * n)
        return np.where(m >= n, 2 * n - 1 - m, m)
    return np.clip(i, 0, n - 1)


def _mitchell(x):
    x = np.abs(x)
    B = C = 1.0 / 3.0
    return np.where(x < 1, ((12 - 9 * B - 6 * C) * x ** 3 + (-18 + 12 * B + 6 * C) * x ** 2 + (6 - 2 * B)) / 6,
                    np.where(x < 2, ((-B - 6 * C) * x ** 3 + (6 * B + 30 * C) * x ** 2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) / 6, 0.0))


def _bicubic(lv, x, y, wrap):
    """Mitchell on level 0 at continuous texel coords (texel i's centre at i + 0.5)"""
    h, w = lv.shape[:2]
    px, py = x - 0.5, y - 0.5
    ix, iy = np.floor(px), np.floor(py)
    fx, fy = px - ix, py - iy
    acc = np.zeros(x.shape + (4,), np.float64)
    for j in range(-1, 3):
        wy = _mitchell(j - fy)
        yy = _wrap(iy.astype(np.int64) + j, h, wrap)
        for i in range(-1, 3):
            wgt = (wy * _mitchell(i - fx))[..., None]
            acc += wgt * lv[yy, _wrap(ix.astype(np.int64) + i, w, wrap)]
    return acc


def _bilinear(lv, x, y, wrap):
    h, w = lv.shape[:2]
    px, py = x - 0.5, y - 0.5
    ix, iy = np.floor(px), np.floor(py)
    fx, fy = (px - ix)[..., None], (py - iy)[..., None]
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    x0, x1 = _wrap(ix, w, wrap), _wrap(ix + 1, w, wrap)
    y0, y1 = _wrap(iy, h, wrap), _wrap(iy + 1, h, wrap)
    return (lv[y0, x0] * (1 - fx) + lv[y0, x1] * fx) * (1 - fy) + (lv[y1, x0] * (1 - fx) + lv[y1, x1] * fx) * fy


def _level(levels, l, x, y, wrap):
    """bilinear on level l (≥ 1) at level-0 texel coords"""
    l = min(l, len(levels) - 1)
    s = float(1 << l)
    return _bilinear(levels[l], x / s, y / s, wrap)


def _sample(levels, x, y, lod, wrap):
    """the picture at texel coords (x, y) for a footprint of 2^lod texels: Mitchell where magnified (lod ≤ 0), blended
    into level 1 over 0..1, trilinear beyond"""
    out = np.zeros(x.shape + (4,), np.float64)
    top = len(levels) - 1
    m0 = lod <= 0
    if m0.any():
        out[m0] = _bicubic(levels[0], x[m0], y[m0], wrap)
    m1 = (lod > 0) & (lod < 1)
    if m1.any():
        f = lod[m1][..., None]
        b = _bicubic(levels[0], x[m1], y[m1], wrap)
        out[m1] = b * (1 - f) + (_level(levels, 1, x[m1], y[m1], wrap) if top >= 1 else b) * f
    m2 = lod >= 1
    if m2.any():
        lo = np.floor(lod[m2]).astype(np.int64)
        f = (lod[m2] - lo)[..., None]
        xs, ys = x[m2], y[m2]
        a = np.zeros(xs.shape + (4,))
        b = np.zeros(xs.shape + (4,))
        for l in np.unique(lo):
            sel = lo == l
            a[sel] = _level(levels, int(l), xs[sel], ys[sel], wrap)
            b[sel] = _level(levels, int(l) + 1, xs[sel], ys[sel], wrap)
        out[m2] = a * (1 - f) + b * f
    return out


def _halfplane(d, nx, ny):
    a = np.maximum(np.abs(nx), np.abs(ny))
    b = np.minimum(np.abs(nx), np.abs(ny))
    w1, w2 = 0.5 * (a - b), 0.5 * (a + b)
    t = w2 - np.abs(d)
    small = t * t / np.maximum(2 * a * b, 1e-30)
    out = np.where(np.abs(d) <= w1, 0.5 + d / np.maximum(a, 1e-30), np.where(d > 0, 1 - small, small))
    return np.where(d <= -w2, 0.0, np.where(d >= w2, 1.0, out))


def _back(inv, X, Y):
    w = inv[6] * X + inv[7] * Y + inv[8]
    return (inv[0] * X + inv[1] * Y + inv[2]) / w, (inv[3] * X + inv[4] * Y + inv[5]) / w


def footprint(J00, J01, J10, J11):
    """a pixel's footprint in texels: (long axis length, short axis length, long axis direction x, y) — the singular
    values of the Jacobian and the long axis in texel space"""
    p = J00 * J00 + J01 * J01
    q = J00 * J10 + J01 * J11
    r = J10 * J10 + J11 * J11
    half = 0.5 * (p + r)
    disc = np.sqrt(np.maximum(0.0, 0.25 * (p - r) ** 2 + q * q))
    s1 = np.sqrt(np.maximum(half + disc, 0.0))
    s2 = np.sqrt(np.maximum(half - disc, 0.0))
    ex = np.where(np.abs(q) > 1e-12, half + disc - r, np.where(p >= r, 1.0, 0.0))
    ey = np.where(np.abs(q) > 1e-12, q, np.where(p >= r, 0.0, 1.0))
    n = np.sqrt(ex * ex + ey * ey)
    return s1, s2, ex / np.maximum(n, 1e-30), ey / np.maximum(n, 1e-30)


def place(levels, cw, ch, invs, box, tmap, wrap):
    """the picture through each inverse matrix (raster px → content px) over the output box (X0, Y0, X1, Y1), the
    samples' premultiplied mean → (rgb (h, w, 3) 0..255, a (h, w, 1) 0..1), as mip.wgsl"""
    X0, Y0, X1, Y1 = box
    ys, xs = np.mgrid[Y0:Y1, X0:X1].astype(np.float64)
    X, Y = xs + 0.5, ys + 0.5
    sx, sy, ox, oy = tmap
    acc = np.zeros(X.shape + (4,))
    for inv in invs:
        u, v = _back(inv, X, Y)
        uxp, vxp = _back(inv, X + 0.5, Y)
        uxm, vxm = _back(inv, X - 0.5, Y)
        uyp, vyp = _back(inv, X, Y + 0.5)
        uym, vym = _back(inv, X, Y - 0.5)
        du = (uxp - uxm, uyp - uym)
        dv = (vxp - vxm, vyp - vym)
        gu, gv = np.hypot(*du), np.hypot(*dv)
        if True:                                               # the content box's edges, exact coverage
            nux, nuy = du[0] / np.maximum(gu, 1e-12), du[1] / np.maximum(gu, 1e-12)
            nvx, nvy = dv[0] / np.maximum(gv, 1e-12), dv[1] / np.maximum(gv, 1e-12)
            cov = (_halfplane(u / gu, nux, nuy) * _halfplane((cw - u) / gu, nux, nuy) *
                   _halfplane(v / gv, nvx, nvy) * _halfplane((ch - v) / gv, nvx, nvy))
        J00, J01, J10, J11 = du[0] * sx, du[1] * sx, dv[0] * sy, dv[1] * sy
        s1, s2, ex, ey = footprint(J00, J01, J10, J11)
        n = np.clip(np.ceil(s1 / np.maximum(s2, 1e-6) - 1e-3), 1, MAXN)
        lod = np.log2(np.maximum(np.maximum(s1 / n, s2), 1e-6))
        tx, ty = (u - ox) * sx, (v - oy) * sy
        on = (cov > 0) & (inv[6] * X + inv[7] * Y + inv[8] > 0)   # w ≤ 0: past the horizon (behind the eye)
        s = np.zeros(X.shape + (4,))
        for k in range(MAXN):
            m = on & (k < n)
            if not m.any():
                break
            off = ((k + 0.5) / n[m] - 0.5) * s1[m]
            s[m] += _sample(levels, tx[m] + off * ex[m], ty[m] + off * ey[m], lod[m], wrap)
        s = s / n[..., None]
        a = np.clip(s[..., 3], 0, 1)
        c = np.clip(s[..., :3], 0, 255 * a[..., None])
        a = a * cov
        acc[..., :3] += c * cov[..., None]                     # (c / a) · (a · cov): premultiplied with the coverage
        acc[..., 3] += a
    acc /= len(invs)
    a = acc[..., 3:]
    return (acc[..., :3] / np.maximum(a, 1e-6)).astype(np.float32), a.astype(np.float32)
