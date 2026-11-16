"""Vector pictures from files (the `svg` layer; an outline for 3D from any picture):

    type: svg
    src: logo.svg          drawn by Skia from the file itself: paths, groups, gradients, strokes — sharp at any size and
                           zoom, because it is drawn anew at the size it shows at (a layer like any: it moves, turns in
                           3D, takes effects and mattes)
    size: [w, h]           the box it is fitted into, keeping its proportions (default: the file's own size)
    color: "#ffffff"       its shape in one colour (an icon drawn black made white), keeping its transparency

    in a 3D scene:  {shape: extrude, src: logo.svg | logo.png, height: 30, depth: 6, bevel: .6}
                           the file's shape — wherever it is not transparent — given thickness (kadr/extrude.py)"""
import functools, math, os, re

import numpy as np
from PIL import Image


@functools.lru_cache(16)
def _dom(path, mtime):
    import skia
    dom = skia.SVGDOM.MakeFromStream(skia.FILEStream(path))    # from the file itself: a stream over bytes Python has let go of reads rubbish
    if dom is None:
        raise ValueError(f'{os.path.basename(path)}: не читается как SVG')
    sz = dom.containerSize()
    w, h = float(sz.width()), float(sz.height())
    if w <= 0 or h <= 0:                                       # no width / height: the viewBox says how large it is
        m = re.search(r'viewBox\s*=\s*["\']\s*([-\d.eE]+)[\s,]+([-\d.eE]+)[\s,]+([-\d.eE]+)[\s,]+([-\d.eE]+)', open(path, encoding='utf-8', errors='replace').read())
        w, h = (float(m.group(3)), float(m.group(4))) if m else (100.0, 100.0)
        dom.setContainerSize(skia.Size(w, h))
    return dom, w, h


def size_of(path):
    """the file's own size, px"""
    _, w, h = _dom(path, os.path.getmtime(path))
    return w, h


def picture(path, w, h, sc, color=None):
    """the file drawn into a w × h box (fitted, centred) at sc px per content px → premultiplied RGBA uint8"""
    import skia
    dom, sw, sh = _dom(path, os.path.getmtime(path))
    W, H = max(1, int(math.ceil(w * sc))), max(1, int(math.ceil(h * sc)))
    surf = skia.Surface.MakeRaster(skia.ImageInfo.Make(W, H, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType))
    c = surf.getCanvas()
    c.clear(skia.Color4f(0, 0, 0, 0))
    k = min(w / sw, h / sh)
    if color is not None:                                      # its shape in one colour: drawn into a layer that keeps only the alpha
        c.saveLayer(None, skia.Paint(ColorFilter=skia.ColorFilters.Blend(skia.Color(*[int(round(v)) for v in color[:3]]), skia.BlendMode.kSrcIn)))
    c.scale(sc, sc)
    c.translate((w - sw * k) / 2, (h - sh * k) / 2)
    c.scale(k, k)
    dom.render(c)
    if color is not None:
        c.restore()
    return surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)


def parse(F, L, lid, out):
    src = F.resolve(str(L.get('src', '')))
    out['src'] = src
    w, h = 100.0, 100.0
    if not os.path.isfile(src):
        F.issue('error', f"нет файла {L.get('src')}", lid)
    else:
        try:
            w, h = size_of(src)
        except (ValueError, OSError) as e:
            F.issue('error', str(e), lid)
    s = L.get('size')
    if s is not None:
        if not (isinstance(s, (list, tuple)) and len(s) == 2):
            F.issue('error', 'size: [w, h] — рамка, в которую вписывается рисунок', lid)
        else:
            w, h = float(s[0]), float(s[1])
    out['psize'] = (w, h)
    out['cell'] = 16.0
    out['tint'] = None
    if L.get('color') is not None:
        from .doc import color
        try:
            out['tint'] = color(L['color'])
        except (ValueError, TypeError) as e:
            F.issue('error', str(e), lid)


def render(L, sc):
    w, h = L['psize']
    sc = min(sc, math.sqrt(40e6 / max(1.0, w * h)))
    a = picture(L['src'], w, h, sc, L['tint'])
    if not a[..., 3].any():
        return None
    return 0.0, 0.0, a.shape[1] / sc, a.shape[0] / sc, Image.fromarray(np.ascontiguousarray(a), 'RGBa')


# ---------------- an outline out of any picture ----------------
def mask(path, long_side=2048):
    """how opaque the file is, drawn large: an SVG by Skia, anything else by its own alpha (a picture with none: by how
    dark it is) → (h, w) float 0 … 1"""
    if path.lower().endswith('.svg'):
        w, h = size_of(path)
        k = long_side / max(w, h)
        return picture(path, w, h, k)[..., 3].astype(np.float32) / 255
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA', 'PA') or 'transparency' in im.info:
        a = np.asarray(im.convert('RGBA'))[..., 3].astype(np.float32) / 255
    else:
        a = 1 - np.asarray(im.convert('L')).astype(np.float32) / 255
    k = long_side / max(a.shape)
    if abs(k - 1) > .05:
        a = np.asarray(Image.fromarray(a, 'F').resize((max(2, int(round(a.shape[1] * k))), max(2, int(round(a.shape[0] * k)))), Image.LANCZOS))
    return np.clip(a, 0, 1)


def _table():
    """for each of a cell's sixteen states: the pairs of edges (0 top, 1 right, 2 bottom, 3 left) a line runs between,
    each turned so that the inside is always on the same hand — worked out, not written by hand: with one pair turned
    the wrong way two pieces began at the same point, one was lost, and every outline fell into scraps"""
    mid = {0: (.5, 0), 1: (1, .5), 2: (.5, 1), 3: (0, .5)}
    corner = [(0, 0), (1, 0), (1, 1), (0, 1)]                  # top-left 1, top-right 2, bottom-right 4, bottom-left 8
    edges_of = {0: (3, 0), 1: (0, 1), 2: (1, 2), 3: (2, 3)}    # the two edges that meet at each corner
    out = {}
    for code in range(1, 15):
        inside = [k for k in range(4) if code >> k & 1]
        if len(inside) == 2 and (inside[1] - inside[0]) == 2:  # a saddle: each inside corner cut off by itself
            pairs = [(edges_of[k], k) for k in inside]
        else:
            cut = [e for e in range(4) if ((code >> e) & 1) != ((code >> ((e + 1) % 4)) & 1)]     # edge e joins corner e and e + 1
            pairs = [((cut[0], cut[1]), inside[0])]
        segs = []
        for (e0, e1), k in pairs:
            p, q, c = np.array(mid[e0]), np.array(mid[e1]), np.array(corner[k])
            turn = (q[0] - p[0]) * (c[1] - p[1]) - (q[1] - p[1]) * (c[0] - p[0])
            segs.append((e0, e1) if turn > 0 else (e1, e0))
        out[code] = segs
    return out


def contours(m, level=.5):
    """the closed lines where the mask crosses `level` (marching squares, the crossing found between the two pixels it
    lies between) → [(n, 2) float] in the mask's px, y down"""
    m = np.pad(np.asarray(m, np.float32), 1)
    h, w = m.shape
    a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, 1:], m[1:, :-1]         # the corners of each cell: top-left, top-right, bottom-right, bottom-left
    code = (a > level) * 1 + (b > level) * 2 + (c > level) * 4 + (d > level) * 8
    ys, xs = np.nonzero((code > 0) & (code < 15))

    def cross(v0, v1):
        return (level - v0) / np.where(np.abs(v1 - v0) > 1e-9, v1 - v0, 1e-9)
    A, B, C, Dv = a[ys, xs], b[ys, xs], c[ys, xs], d[ys, xs]
    top = np.stack([xs + cross(A, B), ys.astype(np.float32)], 1)
    right = np.stack([xs + 1.0, ys + cross(B, C)], 1)
    bottom = np.stack([xs + cross(Dv, C), ys + 1.0], 1)
    left = np.stack([xs.astype(np.float32), ys + cross(A, Dv)], 1)
    E = (top, right, bottom, left)
    TABLE = _table()
    # a crossing is named by the grid edge it lies on, not by where it is: two crossings a thousandth of a pixel apart
    # (a line passing by a pixel's corner) are still two — named by their rounded place they were one, and outlines broke
    def name(e, y, x):
        return (0, y + (e == 2), x) if e in (0, 2) else (1, y, x + (e == 1))
    segs = {}
    cd = code[ys, xs]
    for k in range(len(ys)):
        y, x = int(ys[k]), int(xs[k])
        for e0, e1 in TABLE[int(cd[k])]:
            segs[name(e0, y, x)] = (E[e0][k], name(e1, y, x))
    out = []
    while segs:
        start, (p, nxt) = next(iter(segs.items()))
        del segs[start]
        loop = [p]
        while nxt != start:
            got = segs.pop(nxt, None)
            if got is None:                                    # an open end (cannot be, the mask is padded): drop the piece
                loop = None
                break
            loop.append(got[0])
            nxt = got[1]
        if loop is not None and len(loop) >= 3:
            out.append(np.array(loop, float) - 1.0)            # the padding taken off
    return out


def simplify(p, tol):
    """a closed line with the points it does not need taken out (none of them farther than `tol` from the line left)"""
    n = len(p)
    if n < 8:
        return p
    far = int(np.argmax(np.hypot(*(p - p[0]).T)))
    keep = np.zeros(n + 1, bool)
    q = np.concatenate([p, p[:1]])
    keep[[0, far, n]] = True
    stack = [(0, far), (far, n)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = q[i], q[j]
        ab = b - a
        ln = math.hypot(*ab)
        seg = q[i + 1:j] - a
        d = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / ln if ln > 1e-9 else np.hypot(*seg.T)
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return q[:-1][keep[:-1]]


def outline(path, height):
    """the shape of a file — wherever it is not transparent — as a Skia path, y up, centred, `height` units tall"""
    import skia
    m = mask(path)
    p = skia.Path()
    p.setFillType(skia.PathFillType.kEvenOdd)
    for c in contours(m):
        c = simplify(c, .6)
        if len(c) >= 3:
            p.moveTo(float(c[0, 0]), float(c[0, 1]))
            for x, y in c[1:]:
                p.lineTo(float(x), float(y))
            p.close()
    b = p.getBounds()
    if b.height() <= 0:
        raise ValueError(f'{os.path.basename(path)}: в файле нечего обвести (он весь прозрачный?)')
    k = height / b.height()
    mt = skia.Matrix()
    mt.setTranslate(-b.centerX(), -b.centerY())
    mt.postScale(k, -k)
    out = skia.Path()
    p.transform(mt, out)
    return out
