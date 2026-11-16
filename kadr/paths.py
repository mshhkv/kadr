"""Curves as data: what a shape layer's outline is at time t, whatever the document made it from —

    d / points       an SVG path or a list of points; given as keys {time: …} they morph (After Effects: animating Path)
    shape: text      the outlines of a text (font, font_size, tracking, align): a title that can be stroked and drawn on
                     with `trim`, filled with a gradient, morphed, repeated — «Create Shapes from Text»
    along            the text's glyphs set along a curve (Path Text); `along_offset` runs them down it
    paths            several outlines merged: each {d | points | shape + size + at} with its own `op`
                     (union, subtract, intersect, xor) — Merge Paths
    offset           the outline grown (> 0) or shrunk (< 0) by that many px, corners round — Offset Paths
    repeat           copies, each moved, turned and scaled a step further — Repeater: {count, offset, rotation, scale,
                     anchor, opacity: [first, last]}

in that order: outline → merged → offset → repeated. Everything numeric animates. A layer's `mask: {d…, feather}` is
such a shape too (doc.py makes it a hidden matte layer)."""
import functools, math
import numpy as np

N = 256                         # points a contour is sampled to while it morphs (at a key it is the key's own curve)
OPS = ('union', 'subtract', 'intersect', 'xor')


# ---------------- outlines ----------------
def base(L, t):
    """the outline before `offset` and `repeat`"""
    import skia
    from .render import svg_path
    if L.get('paths'):
        p = None
        for item in L['paths']:
            q = _item_path(item, t)
            p = q if p is None else merged(p, q, item['op'])
        p = p or skia.Path()
    else:
        p = _item_path(L, t)
    if L.get('fill_rule') == 'evenodd':
        p.setFillType(skia.PathFillType.kEvenOdd)
    return p


def _item_path(S, t):
    """one outline: a rect / ellipse, points, an SVG path (plain or morphing), a text"""
    import skia
    from .render import svg_path
    kind = S['shape']
    p = skia.Path()
    if kind in ('rect', 'ellipse'):
        w, h = S['size'].at(t)
        if kind == 'ellipse':
            p.addOval(skia.Rect.MakeWH(w, h))
        elif S['radius'].at(t) > 0:
            r = min(S['radius'].at(t), w / 2, h / 2)                # at most a pill (CSS clamps the same way)
            p.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeWH(w, h), r, r))
        else:
            p.addRect(skia.Rect.MakeWH(w, h))
    elif kind == 'text':
        p = text_path(S, t)
    elif S.get('keys'):                                        # a morph: d or points given at several times
        p = morph_at(S, t)
    elif kind in ('line', 'polyline', 'polygon'):
        p = _points_path(S['points'], S['closed'])
    elif kind == 'path' and S['d']:
        p = svg_path(S['d'])
    at = S.get('at')
    if at is not None:                                         # a part of `paths` set down at its own place
        x, y = at.at(t)
        p.offset(float(x), float(y))
    return p


def _points_path(pts, closed):
    import skia
    p = skia.Path()
    if pts:
        p.moveTo(*pts[0])
        for q in pts[1:]:
            p.lineTo(*q)
        if closed:
            p.close()
    return p


def final(L, t):
    """the layer's whole outline (every copy of the repeater in one path): its bounds are the layer's"""
    p = shaped(L, t)
    cs = copies(L, t)
    if cs is None:
        return p
    import skia
    out = skia.Path()
    for M, _ in cs:
        out.addPath(p, M)
    out.setFillType(p.getFillType())
    return out


def shaped(L, t):
    """the outline one copy has: merged and offset, not repeated"""
    p = base(L, t)
    off = L['offset'].at(t) if L.get('offset') is not None else 0.0
    return grown(p, off) if abs(off) > 1e-6 else p


# ---------------- Merge Paths, Offset Paths ----------------
def merged(a, b, op):
    import skia
    code = {'union': skia.PathOp.kUnion_PathOp, 'subtract': skia.PathOp.kDifference_PathOp, 'intersect': skia.PathOp.kIntersect_PathOp,
            'xor': skia.PathOp.kXOR_PathOp}[op]
    r = skia.Op(a, b, code)
    return r if r is not None else a


def grown(p, d):
    """the outline moved out by d px (in, for d < 0), corners round: the stroke of width 2|d| joined to it or cut from it"""
    import skia
    paint = skia.Paint(Style=skia.Paint.kStroke_Style, StrokeWidth=2 * abs(d), StrokeJoin=skia.Paint.kRound_Join)
    edge = skia.Path()
    if not paint.getFillPath(p, edge):
        return p
    r = skia.Op(p, edge, skia.PathOp.kUnion_PathOp if d > 0 else skia.PathOp.kDifference_PathOp)
    return r if r is not None else p


# ---------------- Repeater ----------------
def copies(L, t):
    """[(skia.Matrix, opacity)] of the repeater's copies in the order they are drawn, or None without one"""
    rp = L.get('repeat')
    if not rp:
        return None
    import skia
    n = max(0.0, rp['count'].at(t))
    full, part = int(math.floor(n + 1e-9)), n - math.floor(n + 1e-9)
    total = full + (1 if part > 1e-6 else 0)
    dx, dy = rp['offset'].at(t)
    rot, sc = rp['rotation'].at(t), rp['scale'].at(t)
    ax, ay = rp['anchor'].at(t)
    o0, o1 = rp['opacity'].at(t)
    out = []
    for i in range(total):
        m = skia.Matrix()
        m.setTranslate(-ax, -ay)
        s = sc ** i
        m.postScale(s, s)
        m.postRotate(rot * i)
        m.postTranslate(ax + dx * i, ay + dy * i)
        u = i / (total - 1) if total > 1 else 0.0
        out.append((m, (o0 + (o1 - o0) * u) * (part if i == full else 1.0)))
    return out


# ---------------- morphing ----------------
def contours(p, n=N):
    """every contour of a path as n points evenly spaced along it: [(points (n, 2), closed)]"""
    import skia
    out = []
    m = skia.PathMeasure(p, False)
    while True:
        ln = m.getLength()
        if ln > 0:
            closed = m.isClosed()
            ds = np.arange(n) * (ln / n if closed else ln / (n - 1))
            pts = np.array([tuple(m.getPosTan(float(min(d, ln)))[0]) for d in ds], float)
            out.append((pts, closed))
        if not m.nextContour():
            break
    return out


def _area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _paired(a, b):
    """b's points set against a's: the same turn direction, and the start that keeps the points' travel shortest"""
    (pa, ca), (pb, cb) = a, b
    if ca and cb:
        if _area(pa) * _area(pb) < 0:
            pb = pb[::-1]
        # the best circular shift: Σ|a_i − b_(i+k)|² is smallest where Σ a_i · b_(i+k) is largest — one FFT
        fa, fb = np.fft.fft(pa[:, 0] + 1j * pa[:, 1]), np.fft.fft(pb[:, 0] + 1j * pb[:, 1])
        k = int(np.argmax(np.real(np.fft.ifft(np.conj(fa) * fb))))
        pb = np.roll(pb, -k, axis=0)
    elif np.sum((pa - pb) ** 2) > np.sum((pa - pb[::-1]) ** 2):
        pb = pb[::-1]
    return pa, pb, ca and cb


def cubics(p):
    """a path as cubic segments: [(start (2,), segments (n, 3, 2): control 1, control 2, end; straight (n,) bool; closed)]
    per contour — lines, quadratics and conics (an oval's arcs) turned into the cubics that draw the same; a closing
    edge that has length is a segment too"""
    import skia
    out, it = [], skia.Path.Iter(p, False)
    start, segs, flat = None, [], []

    def done(closed):
        nonlocal start, segs, flat
        if start is not None and segs:
            out.append((np.array(start, float), np.array(segs, float), np.array(flat, bool), closed))
        start, segs, flat = None, [], []
    while True:
        verb, pts = it.next()
        q = [(v.x(), v.y()) for v in pts]
        if verb == skia.Path.kMove_Verb:
            done(False)
            start = q[0]
        elif verb == skia.Path.kLine_Verb:
            a, b = np.array(q[0]), np.array(q[1])
            segs.append((a + (b - a) / 3, a + (b - a) * 2 / 3, b)); flat.append(True)
        elif verb in (skia.Path.kQuad_Verb, skia.Path.kConic_Verb):
            a, c, b = map(np.array, q)
            k = 2 / 3 if verb == skia.Path.kQuad_Verb else (4 / 3) * it.conicWeight() / (1 + it.conicWeight())
            segs.append((a + (c - a) * k, b + (c - b) * k, b)); flat.append(False)
        elif verb == skia.Path.kCubic_Verb:
            segs.append(tuple(map(np.array, q[1:]))); flat.append(False)
        elif verb == skia.Path.kClose_Verb:
            if start is not None and segs:
                a, b = np.array(segs[-1][2]), np.array(start)
                if np.hypot(*(b - a)) > 1e-6:
                    segs.append((a + (b - a) / 3, a + (b - a) * 2 / 3, b)); flat.append(True)
            done(True)
        else:
            done(False)
            break
    return out


def _same_build(A, B):
    return len(A) == len(B) and all(len(a[1]) == len(b[1]) and a[3] == b[3] for a, b in zip(A, B))


@functools.lru_cache(64)
def _built(kind, src, closed):
    from .render import svg_path
    return cubics(svg_path(src) if kind == 'd' else _points_path(src, closed))


@functools.lru_cache(64)
def _sampled(kind, src, closed):
    """the contours of a key's outline (src: an SVG string or a tuple of points), largest first"""
    from .render import svg_path
    p = svg_path(src) if kind == 'd' else _points_path(src, closed)
    cs = contours(p)
    return sorted(cs, key=lambda c: -abs(_area(c[0])) if c[1] else 0.0) if len(cs) > 1 else cs


def morph_at(S, t):
    """the outline between two keys; at a key — the key's own curve, exactly. Two keys built alike (the same number
    of contours and of segments in each — After Effects' case: the same vertices moved) go vertex to vertex and
    handle to handle, so a rectangle stays a rectangle on its way to another. Otherwise each contour is sampled
    to N points that move straight to their partners; spare contours grow out of (shrink into) a point"""
    import skia
    from .doc import ease
    from .render import svg_path
    keys, kind = S['keys'], S['key_kind']
    make = (lambda v: svg_path(v)) if kind == 'd' else (lambda v: _points_path(v, S['closed']))
    tn = t + 6e-4                                              # keys are written to 1 ms: a frame never lands a hair early
    if tn <= keys[0][0]:
        return make(keys[0][1])
    if tn >= keys[-1][0]:
        return make(keys[-1][1])
    for (t0, v0, e), (t1, v1, _) in zip(keys, keys[1:]):
        if t0 <= tn < t1:
            u = ease(min(1.0, max(0.0, (t - t0) / (t1 - t0))), e)   # the frame's own time: at a key, exactly the key
            if u <= 1e-6:
                return make(v0)
            p = skia.Path()
            A, B = _built(kind, v0, S['closed']), _built(kind, v1, S['closed'])
            if _same_build(A, B):
                for (sa, ga, fa, closed), (sb, gb, fb, _) in zip(A, B):
                    st, sg = sa + (sb - sa) * u, ga + (gb - ga) * u
                    p.moveTo(float(st[0]), float(st[1]))
                    for (c1, c2, e_), straight in zip(sg, fa & fb):
                        if straight:
                            p.lineTo(float(e_[0]), float(e_[1]))
                        else:
                            p.cubicTo(float(c1[0]), float(c1[1]), float(c2[0]), float(c2[1]), float(e_[0]), float(e_[1]))
                    if closed:
                        p.close()
                return p
            A, B = _sampled(kind, v0, S['closed']), _sampled(kind, v1, S['closed'])
            for i in range(max(len(A), len(B))):
                a = A[i] if i < len(A) else (np.repeat(B[i][0].mean(0, keepdims=True), N, 0), B[i][1])
                b = B[i] if i < len(B) else (np.repeat(A[i][0].mean(0, keepdims=True), N, 0), A[i][1])
                pa, pb, closed = _paired(a, b)
                pts = pa + (pb - pa) * u
                p.moveTo(float(pts[0, 0]), float(pts[0, 1]))
                for x, y in pts[1:]:
                    p.lineTo(float(x), float(y))
                if closed:
                    p.close()
            return p
    return make(keys[-1][1])


# ---------------- text as outlines ----------------
@functools.lru_cache(4096)
def _glyph(fontname, g):
    """one glyph's outline in font units (y down), its advance and the font's units per em"""
    import skia, freetype as ft
    from .render import glyph_face
    f, gid = glyph_face(fontname, g)
    f.load_glyph(gid, ft.FT_LOAD_NO_SCALE | ft.FT_LOAD_NO_BITMAP)
    p = skia.Path()
    o = f.glyph.outline

    def move(a, _):
        if p.countPoints():
            p.close()                                          # the contour before this one
        p.moveTo(a.x, -a.y)

    def line(a, _):
        p.lineTo(a.x, -a.y)

    def conic(c, a, _):
        p.quadTo(c.x, -c.y, a.x, -a.y)

    def cubic(c1, c2, a, _):
        p.cubicTo(c1.x, -c1.y, c2.x, -c2.y, a.x, -a.y)
    if o.n_points:
        o.decompose(None, move_to=move, line_to=line, conic_to=conic, cubic_to=cubic)
        p.close()
    return p, f.glyph.advance.x, f.units_per_EM


def text_path(S, t):
    """a text's outlines laid out as the text layer lays it (HarfBuzz: kerning, features, fallback fonts) — straight,
    or with every glyph set on the curve `along` at its place down the line, turned to the curve (Path Text)"""
    import skia
    from .render import text_layout, svg_path
    size = S['font_size'].at(t)
    glyphs, (w, h) = text_layout(S['font'], size, S['text'], S['align'], S['line_height'], S['tracking'], S.get('features', ''))
    out = skia.Path()
    curve = None
    if S.get('along'):
        curve = skia.PathMeasure(svg_path(S['along']), False)
        ln = curve.getLength()
        start = S['along_offset'].at(t)
        base_y = glyphs[0][2] if glyphs else 0.0
    for g, x, y, _ in glyphs:
        gp, adv, upem = _glyph(S['font'], g)
        k = size / upem
        m = skia.Matrix()
        m.setScale(k, k)
        if curve is None:
            m.postTranslate(x, y)
        else:
            half = adv * k / 2
            s = start + x + half                                # where the glyph's middle rides the curve
            if curve.isClosed() and ln > 0:
                s %= ln
            if not (0 <= s <= ln):
                continue                                       # not on the curve yet / already off it
            pos, tan = curve.getPosTan(float(s))
            m.postTranslate(-half, y - base_y)                  # its middle on the baseline at the origin
            m.postRotate(math.degrees(math.atan2(tan.y(), tan.x())))
            m.postTranslate(pos.x(), pos.y())
        out.addPath(gp, m)
    return out


# ---------------- what the renderers ask ----------------
def stamp(L, t):
    """everything the outline depends on at t, hashable: the same stamp — the same outline (a cache's key)"""
    from .doc import Prop

    def val(v):
        if isinstance(v, Prop):
            return v.at(t)
        if isinstance(v, dict):
            return tuple((k, val(x)) for k, x in sorted(v.items(), key=lambda kv: str(kv[0])) if k not in ('raw', 'parent'))
        if isinstance(v, (list, tuple)):
            return tuple(val(x) for x in v)
        return v
    seg = None
    if L.get('keys'):
        ks = L['keys']
        seg = (round(t, 6),) if ks[0][0] < t < ks[-1][0] else (t >= ks[-1][0],)
    return (L['shape'], val(L.get('size')), val(L.get('radius')), val(L.get('points')), L.get('d'), L.get('closed'), L.get('fill_rule'), seg,
            val(L.get('paths')) if L.get('paths') else None, val(L.get('offset')), val(L.get('repeat')), L.get('text'), L.get('font'),
            val(L.get('font_size')), L.get('along'), val(L.get('along_offset')))


def animated(L):
    """True when the outline itself changes with time (a morph, an animated offset, repeater, text size, run along)"""
    from .doc import Prop

    def anim(v):
        if isinstance(v, Prop):
            return v.animated
        if isinstance(v, dict):
            return any(anim(x) for k, x in v.items() if k not in ('raw', 'parent'))
        if isinstance(v, (list, tuple)):
            return any(anim(x) for x in v)
        return False
    return bool(L.get('keys') and len(L['keys']) > 1) or any(anim(L.get(k)) for k in ('paths', 'offset', 'repeat', 'font_size', 'along_offset', 'radius'))
