"""Flat outlines given thickness (kadr/scene3d.py: `shape: text`, `shape: extrude`): 3D lettering and logos.

An outline (the glyphs of a text, an SVG path) is cut into triangles for its faces, walled round its edge, and — with
`bevel` — its edge is rounded off: the face is the outline drawn in by the bevel's size, and between the wall and the
face lie a few rings of the outline drawn in less and less, each a little higher, so the edge catches light the way a
cast letter's does. Insets are made by Skia (the outline less its own thick stroke), the triangles by the core."""
import math

import numpy as np


def flatten(path, tol):
    """a Skia path's closed contours as polygons: curves cut finely enough to stay within `tol` → [(n, 2) float]"""
    import skia
    out, cur = [], []

    def close():
        if len(cur) >= 3:
            p = np.array(cur, float)
            keep = np.ones(len(p), bool)
            keep[1:] = np.hypot(*(p[1:] - p[:-1]).T) > tol * .05
            p = p[keep]
            if len(p) > 3 and np.hypot(*(p[0] - p[-1])) <= tol * .05:
                p = p[:-1]
            if len(p) >= 3 and abs(_area(p)) > tol * tol:
                out.append(p)
        cur.clear()
    it = skia.Path.Iter(path, True)
    while True:
        v, pts = it.next()
        if v == skia.Path.kDone_Verb:
            break
        q = [np.array([a.x(), a.y()]) for a in pts]
        if v == skia.Path.kMove_Verb:
            close()
            cur.append(q[0])
        elif v == skia.Path.kLine_Verb:
            cur.append(q[1])
        elif v in (skia.Path.kQuad_Verb, skia.Path.kConic_Verb):
            w = it.conicWeight() if v == skia.Path.kConic_Verb else 1.0
            n = int(max(2, math.ceil(math.sqrt(np.hypot(*(q[0] - 2 * q[1] + q[2])) / (4 * tol)))))
            for k in range(1, n + 1):
                t = k / n
                a, b, c = (1 - t) ** 2, 2 * t * (1 - t) * w, t * t
                cur.append((a * q[0] + b * q[1] + c * q[2]) / (a + b + c))
        elif v == skia.Path.kCubic_Verb:
            dd = max(np.hypot(*(q[0] - 2 * q[1] + q[2])), np.hypot(*(q[1] - 2 * q[2] + q[3])))
            n = int(max(2, math.ceil(math.sqrt(3 * dd / (4 * tol)))))
            for k in range(1, n + 1):
                t = k / n
                cur.append((1 - t) ** 3 * q[0] + 3 * t * (1 - t) ** 2 * q[1] + 3 * t * t * (1 - t) * q[2] + t ** 3 * q[3])
        elif v == skia.Path.kClose_Verb:
            close()
    close()
    return out


def _area(p):
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _inside(pt, poly):
    """is the point inside the polygon (a ray to the right crosses its edge an odd number of times)"""
    x, y = pt
    a, b = poly, np.roll(poly, -1, 0)
    hit = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / (b[:, 1] - a[:, 1] + 1e-300) + a[:, 0])
    return bool(hit.sum() % 2)


def nest(cs):
    """which contours are solid and which are holes in which: depth = how many others a contour lies inside →
    [(outer, [its holes])] by number, and every contour's depth"""
    n = len(cs)
    inside = [[j for j in range(n) if j != i and _inside(cs[i][0], cs[j])] for i in range(n)]
    depth = [len(v) for v in inside]
    groups = {i: [] for i in range(n) if depth[i] % 2 == 0}
    for i in range(n):
        if depth[i] % 2 == 1:
            parents = [j for j in inside[i] if depth[j] == depth[i] - 1]
            if parents:
                groups[min(parents, key=lambda j: abs(_area(cs[j])))].append(i)
    return list(groups.items()), depth


def cut(cs):
    """the flat shape of these contours as triangles → (which contour and which of its points each vertex is (n, 2),
    triangles (m, 3) counter-clockwise)"""
    import kadr_core as kc
    groups, _ = nest(cs)
    who, tris = [], []
    for outer, holes in groups:
        ring = [outer] + holes
        base = len(who)
        flat, starts, off = [], [], 0
        for k, ci in enumerate(ring):
            if k:
                starts.append(off)
            flat += cs[ci].ravel().tolist()
            who += [(ci, i) for i in range(len(cs[ci]))]
            off += len(cs[ci])
        t = np.array(kc.triangulate(flat, starts), np.int64).reshape(-1, 3)
        if len(t):
            P = np.concatenate([cs[ci] for ci in ring])
            a, b, c = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
            # all turned the same way, by their sum: where an inset's round corner ends on the line of the outline's own
            # edge three points lie in a row, and a triangle of no area between them has no turn of its own to go by —
            # flipped alone it tore the surface open
            if float(((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum()) < 0:
                t = t[:, ::-1]
            t = _mend(t, [len(cs[ci]) for ci in ring])
            tris.append(t + base)
    return np.array(who, np.int64).reshape(-1, 2), (np.concatenate(tris) if tris else np.zeros((0, 3), np.int64))


def _mend(t, sizes):
    """the triangles of a shape with the slivers ear clipping leaves out put back. Three points in a row (an inset's
    round corner ends on the line of the outline's own edge) make an ear of no area, and it is skipped: flat, nothing
    is lost — but the rings are at different heights, and in space that sliver is a slit through the surface. Every
    edge inside must be walked both ways, every edge of a contour once: what is missing forms small loops, and each
    is closed with a fan. `sizes`: how many points each ring has, in the order the points are numbered"""
    have = set()
    for a, b, c in t.tolist():
        have.update(((a, b), (b, c), (c, a)))
    rim, o = [], 0
    for n in sizes:                                            # each ring's edges, the way round the triangles walk them
        fw = [(o + i, o + (i + 1) % n) for i in range(n)]
        bw = [(b, a) for a, b in fw]
        rim += fw if sum(e in have for e in fw) >= sum(e in have for e in bw) else bw
        o += n
    rim = set(rim)
    need = {(b, a) for a, b in have if (b, a) not in have and (a, b) not in rim} | {e for e in rim if e not in have}
    extra = []
    nxt = {}
    for a, b in need:
        nxt.setdefault(a, []).append(b)
    while nxt:
        start = next(iter(nxt))
        loop, cur = [start], start
        while True:
            outs = nxt.get(cur)
            if not outs:
                loop = None
                break
            b = outs.pop()
            if not outs:
                del nxt[cur]
            if b == start:
                break
            loop.append(b)
            cur = b
            if len(loop) > 64:
                loop = None
                break
        if loop and len(loop) >= 3:
            extra += [(loop[0], loop[i], loop[i + 1]) for i in range(1, len(loop) - 1)]
    return np.concatenate([t, np.array(extra, np.int64).reshape(-1, 3)]) if extra else t


def _normals(cs):
    """for every point of every contour: the flat direction out of the shape there (smooth), and its two edges' own"""
    _, depth = nest(cs)
    out = []
    for p, d in zip(cs, depth):
        e = np.roll(p, -1, 0) - p                              # edge i: point i → i + 1
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        sgn = (1.0 if _area(p) > 0 else -1.0) * (1.0 if d % 2 == 0 else -1.0)
        en = np.stack([e[:, 1], -e[:, 0]], 1) * sgn            # out of the solid, for each edge
        prev = np.roll(en, 1, 0)
        vn = en + prev
        ln = np.linalg.norm(vn, axis=1, keepdims=True)
        vn = np.where(ln > 1e-9, vn / np.maximum(ln, 1e-12), en)
        sharp = (en * prev).sum(1) < math.cos(math.radians(35))
        out.append((vn, en, sharp, sgn))
    return out


def mesh(path, depth, bevel=0.0, steps=3, tol=None):
    """the path (y up, already in the scene's units, centred as wanted) pushed `depth` thick along z, centred on z = 0 →
    (vertices (n, 8): place, normal, where on a picture; triangles' point numbers)"""
    from .paths import grown
    import skia
    b = path.getBounds()
    if b.width() <= 0 or b.height() <= 0:
        return np.zeros((0, 8), np.float32), np.zeros(0, np.uint32)
    tol = tol or max(b.width(), b.height()) / 1200
    # Skia's outline work (joining, insetting) is exact to a fraction of a pixel — of its own units. A letter a few
    # scene units tall came back with insets that touched the outline they were drawn in from, and the surface did not
    # close. So the outline is worked on large (its longer side 20 000 units) and the mesh brought back to size
    K = 20000.0 / max(b.width(), b.height())
    big = skia.Path()
    path.transform(skia.Matrix.Scale(K, K), big)
    path, depth, bevel, tol = big, depth * K, float(bevel) * K, tol * K
    b = path.getBounds()
    path = skia.Op(path, path, skia.PathOp.kUnion_PathOp) or path     # overlapping contours (some fonts) become one outline
    bevel = max(0.0, min(float(bevel), depth / 2 - 1e-6))
    n = max(1, int(steps)) if bevel > 0 else 0
    levels = []                                                # (contours, z, tilt of the surface there: 0 a wall, 90° a face)
    for j in range(n + 1):
        ph = (j / n) * math.pi / 2 if n else math.pi / 2
        inset = bevel * (1 - math.cos(ph)) if n else 0.0
        cs = flatten(grown(path, -inset) if inset > 1e-9 else path, tol)
        levels.append((cs, depth / 2 - bevel + bevel * math.sin(ph) if n else depth / 2, ph))
    x0, y0, w, h = b.left(), b.top(), b.width(), b.height()
    V, T = [], []

    def add(verts, tris):
        base = sum(len(v) for v in V)
        V.append(np.asarray(verts, np.float32).reshape(-1, 8))
        T.append(np.asarray(tris, np.int64).reshape(-1, 3) + base)

    def uv(p):
        return np.stack([(p[:, 0] - x0) / w, (p[:, 1] - y0) / h], 1)

    # the faces
    cs, z, _ = levels[-1]
    if cs:
        who, tris = cut(cs)
        P = np.array([cs[c][i] for c, i in who]).reshape(-1, 2)
        for s in (1, -1):
            nz = np.tile([0, 0, s], (len(P), 1))
            add(np.concatenate([P, np.full((len(P), 1), s * z), nz, uv(P)], 1), tris if s > 0 else tris[:, ::-1])
    # the rounded edge: rings between one inset and the next
    for j in range(n):
        (ca, za, pa), (cb, zb, pb) = levels[j], levels[j + 1]
        both = ca + cb
        if not both:
            continue
        who, tris = cut(both)
        na, nb = _normals(ca), _normals(cb)
        P, N3, Z = [], [], []
        for c, i in who:
            lower = c < len(ca)
            vn = (na[c] if lower else nb[c - len(ca)])[0][i]
            ph = pa if lower else pb
            P.append(both[c][i]); Z.append(za if lower else zb)
            N3.append([vn[0] * math.cos(ph), vn[1] * math.cos(ph), math.sin(ph)])
        P, N3, Z = np.array(P).reshape(-1, 2), np.array(N3).reshape(-1, 3), np.array(Z)
        for s in (1, -1):
            add(np.concatenate([P, (s * Z)[:, None], N3 * [1, 1, s], uv(P)], 1), tris if s > 0 else tris[:, ::-1])
    # the wall round the outline
    cs, z, _ = levels[0]
    for p, (vn, en, sharp, sgn) in zip(cs, _normals(cs)):
        m = len(p)
        nxt = (np.arange(m) + 1) % m
        seg = np.hypot(*(p[nxt] - p).T)
        run = np.concatenate([[0], np.cumsum(seg)])
        n0 = np.where(sharp[:, None], en, vn)                  # at the edge's start: its own direction at a corner, else the smooth one
        n1 = np.where(sharp[nxt][:, None], en, vn[nxt])
        zero, one = np.zeros(m), np.ones(m)
        quad = np.stack([np.concatenate([p, np.full((m, 1), -z), n0, zero[:, None], (run[:-1] / run[-1])[:, None], zero[:, None]], 1),
                         np.concatenate([p[nxt], np.full((m, 1), -z), n1, zero[:, None], (run[1:] / run[-1])[:, None], zero[:, None]], 1),
                         np.concatenate([p[nxt], np.full((m, 1), z), n1, zero[:, None], (run[1:] / run[-1])[:, None], one[:, None]], 1),
                         np.concatenate([p, np.full((m, 1), z), n0, zero[:, None], (run[:-1] / run[-1])[:, None], one[:, None]], 1)], 1).reshape(-1, 8)
        i = np.arange(m) * 4
        tri = np.stack([i, i + 1, i + 2, i, i + 2, i + 3], 1).reshape(-1, 3)
        add(quad, tri if sgn > 0 else tri[:, ::-1])
    if not V:
        return np.zeros((0, 8), np.float32), np.zeros(0, np.uint32)
    out = np.concatenate(V).astype(np.float64)
    out[:, :3] /= K
    return out.astype(np.float32), np.concatenate(T).ravel().astype(np.uint32)


def text_outline(text, font, size, align='center', tracking=0.0, line_height=1.2, features=''):
    """a text's outline as the text layer would set it, turned y up and centred on its own box, `size` units an em"""
    import skia
    from .render import text_layout
    from .paths import _glyph
    em = 200.0                                                 # laid out large, scaled to the size asked
    glyphs, _ = text_layout(font, em, text, align, line_height, tracking * em / max(size, 1e-9), features)
    out = skia.Path()
    for g, x, y, _ in glyphs:
        gp, adv, upem = _glyph(font, g)
        m = skia.Matrix()
        m.setScale(em / upem, em / upem)
        m.postTranslate(x, y)
        out.addPath(gp, m)
    return _placed(out, size / em)


def path_outline(d, height):
    """an SVG path's outline turned y up, centred on its own box and scaled to `height` units tall"""
    from .render import svg_path
    p = svg_path(d)
    b = p.getBounds()
    return _placed(p, height / b.height() if b.height() > 0 else 1.0)


def _placed(p, k):
    import skia
    b = p.getBounds()
    m = skia.Matrix()
    m.setTranslate(-b.centerX(), -b.centerY())
    m.postScale(k, -k)
    out = skia.Path()
    p.transform(m, out)
    return out
