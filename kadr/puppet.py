"""A picture bent by pins (After Effects: the Puppet tool) — the `puppet` layer: a cut-out photograph or drawing is
covered by a mesh, pins are stuck into it, and when a pin is led somewhere the picture follows like a sheet of rubber
that resists stretching: an arm waves, a head nods, a tail wags, a flag bends — from one still picture.

    - id: man
      type: puppet
      src: man.png                   a picture with transparency (the mesh covers what is not transparent)
      cutout: true                   …or a photograph whose cut-out was made by `kadr matte` (man.matte.png beside it)
      pins:                          at — where the pin is stuck, px of the picture;  to — where it is led (keys, a formula)
        hand: {at: [330, 250], to: {0: [330, 250], 1.2: [360, 150], ease: inOut}}
        head: {at: [210, 60], move: "=[14 * sin(2 * t), 0]"}      move — the same as a shift from its own place
        hip:  {at: [200, 420]}                                    neither: the pin holds its place
      mesh: 12                       the mesh's cell, px of the picture (by default chosen by the picture's size;
                                     smaller — smoother bends, slower)
    kadr puppet man.png -o look.jpg [--pins 330,250 210,60] [--mesh 12]
                                     the picture under a ruler with its mesh (and the pins tried): where to stick them
    kadr puppet photo.jpg --cutout --pose -o look.jpg
                                     a photograph of a person: macOS finds the joints (neck, shoulders, elbows,
                                     wrists, hips, knees, ankles) and the face (head, eyes, mouth, chin, crown); they
                                     are drawn and printed as pins ready for the document — lead the ones that move,
                                     the rest hold the body

What is far from every pin keeps its shape and turns as a whole with what leads it; parts that do not touch (a hand cut
out apart from the body) do not pull each other — a part without a pin stays where it is. The layer is a picture like
any other: position, scale, turn, effects, parents work on the whole of it; its size and anchor are the picture's own."""
import functools, math, os

import numpy as np
from PIL import Image

LAM = 1e4                                                    # how hard a pin holds, against 1 for a triangle keeping its shape


def _alpha(src, cutout):
    im = Image.open(src)
    if cutout:
        from .matte import out_path as beside
        return np.asarray(Image.open(beside(src)).convert('L').resize(im.size), np.float32) / 255
    if im.mode in ('RGBA', 'LA', 'PA') or 'transparency' in im.info:
        return np.asarray(im.convert('RGBA'), np.float32)[..., 3] / 255
    return np.ones((im.size[1], im.size[0]), np.float32)


def build_mesh(alpha, cell):
    """the picture's opaque part covered by a grid of cells (grown by one cell, so its soft edge is inside), each cut
    into two triangles → vertices (n, 2) picture px, triangles (m, 3), the grid's origin"""
    h, w = alpha.shape
    ny, nx = -(-h // cell), -(-w // cell)
    pad = np.zeros((ny * cell, nx * cell), np.float32)
    pad[:h, :w] = alpha
    occ = pad.reshape(ny, cell, nx, cell).max((1, 3)) > .02
    big = np.zeros((ny + 2, nx + 2), bool)                    # one cell of room on every side
    big[1:-1, 1:-1] = occ
    grown = big.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown |= np.roll(np.roll(big, dy, 0), dx, 1)
    rr, cc = np.nonzero(grown)
    ids = -np.ones((ny + 3, nx + 3), int)
    for r, c in ((0, 0), (0, 1), (1, 0), (1, 1)):
        ids[rr + r, cc + c] = 0
    vr, vc = np.nonzero(ids == 0)
    ids[vr, vc] = np.arange(len(vr))
    V = np.stack([(vc - 1) * cell, (vr - 1) * cell], 1).astype(float)
    a, b, c_, d = ids[rr, cc], ids[rr, cc + 1], ids[rr + 1, cc], ids[rr + 1, cc + 1]
    even = (rr + cc) % 2 == 0                                  # the diagonal turns from cell to cell: no direction is favoured
    T = np.concatenate([np.stack([a, b, d], 1)[even], np.stack([a, d, c_], 1)[even], np.stack([a, b, c_], 1)[~even], np.stack([b, d, c_], 1)[~even]])
    return V, T, grown


def _bary(p, a, b, c):
    m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
    u, v = np.linalg.solve(m, np.asarray(p, float) - a)
    return 1 - u - v, u, v


class Puppet:
    """the mesh of one picture with its pins stuck: everything that does not change from frame to frame.

    Where the vertices go is found in two steps, each a set of linear equations with one answer (Igarashi, Moscovich,
    Hughes: As-Rigid-As-Possible Shape Manipulation, 2005) — so the picture follows the pins smoothly, a small move of
    a pin is a small move of everything. (The first version searched for the least stretched shape step by step, each
    vertex's surroundings turned and all put back together: that energy has several resting shapes, and from frame to
    frame the picture snapped from one to another — 44 px in one frame while the pin moved 9. Counted, not seen.)
      1. every triangle keeps its shape but may turn and change size: each vertex is written in the frame of the
         opposite edge and asked to stay there. The answer is linear in the pins' places: M1 @ pins.
      2. every triangle of the picture is laid, at its own size, over what step 1 made of it (the best turn), and the
         vertices are put where those triangles' edges want them."""

    def __init__(self, alpha, cell, pins):
        self.V, self.T, self.grown = build_mesh(alpha, cell)
        self.cell = cell
        V, T = self.V, self.T
        n, k = len(V), len(pins)
        self.at = np.asarray(pins, float).reshape(-1, 2)
        C = np.zeros((k, n))
        self.off = []                                          # pins stuck where there is no picture
        for i, p in enumerate(pins):
            for tri in T[self._near(p)]:
                w = _bary(p, *V[tri])
                if min(w) >= -1e-9:
                    C[i, tri] = w
                    break
            else:
                self.off.append(i)
        self.C = C
        e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        e = np.unique(np.sort(e, 1), axis=0)
        self.I, self.J = np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])     # every edge, both ways (for the audit)
        self.E = V[self.I] - V[self.J]
        self.lone, self.act = [], np.zeros(0, int)
        if not n or not k:
            return
        # ---- the parts that do not touch each other: one without a pin stays, one with a single pin goes with it as it
        # is, one with two or more is bent (two pins are what fixes a part's turn and size in step 1)
        lab = np.arange(n)
        while True:
            m = lab.copy()
            np.minimum.at(m, self.I, lab[self.J])
            if (m == lab).all():
                break
            lab = m
        self.part = [-1 if i in self.off else int(lab[np.argmax(C[i])]) for i in range(k)]      # each pin's part
        self.lone = [(np.nonzero(lab == q)[0], self.part.index(q)) for q in set(self.part) if q >= 0 and self.part.count(q) == 1]
        bent = sorted(q for q in set(self.part) if q >= 0 and self.part.count(q) >= 2)
        self.act = np.nonzero(np.isin(lab, bent))[0]
        na = len(self.act)
        if not na:
            return
        idx = -np.ones(n, int)
        idx[self.act] = np.arange(na)
        Ta = idx[T[np.isin(lab[T[:, 0]], bent)]]
        Va = V[self.act]
        pins_in = [i for i in range(k) if self.part[i] in bent]
        Ca = np.zeros((k, na))
        Ca[pins_in] = C[pins_in][:, self.act]
        self.Ca = Ca
        # ---- step 1: unknowns x0, y0, x1, y1, …; for each corner of each triangle two equations with six numbers
        A1 = np.zeros((2 * na, 2 * na))
        for s in range(3):
            v0, v1, v2 = Ta[:, s], Ta[:, (s + 1) % 3], Ta[:, (s + 2) % 3]
            ed, d = Va[v1] - Va[v0], Va[v2] - Va[v0]
            l2 = (ed ** 2).sum(1)
            x, y = (d * ed).sum(1) / l2, (d[:, 0] * ed[:, 1] - d[:, 1] * ed[:, 0]) / l2    # v2 = v0 + x e + y R90(e), R90(e) = (e_y, −e_x)
            one, zero = np.ones(len(Ta)), np.zeros(len(Ta))
            cols = np.stack([2 * v0, 2 * v0 + 1, 2 * v1, 2 * v1 + 1, 2 * v2, 2 * v2 + 1], 1)
            for vals in (np.stack([x - 1, y, -x, -y, one, zero], 1),          # r_x = x2 − x0 − x (x1 − x0) − y (y1 − y0)
                         np.stack([-y, x - 1, y, -x, zero, one], 1)):         # r_y = y2 − y0 − x (y1 − y0) + y (x1 − x0)
                for a in range(6):
                    for b in range(6):
                        np.add.at(A1, (cols[:, a], cols[:, b]), vals[:, a] * vals[:, b])
        C2 = np.zeros((2 * k, 2 * na))
        C2[0::2, 0::2], C2[1::2, 1::2] = Ca, Ca
        self.M1 = np.linalg.solve(A1 + LAM * C2.T @ C2, LAM * C2.T)    # step 1 is M1 @ (the pins' places)
        # ---- step 2: the same matrix every frame — its inverse once
        A2 = np.zeros((na, na))
        self.ti, self.tj = np.concatenate([Ta[:, 0], Ta[:, 1], Ta[:, 2]]), np.concatenate([Ta[:, 1], Ta[:, 2], Ta[:, 0]])   # each triangle's three edges
        np.add.at(A2, (self.ti, self.ti), 1.0)
        np.add.at(A2, (self.tj, self.tj), 1.0)
        np.add.at(A2, (self.ti, self.tj), -1.0)
        np.add.at(A2, (self.tj, self.ti), -1.0)
        self.A2inv = np.linalg.inv(A2 + LAM * Ca.T @ Ca)
        self.Ta = Ta
        self.rd = Va[Ta] - Va[Ta].mean(1)[:, None]             # each triangle's corners from its middle
        self.re = Va[self.ti] - Va[self.tj]                    # each triangle edge as it is in the picture

    def _near(self, p):
        """the triangles that may hold the point: those whose box does"""
        tv = self.V[self.T]
        lo, hi = tv.min(1), tv.max(1)
        return np.nonzero((lo[:, 0] <= p[0]) & (hi[:, 0] >= p[0]) & (lo[:, 1] <= p[1]) & (hi[:, 1] >= p[1]))[0]

    def solve(self, to, step1=False):
        """where every vertex goes when the pins are led to `to` (k, 2)"""
        V = self.V
        to = np.asarray(to, float).reshape(-1, 2)
        P = V.copy()
        if len(to) == 0 or len(V) == 0 or np.abs(to - self.at).max() < 1e-9:
            return P
        for verts, pin in self.lone:
            P[verts] += to[pin] - self.at[pin]
        na = len(self.act)
        if not na:
            return P
        P1 = (self.M1 @ to.ravel()).reshape(-1, 2)
        if step1:
            P[self.act] = P1
            return P
        pt = P1[self.Ta]
        pd = pt - pt.mean(1)[:, None]                          # each triangle after step 1, from its middle
        th = np.arctan2((self.rd[..., 0] * pd[..., 1] - self.rd[..., 1] * pd[..., 0]).sum(1), (self.rd * pd).sum((1, 2)))   # its best turn
        c, s = np.tile(np.cos(th), 3), np.tile(np.sin(th), 3)
        f = np.stack([c * self.re[:, 0] - s * self.re[:, 1], s * self.re[:, 0] + c * self.re[:, 1]], 1)   # its edges at their own size, turned
        rhs = np.stack([np.bincount(self.ti, f[:, a], na) - np.bincount(self.tj, f[:, a], na) for a in (0, 1)], 1) + LAM * self.Ca.T @ to
        P[self.act] = self.A2inv @ rhs
        return P


_puppets = {}


def puppet_of(L):
    key = (L['src'], os.path.getmtime(L['src']), L['cutout'], L['mesh'], tuple(map(tuple, L['pin_at'])))
    hit = _puppets.get(key)
    if hit is None:
        if len(_puppets) > 6:
            _puppets.clear()
        hit = _puppets[key] = Puppet(_alpha(L['src'], L['cutout']), L['mesh'], L['pin_at'])
    return hit


def auto_cell(alpha):
    """a cell that gives about 900 cells over what is opaque (never under 6 px): bends stay smooth, the mesh is made in a second"""
    area = float((alpha > .02).sum())
    return int(max(6, round(math.sqrt(max(area, 1.0) / 900))))


def parse(F, L, lid, out):
    src = F.resolve(str(L.get('src', '')))
    out['src'], out['cutout'] = src, bool(L.get('cutout', False))
    out['psize'], out['cell'], out['pin_at'], out['pins'], out['pin_names'], out['mesh'] = (100.0, 100.0), 16.0, [], [], [], 16
    if not os.path.isfile(src):
        F.issue('error', f"нет файла {L.get('src')}", lid)
        return
    try:
        if out['cutout']:
            from .matte import out_path as beside
            if not os.path.exists(beside(src)):
                F.issue('error', f'вырезки ещё нет: сначала kadr matte {L["src"]}', lid)
                out['cutout'] = False
        alpha = _alpha(src, out['cutout'])
    except (OSError, ValueError) as e:
        F.issue('error', f'{L.get("src")}: не читается как картинка ({e})', lid)
        return
    h, w = alpha.shape
    out['psize'] = (float(w), float(h))
    if not (alpha > .02).any():
        F.issue('error', 'картинка целиком прозрачная: булавки ставить не во что', lid)
        return
    cell = L.get('mesh')
    if cell is None:
        cell = auto_cell(alpha)
    elif not (isinstance(cell, (int, float)) and 3 <= cell <= 200):
        F.issue('error', 'mesh: ячейка сетки, от 3 до 200 px картинки', lid)
        cell = auto_cell(alpha)
    out['mesh'] = int(round(cell))
    pins = L.get('pins') or {}
    if not isinstance(pins, dict) or not pins:
        F.issue('error', 'pins: {имя: {at: [x, y], to: … или move: …}, …} — хотя бы одна булавка', lid)
        return
    for name, p in pins.items():
        where = f'{lid}.pins.{name}'
        if not isinstance(p, dict) or not (isinstance(p.get('at'), (list, tuple)) and len(p['at']) == 2):
            F.issue('error', 'булавке нужно at: [x, y] — где она на картинке', where)
            continue
        for k in p:
            if k not in ('at', 'to', 'move'):
                F.issue('error', f'неизвестное «{k}»; у булавки есть at, to, move', where, k, {'at', 'to', 'move'})
        if 'to' in p and 'move' in p:
            F.issue('error', 'одно из двух: to (куда прийти) или move (на сколько сдвинуться)', where)
        at = (float(p['at'][0]), float(p['at'][1]))
        pr = F.prop(p['to'], f'{where}.to', 2) if 'to' in p else F.prop(p['move'], f'{where}.move', 2) if 'move' in p else None
        if pr is not None:
            pr.film, pr.t_in = F, 0.0
        out['pin_at'].append(at)
        out['pins'].append((pr, 'move' in p))
        out['pin_names'].append(str(name))
    if not out['pin_at']:
        return
    n = int((build_mesh(alpha, out['mesh'])[2]).sum())
    if n > 2600:
        F.issue('error', f'сетка слишком мелкая для такой картинки: {n} ячеек (предел 2600). Возьмите mesh побольше — {int(math.ceil(out["mesh"] * math.sqrt(n / 2000)))} или больше', lid)
        out['pin_at'], out['pins'], out['pin_names'] = [], [], []
        return
    pup = puppet_of(out)
    for k in pup.off:
        F.issue('error', f'булавка стоит на прозрачном месте картинки ({out["pin_at"][k][0]:g}, {out["pin_at"][k][1]:g}): посмотрите, где сетка — kadr puppet {L["src"]} -o лист.jpg', f'{lid}.pins.{out["pin_names"][k]}')


def targets(L, t):
    out = []
    for at, (pr, rel) in zip(L['pin_at'], L['pins']):
        if pr is None:
            out.append(at)
        else:
            v = pr.at(t)
            out.append((at[0] + v[0], at[1] + v[1]) if rel else (v[0], v[1]))
    return np.array(out, float).reshape(-1, 2)


def audit(F, L, limit=500):
    """what the eye would catch in the bent picture, found by counting → [(first frame, last frame, words)]:
    a fold (triangles of the mesh turned inside out: a pin led too far, across the picture itself), a stretch (pins
    pulled apart further than the picture can follow without thinning), a jump (in one frame the picture moves several times further than in the frames before and after: a free end
    swinging far and fast on a turning forearm is not one — its speed changes from frame to frame little by little)"""
    if not L['pin_at'] or all(pr is None for pr, _ in L['pins']):
        return []
    pup = puppet_of(L)
    V, T = pup.V, pup.T
    area = lambda P: (P[T[:, 1], 0] - P[T[:, 0], 0]) * (P[T[:, 2], 1] - P[T[:, 0], 1]) - (P[T[:, 1], 1] - P[T[:, 0], 1]) * (P[T[:, 2], 0] - P[T[:, 0], 0])
    a0 = area(V)
    e0 = np.hypot(*pup.E.T)
    i0, i1 = int(math.ceil(L['in'] * F.fps - 1e-6)), min(int(math.ceil(L['out'] * F.fps - 1e-6)), F.frames)
    fold, stretch, prev, moved = [], [], None, {}
    for i in range(i0, min(i1, i0 + limit)):
        to = targets(L, i / F.fps)
        P = pup.solve(to)
        if (area(P) * a0 < 0).mean() > .002:               # (a pin's own neighbourhood twisted by a large turn is one or two triangles of thousands)
            fold.append(i)
        e = np.hypot(*(P[pup.I] - P[pup.J]).T) / e0
        if float(np.sort(e)[-max(1, len(e) // 20)]) > 1.5:     # one edge in twenty half as long again: not a twist at one joint, the picture is being pulled
            stretch.append(i)
        if prev is not None:
            moved[i] = float(np.abs(P - prev).max())
        prev = P
    jump = [i for i, d in moved.items() if d > 8 and i - 1 in moved and i + 1 in moved and d > 3 * max(moved[i - 1], moved[i + 1])]
    out = []
    from .check import ranges
    for a, b in ranges(fold):
        out.append((a, b, 'картинка на булавках сложилась складкой: булавку увели слишком далеко, поперёк самой картинки'))
    for a, b in ranges(stretch):
        out.append((a, b, 'булавки растягивают картинку в полтора раза и больше: она тянется как резина; разведите их меньше'))
    for a, b in ranges(jump):
        out.append((a, b, 'картинка на булавках дёргается: за один кадр она сдвигается в разы дальше, чем в соседних. Так бывает, когда булавку ведут сквозь саму картинку или почти вплотную к другой'))
    return out


@functools.lru_cache(4)
def _picture(src, mtime, cutout):
    import skia
    a = np.asarray(Image.open(src).convert('RGBA')).copy()
    if cutout:
        a[..., 3] = np.round(_alpha(src, True) * 255).astype(np.uint8)
    pm = a.astype(np.float32)
    pm[..., :3] *= pm[..., 3:] / 255                           # the mesh is drawn from colour already times its alpha: no dark rim
    pm = np.ascontiguousarray(np.round(pm).astype(np.uint8))
    return skia.Image.fromarray(pm, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType, copy=True)


def render(F, L, t, sc):
    """the bent picture at t at `sc` px per picture px → (x0, y0, w, h in picture px, premultiplied PIL RGBa) or None"""
    import skia
    if not L['pin_at']:
        return None
    pup = puppet_of(L)
    P = pup.solve(targets(L, t))
    lo, hi = np.floor(P.min(0)) - 1, np.ceil(P.max(0)) + 1
    sc = min(sc, math.sqrt(40e6 / max(1.0, float((hi - lo).prod()))))
    W, H = int(math.ceil((hi[0] - lo[0]) * sc)), int(math.ceil((hi[1] - lo[1]) * sc))
    if W <= 0 or H <= 0:
        return None
    img = _picture(L['src'], os.path.getmtime(L['src']), L['cutout'])
    surf = skia.Surface.MakeRaster(skia.ImageInfo.Make(W, H, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType))
    c = surf.getCanvas()
    c.clear(skia.Color4f(0, 0, 0, 0))
    c.scale(sc, sc)
    c.translate(-float(lo[0]), -float(lo[1]))
    samp = skia.SamplingOptions(skia.CubicResampler.Mitchell()) if sc >= .7 else skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)
    paint = skia.Paint(Shader=(img if sc >= .7 else img.withDefaultMipmaps()).makeShader(skia.TileMode.kDecal, skia.TileMode.kDecal, samp))
    vt = skia.Vertices(skia.Vertices.kTriangles_VertexMode, [skia.Point(float(x), float(y)) for x, y in P], [skia.Point(float(x), float(y)) for x, y in pup.V],
                       None, [int(i) for i in pup.T.ravel()])
    c.drawVertices(vt, paint)
    a = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
    if not a[..., 3].any():
        return None
    return float(lo[0]), float(lo[1]), W / sc, H / sc, Image.fromarray(np.ascontiguousarray(a), 'RGBa')


JOINTS = ('crown', 'face', 'nose', 'eye_l', 'eye_r', 'mouth', 'chin', 'neck', 'shoulder_l', 'shoulder_r', 'elbow_l', 'elbow_r', 'wrist_l', 'wrist_r',
          'hip_l', 'hip_r', 'knee_l', 'knee_r', 'ankle_l', 'ankle_r', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle')


def joints(src, cutout=False, cell=None, sure=.3):
    """the person's joints and the points of the face that can be pins: seen with confidence and standing on the mesh
    → {name: (x, y)}. The middle of the face is called head (the nose of the body's joints when no face is seen)"""
    from .matte import pose
    found = pose(src)
    alpha = _alpha(src, cutout)
    pup = Puppet(alpha, int(cell or auto_cell(alpha)), [])
    out = {}
    for name in JOINTS:
        if name in found and found[name][2] >= sure:
            x, y = found[name][:2]
            if any(min(_bary((x, y), *pup.V[tri])) >= -1e-9 for tri in pup.T[pup._near((x, y))]):
                key = 'head' if name in ('face', 'nose') else name
                if key not in out:
                    out[key] = (round(x), round(y))
    return out


def sheet(src, out, pins=(), cell=None, cutout=False, names=()):
    """the picture under a ruler with its mesh and the pins tried → a file to look at; → (cell, cells, pins off the picture)"""
    from PIL import ImageDraw, ImageFont
    from .render import font_file
    font = ImageFont.truetype(font_file('Arial'), 15)
    alpha = _alpha(src, cutout)
    h, w = alpha.shape
    cell = int(cell or auto_cell(alpha))
    pup = Puppet(alpha, cell, [tuple(p) for p in pins])
    k = min(3.0, max(1.0, 1400 / max(w, h))) if max(w, h) < 1400 else 1400 / max(w, h)
    m = 46                                                     # the ruler's margin
    rgb = np.asarray(Image.open(src).convert('RGB'), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((xx // 16 + yy // 16) % 2 == 0)[..., None], 62.0, 78.0)
    base = Image.fromarray(np.round(rgb * alpha[..., None] + board * (1 - alpha[..., None])).astype(np.uint8)).resize((int(w * k), int(h * k)), Image.LANCZOS)
    page = Image.new('RGB', (base.width + m + 8, base.height + m + 8), '#16181d')
    page.paste(base, (m, m))
    d = ImageDraw.Draw(page, 'RGBA')
    for a, b, c_ in pup.T:
        pts = [(m + pup.V[i][0] * k, m + pup.V[i][1] * k) for i in (a, b, c_, a)]
        d.line(pts, fill=(80, 220, 255, 70), width=1)
    step = 50 if max(w, h) <= 700 else 100 if max(w, h) <= 1600 else 200
    for x in range(0, w + 1, step):
        d.line([(m + x * k, m - 8), (m + x * k, m + base.height)], fill=(255, 255, 255, 46), width=1)
        d.text((m + x * k + 2, 4), str(x), fill='#ffd060', font=font)
    for y in range(0, h + 1, step):
        d.line([(m - 8, m + y * k), (m + base.width, m + y * k)], fill=(255, 255, 255, 46), width=1)
        d.text((4, m + y * k + 2), str(y), fill='#ffd060', font=font)
    for i, p in enumerate(pins):
        x, y = m + p[0] * k, m + p[1] * k
        col = '#ff3030' if i in pup.off else '#ffe000'
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=col, width=3)
        d.text((x + 11, y - 9), (f'{names[i]}  ' if i < len(names) and names[i] else '') + f'{p[0]:g},{p[1]:g}' + (' — мимо' if i in pup.off else ''), fill=col, font=font, stroke_width=3, stroke_fill='#101216')
    page.save(out, quality=92)
    return cell, int(pup.grown.sum()), [pins[i] for i in pup.off]
