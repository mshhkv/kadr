"""A 3D scene looked at from all sides and checked (the `kadr scene` command, the server's `scene` tool, `kadr check`).

A scene is built blind, a number at a time; a camera shows it from one side only. So:

  sheet()   the scene — or one named part of it with all that hangs on it — from six sides in an even light: front,
            back, left, right, top, three quarters; the scene's own lamps, surroundings and lens are put aside, a body
            is shown as it is shaped and painted. A film's layer is shown in the pose it has at the asked moment.
  audit()   what cannot be right, found by distances (each body is known as what it is — a box, a ball, a turned
            profile, a ring, a sheet — so no triangles are needed):
              through   two solid bodies of different things run through each other (a leg through a chair's seat).
                        Bodies of one thing — the children of one top node of the scene — may: sinking a part into
                        its neighbour is how a thing is put together. So a thing is a group: a chair's back, seat and
                        headrest under one node
              pokes     a body runs through a sheet of another thing (a book through the wall)
              hidden    a body lies wholly inside another body: it will never be seen
              floats    a thing touches no other thing and none is near it (a mug above the table); a body torn
                        off its own thing (a shoe a step away from its leg)
              flicker   two sheets lie in one plane, one over the other: the picture will flicker between them
            A thing's «thickness» here is the radius of the largest ball that fits in it.
            What is meant goes into the scene: accept: [[mug, laptop], [fly]] — a pair that is meant to meet, a body
            that is meant to hang.

Scenes written in Kadr's words are measured exactly; a body of an exported scene, a model, a text or an outline is
taken as the box round it (the report says so)."""
import math, os

import numpy as np

from . import scene3d

VIEWS = (('спереди', (0, 0, 1)), ('сзади', (0, 0, -1)), ('слева', (-1, 0, 0)), ('справа', (1, 0, 0)), ('сверху', (0, 1, .03)), ('три четверти', (1, .6, 1)))
NEUTRAL = dict(exposure=1.0, bloom=0.0, tint=[1.0, 1.0, 1.0], vignette=[.42, .86, 0.0], grain=0.0, shutter=0.0)
THROUGH, DEEP, POKES, FLOATS = .2, .5, .05, .25                 # shares of a body's own thickness / size past which it is told


class _Const:
    def __init__(self, v):
        self.v = v

    def at(self, t):
        return self.v


def open_scene(path, layer=None):
    """a film's 3D layer, or a scene's own file as a film of that one layer → (film, layer)"""
    import yaml
    from .doc import Film, DocError, Issue
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise DocError([Issue('error', f'нет файла {path}')])
    film = False
    if path.lower().endswith(('.yaml', '.yml')):
        d = yaml.safe_load(open(path, encoding='utf-8'))
        film = isinstance(d, dict) and 'layers' in d
    if not film:
        F = Film(data={'film': {'size': [480, 480], 'fps': 30, 'duration': 1}, 'layers': [dict(id='scene', type='scene3d', src=path)]})
        return F, F.layers[0]
    F = Film(path)

    def walk(f):
        for L in f.layers:
            if L.get('type') == 'scene3d':
                yield L
            if L.get('comp') is not None and hasattr(L['comp'], 'layers'):
                yield from walk(L['comp'])
    found = list(walk(F))
    if layer is not None:
        found = [L for L in found if L['id'] == layer]
    if not found:
        raise DocError([Issue('error', f'в фильме нет слоя scene3d «{layer}»' if layer is not None else 'в фильме нет ни одного слоя scene3d')])
    return F, found[0]


def _camera(pos, at, fov=22.0):
    return dict(position=_Const(tuple(pos)), look_at=_Const(tuple(at)), fov=_Const(fov), lens=_Const(0.0), focus=None)


def pose(sc, L, t, fps=30.0):
    """where every node stands at t (the layer's rig applied) → (world matrices (N, 4, 4), which are seen (N,))"""
    probe = {}
    sc.frame(t, 8, 8, fps, samples=1, flags={'still': True, 'probe': probe}, over=L.get('over'), cam_over=_camera((0, 0, 100), (0, 0, 0)), cuts=L.get('cuts'), lit=L.get('lit'), mts=L.get('mts'))
    return probe['world'], probe['seen'] > .5


def subtree(sc, name):
    """the node called `name` and all that hangs on it → their numbers"""
    ids = [i for i, n in enumerate(sc.nodes) if n.get('name') == name]
    if not ids:
        from .doc import DocError, Issue
        import difflib
        names = sorted({n['name'] for n in sc.nodes if n.get('name') and not n['name'].isdigit()})
        near = difflib.get_close_matches(str(name), names, n=1)
        raise DocError([Issue('error', f'в сцене нет узла «{name}»' + (f' (может быть, {near[0]}?)' if near else ''))])
    keep = {ids[0]}
    for i in sc.order:
        if sc.parent[i] in keep:
            keep.add(i)
    return keep


def subtree_names(sc, name):
    """the names of the node `name` and of all that hangs on it"""
    return {sc.nodes[i]['name'] for i in subtree(sc, name)}


# ---- distances: each body as what it is, in its own space

def _box(half, r):
    h = np.asarray(half, float) - r

    def f(p):
        q = np.abs(p) - h
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0) - r
    return f


def _turned(profile, solid):
    P = np.asarray(profile, float)
    a, ab = P[:-1], P[1:] - P[:-1]
    A = np.concatenate([P, P[:1]])
    a2, b2 = A[:-1], A[1:]
    dy = np.where(b2[:, 1] - a2[:, 1] == 0, 1e-18, b2[:, 1] - a2[:, 1])

    def f(p):
        q = np.stack([np.hypot(p[:, 0], p[:, 2]), p[:, 1]], -1)
        t = np.clip(((q[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-18)[None], 0, 1)
        d = np.linalg.norm(q[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(1)
        if not solid:
            return d
        cross = (a2[None, :, 1] > q[:, None, 1]) != (b2[None, :, 1] > q[:, None, 1])
        x = a2[None, :, 0] + (q[:, None, 1] - a2[None, :, 1]) * (b2[None, :, 0] - a2[None, :, 0]) / dy[None]
        inside = (cross & (q[:, None, 0] < x)).sum(1) % 2 == 1
        return np.where(inside, -d, d)
    return f


def _distance(solid):
    """→ (signed distance in the body's own space: below zero inside; a sheet's and a shell's is never below zero), kind"""
    k = solid[0]
    if k == 'box':
        return _box(solid[1], solid[2]), 'solid'
    if k == 'ball':
        return (lambda p, r=solid[1]: np.linalg.norm(p, axis=1) - r), 'solid'
    if k == 'torus':
        return (lambda p, R=solid[1], tb=solid[2]: np.hypot(np.hypot(p[:, 0], p[:, 1]) - R, p[:, 2]) - tb), 'solid'
    if k in ('lathe', 'shell'):
        return _turned(solid[1], k == 'lathe'), 'solid' if k == 'lathe' else 'shell'
    if k == 'disc':
        return (lambda p, r=solid[1]: np.hypot(np.maximum(np.hypot(p[:, 0], p[:, 1]) - r, 0), p[:, 2])), 'sheet'
    hw, hh = solid[1], solid[2]                                 # a sheet facing its own z
    return (lambda p: np.sqrt((np.maximum(np.abs(p[:, :2]) - (hw, hh), 0) ** 2).sum(1) + p[:, 2] ** 2)), 'sheet'


def bodies(sc, world, seen, keep=None):
    """the scene's bodies as they stand: each with its distance, its place, samples of its surface and of its inside"""
    top = {}
    for i in sc.order:                                          # the thing a node belongs to: the top node it hangs on
        p = sc.parent[i]
        top[i] = i if p < 0 or (sc.nodes[p]['kind'] == 'Scene') else top[p]
    out = []
    for i, n in enumerate(sc.nodes):
        if n['kind'] != 'Mesh' or not seen[i] or (keep is not None and i not in keep):
            continue
        first, cnt, base = sc.geos[n['geo']]
        v = sc.verts[base + np.unique(sc.indices[first:first + cnt]), :3].astype(float)
        lo, hi = v.min(0), v.max(0)
        W = world[i]
        sv = np.linalg.svd(W[:3, :3], compute_uv=False)
        if sv.min() < 1e-9:
            continue
        approx = 'solid' not in n
        if approx:                                             # not one of Kadr's own bodies: the box round it (flat: a sheet)
            ext = hi - lo
            if ext.min() < 1e-6 * max(ext.max(), 1e-9):
                ax = int(np.argmin(ext))
                o = [a for a in range(3) if a != ax]
                c0 = (lo + hi) / 2
                def f(p, o=o, ax=ax, c0=c0, hw=ext[o] / 2):
                    q = p - c0
                    return np.sqrt((np.maximum(np.abs(q[:, o]) - hw, 0) ** 2).sum(1) + q[:, ax] ** 2)
                kind = 'sheet'
            else:
                c0 = (lo + hi) / 2
                f, kind = (lambda p, g=_box(ext / 2, 0), c0=c0: g(p - c0)), 'solid'
        else:
            f, kind = _distance(n['solid'])
        m = sc.J['mats'][n['mat']]
        b = dict(i=i, name=n.get('name') or f'#{i}', top=top[i], kind=kind, approx=approx, W=W, Wi=np.linalg.inv(W), k=float(sv.min()), f=f, flat=n.get('solid', ('',))[0] == 'sheet' or (approx and kind == 'sheet'),
                 clear=bool(m.get('transparent')) or m.get('opacity', 1) < 1 or bool(m.get('additive')))
        b['surf'] = (W @ np.concatenate([v[::max(1, len(v) // 300)], np.ones((len(v[::max(1, len(v) // 300)]), 1))], 1).T).T[:, :3]
        cs = (W @ np.concatenate([np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]), np.ones((8, 1))], 1).T).T[:, :3]
        b['lo'], b['hi'] = np.minimum(cs.min(0), b['surf'].min(0)), np.maximum(cs.max(0), b['surf'].max(0))
        b['size'] = float((b['hi'] - b['lo']).max())
        if kind == 'solid':
            g = np.stack(np.meshgrid(*[np.linspace(lo[a], hi[a], 7) for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
            d = f(g)
            g = g[d <= 0]
            b['inner'] = (W @ np.concatenate([g, np.ones((len(g), 1))], 1).T).T[:, :3] if len(g) else np.zeros((0, 3))
            b['thick'] = float(-d.min() * b['k']) if len(g) else 0.0
            b['step'] = float((hi - lo).max() / 6 * sv.max())
        out.append(b)
    for b in out:
        b['thing'] = sc.nodes[b['top']].get('name') or f"#{b['top']}"
    return out


def _d(b, pts):
    """how far the world points are from the body (below zero: inside it), in the world's units — never more than the truth"""
    return b['f']((b['Wi'] @ np.concatenate([pts, np.ones((len(pts), 1))], 1).T).T[:, :3]) * b['k']


def _apart(a, b):
    """the gap between the boxes round two bodies (0: they overlap)"""
    return float(np.linalg.norm(np.maximum(np.maximum(a['lo'] - b['hi'], b['lo'] - a['hi']), 0)))


def _common(a, b):
    """the thickness of what two solid bodies share: the largest ball that fits in both"""
    pts = np.concatenate([a['inner'], b['inner']])
    if not len(pts):
        return 0.0
    val = np.minimum(-_d(a, pts), -_d(b, pts))
    k = int(np.argmax(val))
    best, p, step = float(val[k]), pts[k].copy(), min(a['step'], b['step'])
    moves = np.concatenate([np.eye(3), -np.eye(3)])
    for _ in range(14):                                         # from the best of the samples, by ever smaller steps
        q = p[None] + moves * step
        v = np.minimum(-_d(a, q), -_d(b, q))
        j = int(np.argmax(v))
        if v[j] > best:
            best, p = float(v[j]), q[j]
        else:
            step /= 2
    return best


def _num(v):
    return f'{v:.1f}'.replace('.', ',') if abs(v) >= 1 else f'{v:.2f}'.replace('.', ',')


def audit(sc, world, seen, keep=None):
    """→ [(kind, the names it is about, words)], the worst first within each kind"""
    B = bodies(sc, world, seen, keep)
    meant = {frozenset(q) for q in sc.J.get('accept') or [] if len(q) == 2}     # pairs the scene itself says are meant to meet
    alone = {q[0] for q in sc.J.get('accept') or [] if len(q) == 1}             # and bodies meant to hang in the air
    ok = lambda a, b: any(frozenset((x, y)) in meant for x in (a['name'], a['thing']) for y in (b['name'], b['thing']))
    who = lambda b: b['name'] if b['thing'] == b['name'] else f"{b['name']} ({b['thing']})"
    rough = lambda *bs: ' — мерено по рамке вокруг тела, приблизительно' if any(b['approx'] for b in bs) else ''
    out, inside = [], set()
    for a in B:                                                 # hidden: wholly inside another body
        for b in B:
            if a is b or b['kind'] != 'solid' or b['clear'] or _apart(a, b) > 0 or ok(a, b):
                continue
            if (a['lo'] >= b['lo'] - 1e-9).all() and (a['hi'] <= b['hi'] + 1e-9).all() and _d(b, a['surf']).max() < -1e-6 * max(a['size'], 1e-9):
                inside.add((a['i'], b['i']))
                out.append((1.0, 'hidden', (a['name'], b['name']), f"{who(a)} целиком внутри {who(b)}: его не видно{rough(b)}"))
                break
    for x, a in enumerate(B):
        for b in B[x + 1:]:
            if a['top'] == b['top'] or _apart(a, b) > 0 or (a['i'], b['i']) in inside or (b['i'], a['i']) in inside or ok(a, b):
                continue
            if a['kind'] == 'solid' and b['kind'] == 'solid':   # through: two things' solid bodies share a part
                c = _common(a, b)
                thin = min(a['thick'], b['thick'])
                # told when half the thinner body's thickness is shared — or a fifth, if that is also a real part of its
                # size (a card sunk a hair into the floor it lies on is not a card through the floor)
                if thin > 0 and (c / thin > DEEP or (c / thin > THROUGH and c > .02 * min(a['size'], b['size']))):
                    small, big = (a, b) if a['thick'] <= b['thick'] else (b, a)
                    out.append((c / thin, 'through', (a['name'], b['name']), f"{who(small)} проходит сквозь {who(big)}: общая часть толщиной {_num(c)} — это {c / thin * 100:.0f} % толщины {small['name']}{rough(a, b)}"))
            for s, o in ((a, b), (b, a)):                       # pokes: a body on both sides of another thing's sheet
                if not s['flat'] or o['flat']:
                    continue
                q = (s['Wi'] @ np.concatenate([o['surf'], np.ones((len(o['surf']), 1))], 1).T).T[:, :3]
                n_ = sc.nodes[s['i']]
                if 'solid' not in n_:
                    continue
                inset = 1e-3 * s['size'] / s['k']                 # a floor a wall stands on runs past it and is not through it
                on = (np.abs(q[:, 0]) < n_['solid'][1] - inset) & (np.abs(q[:, 1]) < n_['solid'][2] - inset)
                if on.sum() < 2:
                    continue
                front, back = float(q[on, 2].max()) * s['k'], float(-q[on, 2].min()) * s['k']
                deep = min(front, back)
                if deep > POKES * o['size']:
                    out.append((deep / o['size'], 'pokes', (o['name'], s['name']), f"{who(o)} торчит сквозь лист {who(s)} на {_num(deep)}"))
    flats = [b for b in B if b['flat'] and 'solid' in sc.nodes[b['i']]]
    for x, a in enumerate(flats):                               # flicker: two sheets in one plane, one over the other
        for b in flats[x + 1:]:
            na, nb = a['W'][:3, 2] / np.linalg.norm(a['W'][:3, 2]), b['W'][:3, 2] / np.linalg.norm(b['W'][:3, 2])
            if abs(float(na @ nb)) < .9998 or _apart(a, b) > 0 or ok(a, b):
                continue
            off = abs(float(na @ (b['W'][:3, 3] - a['W'][:3, 3])))
            q = (a['Wi'] @ np.concatenate([b['surf'], np.ones((len(b['surf']), 1))], 1).T).T[:, :3]
            sa = sc.nodes[a['i']]['solid']
            lo2, hi2 = q[:, :2].min(0), q[:, :2].max(0)
            over = (min(hi2[0], sa[1]) - max(lo2[0], -sa[1]) > 1e-6) and (min(hi2[1], sa[2]) - max(lo2[1], -sa[2]) > 1e-6)
            if over and off < 1e-4 * max(a['size'], b['size']) and float(np.abs(q[:, 2]).max()) * a['k'] < 1e-4 * max(a['size'], b['size']):
                out.append((1.0, 'flicker', (a['name'], b['name']), f"{who(a)} и {who(b)} лежат в одной плоскости один на другом: картинка будет мерцать между ними (разведите их хоть немного)"))
    def gap(a, others):
        """how far the body is from the nearest of the others → (the gap, that one)"""
        best, nb = math.inf, None
        for box_gap, k in sorted((_apart(a, b), k) for k, b in enumerate(others))[:6]:
            if box_gap >= best:
                break
            g = float(max(0.0, min(_d(others[k], a['surf']).min(), _d(a, others[k]['surf']).min())))
            if g < best:
                best, nb = g, others[k]
        return best, nb
    things = {}
    for b in B:
        things.setdefault(b['top'], []).append(b)
    for top, own in things.items():                             # floats: a thing none of whose bodies is near another thing
        rest = [b for b in B if b['top'] != top]
        name = own[0]['thing']
        if not rest or name in alone or not any(b['kind'] == 'solid' for b in own):
            continue
        size = float((np.max([b['hi'] for b in own], 0) - np.min([b['lo'] for b in own], 0)).max())
        g, nb, a = min(((*gap(b, rest), b) for b in own), key=lambda q: q[0])
        if g > FLOATS * size:
            out.append((g / size, 'floats', (name,), f"{name} висит в воздухе: до ближайшего тела ({nb['name']}) {_num(g)} при собственном размере {_num(size)}{rough(a, nb)}"))
    for a in B:                                                 # and a body torn off its own thing: nothing of any thing is near it
        own = things[a['top']]
        if a['kind'] != 'solid' or len(own) < 2 or a['name'] in alone or a['thing'] in alone:
            continue
        g, nb = gap(a, [b for b in B if b is not a])
        if g > FLOATS * a['size']:
            out.append((g / a['size'], 'floats', (a['name'],), f"{who(a)} оторвано от своей вещи и висит в воздухе: до ближайшего тела ({nb['name']}) {_num(g)} при собственном размере {_num(a['size'])}{rough(a, nb)}"))
    order = {'through': 0, 'pokes': 1, 'hidden': 2, 'floats': 3, 'flicker': 4}
    return [(k, names, words) for _, k, names, words in sorted(out, key=lambda q: (order[q[1]], -q[0]))]


# ---- the look from six sides

def extent(sc, world, seen, keep=None, content=False):
    """the box round what is drawn → (its lowest corner, its highest). content: round most of it — the few bodies far
    larger than the rest (a wall, a floor, a long table) are left to run out of the picture, or the things on them would
    be specks"""
    los, his = [], []
    for i, n in enumerate(sc.nodes):
        if n['kind'] != 'Mesh' or not seen[i] or (keep is not None and i not in keep):
            continue
        first, cnt, base = sc.geos[n['geo']]
        v = sc.verts[base + np.unique(sc.indices[first:first + cnt]), :3].astype(float)
        a, b = v.min(0), v.max(0)
        cs = (world[i] @ np.concatenate([np.array([[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])]), np.ones((8, 1))], 1).T).T[:, :3]
        los.append(cs.min(0)); his.append(cs.max(0))
    if not los:
        return np.full(3, np.inf), np.full(3, -np.inf)
    los, his = np.array(los), np.array(his)
    if content and len(los) >= 20:
        k = max(1, len(los) // 25)                             # one body in twenty-five may stick out on each side
        return np.sort(los, 0)[k], np.sort(his, 0)[-1 - k]
    return los.min(0), his.max(0)


def sheet(sc, L, t, fps=30.0, node=None, tile=480, world=None, seen=None, whole=False):
    """the six views as one picture (PIL), three across → (picture, the box round what is there). A whole scene is framed
    round most of its bodies (extent's `content`) unless `whole`; a named part — round all of it"""
    from PIL import Image, ImageDraw, ImageFont
    from .cli import font_file
    if world is None:
        world, seen = pose(sc, L, t, fps)
    keep = subtree(sc, node) if node else None
    lo, hi = extent(sc, world, seen, keep)
    if not np.isfinite(lo).all():
        from .doc import DocError, Issue
        raise DocError([Issue('error', 'показывать нечего: в сцене (в этой её части) нет ни одного видимого тела')])
    flo, fhi = extent(sc, world, seen, keep, content=not whole and not node)
    c = (flo + fhi) / 2
    corners = np.array([[x, y, z] for x in (flo[0], fhi[0]) for y in (flo[1], fhi[1]) for z in (flo[2], fhi[2])]) - c
    far = float(np.linalg.norm(np.maximum(np.abs(lo - c), np.abs(hi - c))))     # the farthest anything is from the middle
    fov = 22.0
    fnt = ImageFont.truetype(font_file('Arial'), 20)
    pic = Image.new('RGB', (3 * tile, 2 * (tile + 34)), (24, 25, 28))
    dr = ImageDraw.Draw(pic)
    for k, (name, d) in enumerate(VIEWS):
        d = np.array(d, float) / np.linalg.norm(d)
        cw = scene3d.look_at(c + d, c)
        ex, ey, ez = (float(np.abs(corners @ cw[:3, a]).max()) for a in range(3))
        dist = max(ex, ey, 1e-6) / math.tan(math.radians(fov / 2)) * 1.1 + ez
        pos = c + d * dist
        sun = pos + (cw[:3, 1] * .7 - cw[:3, 0] * .5) * dist    # over the camera's left shoulder
        lamps = [dict(kind='HemisphereLight', color=[1.0, 1.0, 1.0], intensity=.95, pos=[0, 1, 0], ground=[.5, .5, .5]),
                 dict(kind='DirectionalLight', color=[1.0, 1.0, 1.0], intensity=1.5, pos=list(sun), target=list(c))]
        a = sc.render(t, tile, tile, fps, samples=6, look=NEUTRAL, flags={'lights': lamps, 'noenv': True, 'nodes': keep, 'clip': (max(dist - far * 1.05, dist * .02), dist + far * 1.05), 'noshadow': True},
                      over=L.get('over'), cam_over=_camera(pos, c, fov), mts=L.get('mts')).download()
        al = np.clip(a[..., 3:4], 0, 1)
        rgb = np.clip(a[..., :3] * al + np.array([58.0, 61.0, 69.0]) * (1 - al), 0, 255).astype(np.uint8)
        x, y = (k % 3) * tile, (k // 3) * (tile + 34)
        pic.paste(Image.fromarray(rgb), (x, y + 34))
        dr.text((x + 10, y + 6), name, font=fnt, fill=(255, 200, 100))
    return pic, (lo, hi)


def look(path, layer=None, at='0', node=None, tile=480, check=True, whole=False):
    """the whole of `kadr scene`: → (the sheet, lines of words)"""
    F, L = open_scene(path, layer)
    t = F.time(at) if at not in (None, '') else 0.0
    sc = scene3d.layer_scene(L)
    world, seen = pose(sc, L, t, F.fps)
    pic, (lo, hi) = sheet(sc, L, t, F.fps, node, tile, world, seen, whole)
    keep = subtree(sc, node) if node else None
    n = sum(1 for i, x in enumerate(sc.nodes) if x['kind'] == 'Mesh' and seen[i] and (keep is None or i in keep))
    words = [f"{node or os.path.basename(path)}: тел {n}; ширина × высота × глубина {_num(hi[0] - lo[0])} × {_num(hi[1] - lo[1])} × {_num(hi[2] - lo[2])}"
             + (f' (на {t:.2f} с)' if t else '')]
    if not node:
        things = [x['name'] for i, x in enumerate(sc.nodes) if (sc.parent[i] < 0 or sc.nodes[sc.parent[i]]['kind'] == 'Scene') and x.get('name') and not x['name'].isdigit() and x['kind'] in ('Mesh', 'Group')]
        if things:
            words.append('вещи (каждую можно снять отдельно, node): ' + ', '.join(things))
    if check:
        found = audit(sc, world, seen, keep)
        words += [w for _, _, w in found] or ['— замечаний к сцене нет']
        if sc.J.get('accept') and not node:
            words.append(f"(пар, о которых сцена сама говорит «так задумано»: {len(sc.J['accept'])})")
    return pic, words
