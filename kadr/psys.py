"""Particle systems (AE.md, M5): two layer types, drawn in the layer's own plane and then placed like a picture — so
they move, turn, tilt in 3D, take effects, mattes, blend modes and `echo` (trails) like any layer.

    type: emitter      particles are born, fly under forces, hit things and die (Particular, Newton)
      size: [w, h]     the layer's plane; every coordinate below is in it (default: the frame)
      seed, prewarm    prewarm: seconds lived before the layer begins (snow already falling at the first frame)
      emit:            shape: point | line | box | circle | path;  at: [x, y] (keys, formulas, layer("id").world);
                       to (line), size (box), radius + edge (circle), d | points (path, emitted along the curve);
                       rate (particles / s, animates), bursts: [[time, count], …];
                       direction (deg, 0 → right, 90 ↓, or «out»: away from `at` / across the path), spread (deg),
                       speed: n | [from, to] (px / s), inherit (share of the emitter's own velocity)
      particle:        life, size, rotation, spin: n | [from, to];  size_over_life, opacity_over_life: [v, v, …]
                       or [[u, v], …];  opacity;  color: one | several (each takes one);  color_over_life: [c, c, …];
                       sprite: disc | glow | square | ring | star | spark | streak | image (+ src);
                       align (turned along its flight), stretch (streak: seconds of flight it is long);
                       blend: normal | add (how particles lie on each other inside the layer)
      forces:          gravity: n | [x, y] (px / s²);  wind: [x, y] (px / s) with drag (1 / s);
                       turbulence: n | {strength, scale, speed} (a swirling field);
                       points: [{at, strength (+ pulls, − pushes), radius}];  vortex: [{at, strength, radius}]
      collide:         floor: y;  box: [x0, y0, x1, y1] (they stay inside);  circles: [{at, radius}];
                       segments: [[x0, y0, x1, y1], …] (ramps, shelves);  bounce, friction, kill;
                       each_other: true — particles are solid discs that push one another (Newton), up to 1500
      links:           lines between near particles: {distance, width, color, opacity, max}

    type: plexus       points that wander and the lines between near ones (Plexus), or a grid that waves (Form)
      points:          count | grid: [nx, ny];  depth (px of volume, 0: flat);  size, color, opacity, sprite;
                       drift: n | [from, to] (px / s, each its own way), edges: bounce | wrap;
                       wander: n | {amount: n | [x, y, z], scale, speed} — a smooth field that carries them (the wave
                       of a Form grid; `amount` animates: a grid bursting apart)
      links:           {distance, width, color, opacity, max}; false: none; on a grid — the mesh itself
      triangles:       n | {opacity, color}: faces where three points are all linked
      turn: [x, y, z]  the cloud turned in 3D about its centre (deg, animates), lens (eye distance, px), fade (far dims)

An emitter is a simulation: fixed steps from the layer's beginning (minus prewarm), each particle's random numbers
taken by its own number — so any process, starting anywhere, arrives at the same frame. A plexus is a formula of time.
Both renderers draw them the same way: Skia on a float canvas, one textured quad per particle."""
import collections, functools, math, os

import numpy as np
from PIL import Image

from .effects import fbm

SHAPES = ('point', 'line', 'box', 'circle', 'path')
SPRITES = ('disc', 'glow', 'square', 'ring', 'star', 'spark', 'streak', 'image')
EMIT = {'shape', 'at', 'to', 'size', 'radius', 'edge', 'd', 'points', 'closed', 'rate', 'bursts', 'direction', 'spread', 'speed', 'inherit'}
PARTICLE = {'life', 'size', 'size_over_life', 'opacity', 'opacity_over_life', 'color', 'color_over_life', 'sprite', 'src', 'rotation', 'spin',
            'align', 'stretch', 'blend'}
FORCES = {'gravity', 'wind', 'drag', 'turbulence', 'points', 'vortex'}
COLLIDE = {'floor', 'box', 'circles', 'segments', 'bounce', 'friction', 'kill', 'each_other'}
LINKS = {'distance', 'width', 'color', 'opacity', 'max'}
POINTS = {'count', 'grid', 'depth', 'size', 'color', 'opacity', 'sprite', 'drift', 'edges', 'wander', 'margin'}
EMITTER_KEYS = {'size', 'seed', 'prewarm', 'emit', 'particle', 'forces', 'collide', 'links'}
PLEXUS_KEYS = {'size', 'seed', 'points', 'links', 'triangles', 'turn', 'lens', 'fade'}
SUB = 2                         # simulation steps per frame (4 when particles push each other)
PAIR_MAX = 1500                 # particles that can push each other: every pair is looked at
LINK_MAX = 6000                 # points that can be linked
RCOLS = 12                      # random numbers each particle owns


# ---------------- the document ----------------
def _sub(F, L, key, allowed, lid):
    d = L.get(key) or {}
    if not isinstance(d, dict):
        F.issue('error', f'{key}: нужен словарь полей', lid)
        return {}
    for k in d:
        if k not in allowed:
            F.issue('error', f'{key}: нет поля «{k}»', lid, k, allowed)
    return d


def _curve(F, v, where):
    """a value along a particle's life: [v, v, …] evenly spaced, or [[u, v], …] → (us, vs)"""
    try:
        if isinstance(v, (int, float)):
            v = [v, v]
        if v and isinstance(v[0], (list, tuple)):
            pts = sorted((float(a), float(b)) for a, b in v)
            return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        vs = np.array([float(x) for x in v])
        return np.linspace(0, 1, len(vs)) if len(vs) > 1 else np.array([0.0]), vs
    except (TypeError, ValueError):
        F.issue('error', 'нужен список значений вдоль жизни: [1, 1, 0] или [[0, 0], [0.2, 1], [1, 0]]', where)
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])


def _colors(F, v, where):
    """one colour or a list of them → array (k, 3) 0..255"""
    from .doc import color
    one = not isinstance(v, (list, tuple)) or (len(v) == 3 and all(isinstance(c, (int, float)) for c in v))
    try:
        return np.array([color(c) for c in ([v] if one else v)], float)
    except ValueError as e:
        F.issue('error', str(e), where)
        return np.array([[255.0, 255, 255]])


def _links(F, v, lid, default=None):
    from .doc import color
    if v is None:
        v = default
    if v is None or v is False:
        return None
    v = {} if v is True else v
    if not isinstance(v, dict):
        F.issue('error', 'links: нужен словарь {distance, width, color, opacity, max} или false', lid)
        return None
    for k in v:
        if k not in LINKS:
            F.issue('error', f'links: нет поля «{k}»', lid, k, LINKS)
    return dict(distance=F.prop(v.get('distance', 150), f'{lid}.links.distance'), width=F.prop(v.get('width', 1.2), f'{lid}.links.width'),
                color=_colors(F, v.get('color', '#ffffff'), f'{lid}.links.color')[0], opacity=F.prop(v.get('opacity', 0.7), f'{lid}.links.opacity'),
                max=int(v.get('max', 0)))


def _field(F, items, lid, name):
    out = []
    for i, p in enumerate(items or []):
        if not isinstance(p, dict) or 'at' not in p:
            F.issue('error', f'{name}[{i}]: нужно {{at: [x, y], strength, radius}}', lid)
            continue
        for k in p:
            if k not in ('at', 'strength', 'radius'):
                F.issue('error', f'{name}[{i}]: нет поля «{k}»', lid, k, {'at', 'strength', 'radius'})
        w = f'{lid}.{name}[{i}]'
        out.append(dict(at=F.prop(p['at'], w + '.at', 2), strength=F.prop(p.get('strength', 500), w + '.strength'),
                        radius=F.prop(p.get('radius', 0), w + '.radius')))
    return out


def _stamp(F, out, *trees):
    """formulas inside the nested settings know their film (layer(), var()) and when the layer began (lt)"""
    from .doc import Prop

    def walk(v):
        if isinstance(v, Prop):
            v.film, v.t_in = F, out['in']
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    for tr in trees:
        walk(tr)


def _plane(F, L, lid, out):
    s = L.get('size', [F.W, F.H])
    if not (isinstance(s, (list, tuple)) and len(s) == 2):
        F.issue('error', 'size: нужно [w, h] — плоскость слоя', lid)
        s = [F.W, F.H]
    out['psize'] = (float(s[0]), float(s[1]))
    out['cell'] = 16.0                                         # the margin the frame's view of the plane is padded by
    out['seed'] = int(L.get('seed', 1))
    return out['psize']


def parse_emitter(F, L, lid, out):
    w, h = _plane(F, L, lid, out)
    out['prewarm'] = max(0.0, float(L.get('prewarm', 0)))
    e = _sub(F, L, 'emit', EMIT, lid)
    E = out['emit'] = {}
    E['shape'] = e.get('shape', 'point')
    if E['shape'] not in SHAPES:
        F.issue('error', f'emit.shape «{E["shape"]}»', lid, E['shape'], SHAPES)
        E['shape'] = 'point'
    E['at'] = F.prop(e.get('at', [0, 0] if E['shape'] == 'path' else [w / 2, h / 2]), f'{lid}.emit.at', 2)
    E['to'] = F.prop(e.get('to', [w / 2 + 100, h / 2]), f'{lid}.emit.to', 2)
    if E['shape'] == 'line' and 'to' not in e:
        F.issue('error', 'emit.shape line: нужен to: [x, y] — второй конец отрезка', lid)
    E['size'] = F.prop(e.get('size', [200, 100]), f'{lid}.emit.size', 2)
    E['radius'] = F.prop(e.get('radius', 60), f'{lid}.emit.radius')
    E['edge'] = bool(e.get('edge', False))
    E['path'] = None
    if E['shape'] == 'path':
        import skia
        from .render import svg_path
        from .paths import _points_path
        if 'd' in e:
            p = svg_path(str(e['d']))
        elif 'points' in e:
            p = _points_path([tuple(map(float, q)) for q in e['points']], bool(e.get('closed', False)))
        else:
            F.issue('error', 'emit.shape path: нужен d: "M …" или points: [[x, y], …]', lid)
            p = skia.Path()
        pts, m = [], skia.PathMeasure(p, False)
        total = 0.0
        while True:                                            # every contour, evenly along its length
            ln = m.getLength()
            if ln > 0:
                k = max(2, int(math.ceil(ln / 2)))
                for d in np.linspace(0, ln, k, endpoint=False):
                    q, tg = m.getPosTan(float(d))
                    pts.append((q.x(), q.y(), tg.x(), tg.y()))
                total += ln
            if not m.nextContour():
                break
        if not pts:
            F.issue('error', 'emit.shape path: кривая пустая', lid)
            pts = [(w / 2, h / 2, 1.0, 0.0)]
        E['path'] = np.array(pts, float)
    E['rate'] = F.prop(e.get('rate', 0 if 'bursts' in e else 100), f'{lid}.emit.rate')
    E['bursts'] = []
    for b in e.get('bursts') or []:
        if not (isinstance(b, (list, tuple)) and len(b) == 2):
            F.issue('error', 'emit.bursts: нужно [[время, сколько], …]', lid)
            continue
        E['bursts'].append((F.time(b[0], lid), int(b[1])))
    d = e.get('direction', -90)
    E['out'] = isinstance(d, str) and d.strip() == 'out'
    E['direction'] = F.prop(0 if E['out'] else d, f'{lid}.emit.direction')
    E['spread'] = F.prop(e.get('spread', 0 if E['out'] else 30), f'{lid}.emit.spread')
    E['speed'] = F.prop(e.get('speed', [100, 200]), f'{lid}.emit.speed')
    E['inherit'] = float(e.get('inherit', 0))

    p = _sub(F, L, 'particle', PARTICLE, lid)
    P = out['particle'] = {}
    P['life'] = F.prop(p.get('life', [1, 2]), f'{lid}.particle.life')
    P['size'] = F.prop(p.get('size', [4, 8]), f'{lid}.particle.size')
    P['size_curve'] = _curve(F, p.get('size_over_life', [1, 1]), f'{lid}.particle.size_over_life')
    P['opacity'] = F.prop(p.get('opacity', 1), f'{lid}.particle.opacity')
    P['opacity_curve'] = _curve(F, p.get('opacity_over_life', [1, 1, 0]), f'{lid}.particle.opacity_over_life')
    P['colors'] = _colors(F, p.get('color', '#ffffff'), f'{lid}.particle.color')
    P['ramp'] = _colors(F, p['color_over_life'], f'{lid}.particle.color_over_life') if p.get('color_over_life') else None
    P['sprite'] = p.get('sprite', 'image' if 'src' in p else 'disc')
    if P['sprite'] not in SPRITES:
        F.issue('error', f'particle.sprite «{P["sprite"]}»', lid, P['sprite'], SPRITES)
        P['sprite'] = 'disc'
    P['src'] = None
    if P['sprite'] == 'image':
        if 'src' not in p:
            F.issue('error', 'particle.sprite image: нужен src — картинка частицы', lid)
            P['sprite'] = 'disc'
        else:
            P['src'] = F.resolve(p['src'])
            if not os.path.exists(P['src']):
                F.issue('error', f'нет файла {p["src"]}', lid)
                P['sprite'], P['src'] = 'disc', None
    P['rotation'] = F.prop(p.get('rotation', 0), f'{lid}.particle.rotation')
    P['spin'] = F.prop(p.get('spin', 0), f'{lid}.particle.spin')
    P['align'] = bool(p.get('align', P['sprite'] == 'streak'))
    P['stretch'] = float(p.get('stretch', 0.05))
    P['blend'] = p.get('blend', 'add' if P['sprite'] in ('glow', 'spark', 'streak') else 'normal')
    if P['blend'] not in ('add', 'normal'):
        F.issue('error', f'particle.blend «{P["blend"]}»: add или normal', lid)
        P['blend'] = 'normal'

    f = _sub(F, L, 'forces', FORCES, lid)
    G = out['forces'] = {}
    g = f.get('gravity', 0)
    G['gravity'] = F.prop([0, g] if isinstance(g, (int, float)) else g, f'{lid}.forces.gravity')
    G['wind'] = F.prop(f.get('wind', [0, 0]), f'{lid}.forces.wind', 2)
    G['drag'] = F.prop(f.get('drag', 1.0 if 'wind' in f else 0), f'{lid}.forces.drag')
    tb = f.get('turbulence')
    G['turbulence'] = None
    if tb:
        tb = {'strength': tb} if not isinstance(tb, dict) or any(not isinstance(k, str) or k == 'ease' for k in tb) else tb
        for k in tb:
            if k not in ('strength', 'scale', 'speed'):
                F.issue('error', f'forces.turbulence: нет поля «{k}»', lid, k, {'strength', 'scale', 'speed'})
        G['turbulence'] = dict(strength=F.prop(tb.get('strength', 300), f'{lid}.forces.turbulence.strength'),
                               scale=max(1.0, float(tb.get('scale', 200))), speed=float(tb.get('speed', 0.3)))
    G['points'] = _field(F, f.get('points'), lid, 'forces.points')
    G['vortex'] = _field(F, f.get('vortex'), lid, 'forces.vortex')

    out['collide'] = None
    if L.get('collide'):
        c = _sub(F, L, 'collide', COLLIDE, lid)
        C = out['collide'] = dict(floor=F.prop(c['floor'], f'{lid}.collide.floor') if 'floor' in c else None,
                                  box=F.prop(c['box'], f'{lid}.collide.box', 4) if 'box' in c else None, circles=[], segments=[],
                                  bounce=float(c.get('bounce', 0.5)), friction=float(c.get('friction', 0.1)), kill=bool(c.get('kill', False)),
                                  each_other=bool(c.get('each_other', False)))
        for i, q in enumerate(c.get('circles') or []):
            if not isinstance(q, dict) or 'at' not in q:
                F.issue('error', f'collide.circles[{i}]: нужно {{at: [x, y], radius}}', lid)
                continue
            C['circles'].append(dict(at=F.prop(q['at'], f'{lid}.collide.circles[{i}].at', 2), radius=F.prop(q.get('radius', 50), f'{lid}.collide.circles[{i}].radius')))
        for i, q in enumerate(c.get('segments') or []):
            C['segments'].append(F.prop(q, f'{lid}.collide.segments[{i}]', 4))
    out['links'] = _links(F, L.get('links'), lid)
    _stamp(F, out, E, P, G, out['collide'], out['links'])
    # how many live at once, roughly: a slip of the pen (rate 100000) should be said, not rendered for an hour
    try:
        rate = max([E['rate'].value or 0] if not E['rate'].keys else [k[1] for k in E['rate'].keys]) if E['rate'].expr is None else 0
        life = P['life'].value if P['life'].keys is None and P['life'].expr is None else None
        life = max(life) if isinstance(life, tuple) else (life or 0)
        alive = rate * life + max([b[1] for b in E['bursts']] or [0])
        if alive > 60000:
            F.issue('warn', f'около {alive:.0f} частиц одновременно (rate × life): кадр будет рисоваться долго', lid)
        if out['collide'] and out['collide']['each_other'] and alive > PAIR_MAX:
            F.issue('warn', f'collide.each_other: около {alive:.0f} частиц одновременно, а толкаться могут до {PAIR_MAX} — остальные пройдут насквозь', lid)
    except (TypeError, ValueError):
        pass


def parse_plexus(F, L, lid, out):
    w, h = _plane(F, L, lid, out)
    p = _sub(F, L, 'points', POINTS, lid)
    Q = out['points'] = {}
    Q['grid'] = None
    if 'grid' in p:
        g = p['grid']
        if not (isinstance(g, (list, tuple)) and len(g) == 2 and int(g[0]) >= 2 and int(g[1]) >= 2):
            F.issue('error', 'points.grid: нужно [nx, ny], не меньше 2 × 2', lid)
        else:
            Q['grid'] = (int(g[0]), int(g[1]))
    Q['count'] = Q['grid'][0] * Q['grid'][1] if Q['grid'] else max(1, int(p.get('count', 80)))
    if Q['count'] > LINK_MAX:
        F.issue('error', f'points: {Q["count"]} точек — больше {LINK_MAX}', lid)
        Q['count'], Q['grid'] = LINK_MAX, None
    Q['depth'] = max(0.0, float(p.get('depth', 0)))
    Q['margin'] = float(p.get('margin', 0))
    Q['size'] = F.prop(p.get('size', [2, 5]), f'{lid}.points.size')
    Q['colors'] = _colors(F, p.get('color', '#ffffff'), f'{lid}.points.color')
    Q['opacity'] = F.prop(p.get('opacity', 1), f'{lid}.points.opacity')
    Q['sprite'] = p.get('sprite', 'disc')
    if Q['sprite'] not in ('disc', 'glow', 'square', 'ring', 'star', 'spark'):
        F.issue('error', f'points.sprite «{Q["sprite"]}»', lid, Q['sprite'], ('disc', 'glow', 'square', 'ring', 'star', 'spark'))
        Q['sprite'] = 'disc'
    d = p.get('drift', 0 if Q['grid'] else [5, 25])
    Q['drift'] = (float(d), float(d)) if isinstance(d, (int, float)) else (float(d[0]), float(d[1]))
    Q['edges'] = p.get('edges', 'bounce')
    if Q['edges'] not in ('bounce', 'wrap'):
        F.issue('error', f'points.edges «{Q["edges"]}»: bounce или wrap', lid)
        Q['edges'] = 'bounce'
    wd = p.get('wander')
    Q['wander'] = None
    if wd:
        wd = {'amount': wd} if not isinstance(wd, dict) or any(not isinstance(k, str) or k == 'ease' for k in wd) else wd
        for k in wd:
            if k not in ('amount', 'scale', 'speed'):
                F.issue('error', f'points.wander: нет поля «{k}»', lid, k, {'amount', 'scale', 'speed'})
        Q['wander'] = dict(amount=F.prop(wd.get('amount', 40), f'{lid}.points.wander.amount'), scale=max(1.0, float(wd.get('scale', 300))),
                           speed=float(wd.get('speed', 0.3)))
    out['links'] = _links(F, L.get('links'), lid, default={})
    tr = L.get('triangles')
    out['triangles'] = None
    if tr:
        tr = {'opacity': tr} if isinstance(tr, (int, float)) else tr
        for k in tr:
            if k not in ('opacity', 'color'):
                F.issue('error', f'triangles: нет поля «{k}»', lid, k, {'opacity', 'color'})
        out['triangles'] = dict(opacity=F.prop(tr.get('opacity', 0.12), f'{lid}.triangles.opacity'),
                                color=_colors(F, tr['color'], f'{lid}.triangles.color')[0] if 'color' in tr else None)
        if out['links'] is None:
            F.issue('error', 'triangles: грани стоят на связях — нужен links', lid)
            out['triangles'] = None
    out['turn'] = F.prop(L.get('turn', [0, 0, 0]), f'{lid}.turn', 3)
    out['lens'] = max(50.0, float(L.get('lens', 1200)))
    out['fade'] = float(L.get('fade', 0.6 if (Q['depth'] or 'turn' in L) else 0))
    _stamp(F, out, Q, out['links'], out['triangles'], {'turn': out['turn']})


# ---------------- the emitter's simulation ----------------
def _pair(v):
    return (float(v[0]), float(v[1])) if isinstance(v, tuple) else (float(v), float(v))


def _vec2(v):
    return np.array([0.0, float(v)]) if not isinstance(v, tuple) else np.array([float(v[0]), float(v[1])])


def _curl(x, y, t, scale, speed, seed):
    """a swirling field with no sources or sinks (particles in it neither bunch nor thin out): the curl of a smooth
    noise that slides with time → (ax, ay), about ±1"""
    u, v, e = x / scale, y / scale, 0.05
    s = t * speed

    def psi(a, b):
        return fbm(a + 0.37 * s, b - 0.61 * s, seed + 31, 2) + 0.5 * fbm(a * 1.9 - 0.53 * s, b * 1.9 + 0.29 * s, seed + 47, 2)
    return (psi(u, v + e) - psi(u, v - e)) / (2 * e) * 0.6, -(psi(u + e, v) - psi(u - e, v)) / (2 * e) * 0.6


class Sim:
    """the emitter's particles at any time: stepped from the beginning, snapshots kept so going back a little (echo,
    a repeated frame) costs nothing"""
    FIELDS = ('pos', 'vel', 'birth', 'life', 'idx', 'size0', 'rot0', 'spin')

    def __init__(self, L, fps):
        self.L = L
        self.h = 1.0 / (fps * (4 if (L['collide'] or {}).get('each_other') else SUB))
        self.t0 = L['in'] - L['prewarm']
        self._blocks, self.snaps = {}, collections.OrderedDict()
        self.reset()

    def reset(self):
        self.k, self.count, self.cum = 0, 0, 0.5
        self.S = dict(pos=np.zeros((0, 2)), vel=np.zeros((0, 2)), birth=np.zeros(0), life=np.zeros(0), idx=np.zeros(0, np.int64),
                      size0=np.zeros(0), rot0=np.zeros(0), spin=np.zeros(0))

    def rand(self, idx):
        """the random numbers of particles, by their numbers: the same whatever else was born"""
        out = np.empty((len(idx), RCOLS))
        blk = idx // 4096
        for b in np.unique(blk):
            T = self._blocks.get(int(b))
            if T is None:
                T = self._blocks[int(b)] = np.random.default_rng([self.L['seed'], int(b)]).random((4096, RCOLS))
            m = blk == b
            out[m] = T[idx[m] % 4096]
        return out

    def radius(self, S, t):
        us, vs = self.L['particle']['size_curve']
        return S['size0'] * np.interp(np.clip((t - S['birth']) / np.maximum(S['life'], 1e-9), 0, 1), us, vs) / 2

    def at(self, t):
        """(the state at the last step not after t, seconds from that step to t) or None before the beginning"""
        K = int(math.floor((t - self.t0) / self.h + 1e-6))
        if K < 0:
            return None
        hit = self.snaps.get(K)
        if hit is None:
            if K < self.k:
                below = [q for q in self.snaps if q < K]
                if below:
                    S, self.cum, self.count = self.snaps[max(below)]
                    self.S, self.k = {k: v.copy() for k, v in S.items()}, max(below)
                else:
                    self.reset()
            while self.k < K:
                self.step()
            hit = self.snaps[K] = ({k: v.copy() for k, v in self.S.items()}, self.cum, self.count)
            while len(self.snaps) > 64:
                self.snaps.popitem(last=False)
        else:
            self.snaps.move_to_end(K)
        return hit[0], t - (self.t0 + K * self.h)

    def step(self):
        h = self.h
        t0 = self.t0 + self.k * h
        t1 = t0 + h
        S = self.S
        if len(S['idx']):
            self.move(S, t0, h)
        self.emit(S, t0, t1)
        keep = (t1 - S['birth']) < S['life']
        if not keep.all():
            for k in self.FIELDS:
                S[k] = S[k][keep]
        self.k += 1

    def move(self, S, t, h):
        L = self.L
        G = L['forces']
        pos, vel = S['pos'], S['vel']
        g = _vec2(G['gravity'].at(t))
        acc = np.broadcast_to(g, pos.shape).copy()
        tb = G['turbulence']
        if tb:
            s = tb['strength'].at(t)
            if s:
                ax, ay = _curl(pos[:, 0], pos[:, 1], t, tb['scale'], tb['speed'], L['seed'])
                acc[:, 0] += s * ax
                acc[:, 1] += s * ay
        for kind in ('points', 'vortex'):
            for p in G[kind]:
                c, st, R = np.array(p['at'].at(t)), p['strength'].at(t), p['radius'].at(t)
                d = c[None] - pos
                dist = np.hypot(d[:, 0], d[:, 1]) + 1e-6
                f = st * (np.clip(1 - dist / R, 0, 1) if R > 0 else 1.0) / dist
                if kind == 'points':
                    acc[:, 0] += d[:, 0] * f
                    acc[:, 1] += d[:, 1] * f
                else:                                          # around the point: clockwise on screen when positive
                    acc[:, 0] += d[:, 1] * f
                    acc[:, 1] -= d[:, 0] * f
        was = vel.copy()
        vel += h * acc
        drag = G['drag'].at(t)
        if drag > 0:                                           # the air's own speed is `wind`; drag pulls towards it
            wind = np.array(G['wind'].at(t))
            vel[:] = wind + (vel - wind) * math.exp(-drag * h)
        prev = pos.copy()
        pos += h * (was + vel) / 2                             # exact under a constant force (a thrown stone's parabola)
        if L['collide']:
            self.collide(S, prev, t + h, h, g)

    def collide(self, S, prev, t, h, g):
        C = self.L['collide']
        pos, vel = S['pos'], S['vel']
        r = self.radius(S, t)
        e, fr = C['bounce'], C['friction']
        rest = 2.5 * h * float(np.hypot(*g)) + 1e-9            # slower than this into a surface: it lies on it
        hit = np.zeros(len(pos), bool)

        def reflect(m, nx, ny):
            vn = vel[m, 0] * nx + vel[m, 1] * ny
            into = vn < 0
            tx, ty = vel[m, 0] - vn * nx, vel[m, 1] - vn * ny
            out = np.where(into, np.where(-vn < rest, 0.0, -vn * e), vn)
            # friction as between real things: the harder the hit, the more sideways speed it takes (a share `friction`
            # of the push). Lying on a slope the push is gravity's, so a ball slides when the slope is steeper than
            # the friction — v1 took a share of the sideways speed at every touch, a hundred times a second, and
            # balls stuck to a shelf they should have rolled down
            ts = np.hypot(tx, ty)
            f = np.where(into, np.maximum(0.0, ts - fr * (out - vn)) / np.maximum(ts, 1e-9), 1.0)
            vel[m, 0], vel[m, 1] = tx * f + out * nx, ty * f + out * ny
            hit[m] = True

        def wall(nx, ny, depth, bounce=True):                  # a half-plane: the normal points out of the solid
            m = depth > 0
            if m.any():
                pos[m, 0] += nx * depth[m]
                pos[m, 1] += ny * depth[m]
                if bounce:
                    reflect(m, nx, ny)

        def walls(bounce=True):
            if C['floor'] is not None:
                wall(0.0, -1.0, pos[:, 1] + r - C['floor'].at(t), bounce)
            if C['box'] is not None:
                x0, y0, x1, y1 = C['box'].at(t)
                wall(1.0, 0.0, x0 - (pos[:, 0] - r), bounce)
                wall(-1.0, 0.0, pos[:, 0] + r - x1, bounce)
                wall(0.0, 1.0, y0 - (pos[:, 1] - r), bounce)
                wall(0.0, -1.0, pos[:, 1] + r - y1, bounce)

        walls()
        for q in C['circles']:
            c, R = np.array(q['at'].at(t)), q['radius'].at(t)
            d = pos - c[None]
            dist = np.hypot(d[:, 0], d[:, 1])
            m = dist < R + r
            if m.any():
                ds = np.maximum(dist[m], 1e-6)
                nx, ny = d[m, 0] / ds, d[m, 1] / ds
                pos[m, 0], pos[m, 1] = c[0] + nx * (R + r[m]), c[1] + ny * (R + r[m])
                reflect(m, nx, ny)
        for q in C['segments']:
            x0, y0, x1, y1 = q.at(t)
            ab = np.array([x1 - x0, y1 - y0])
            ln2 = float(ab @ ab) or 1e-9
            ln = math.sqrt(ln2)
            px, py = -ab[1] / ln, ab[0] / ln                   # the segment's own normal
            s0 = (prev[:, 0] - x0) * px + (prev[:, 1] - y0) * py
            s1 = (pos[:, 0] - x0) * px + (pos[:, 1] - y0) * py
            cross = (s0 * s1 < 0)                              # flew through it within the step: stopped where it crossed
            if cross.any():
                u = s0[cross] / (s0[cross] - s1[cross])
                hx = prev[cross, 0] + u * (pos[cross, 0] - prev[cross, 0])
                hy = prev[cross, 1] + u * (pos[cross, 1] - prev[cross, 1])
                tt = ((hx - x0) * ab[0] + (hy - y0) * ab[1]) / ln2
                ok = (tt >= 0) & (tt <= 1)
                ii = np.nonzero(cross)[0][ok]
                if len(ii):
                    sg = np.sign(s0[ii])
                    pos[ii, 0], pos[ii, 1] = hx[ok] + sg * px * r[ii], hy[ok] + sg * py * r[ii]
                    reflect(ii, sg * px, sg * py)
            tt = np.clip(((pos[:, 0] - x0) * ab[0] + (pos[:, 1] - y0) * ab[1]) / ln2, 0, 1)
            dx, dy = pos[:, 0] - (x0 + tt * ab[0]), pos[:, 1] - (y0 + tt * ab[1])
            dist = np.hypot(dx, dy)
            m = dist < r
            if m.any():
                ds = np.maximum(dist[m], 1e-6)
                nx, ny = dx[m] / ds, dy[m] / ds
                pos[m, 0] += nx * (r[m] - dist[m])
                pos[m, 1] += ny * (r[m] - dist[m])
                reflect(m, nx, ny)
        n = len(pos)
        if C['each_other'] and 1 < n <= PAIR_MAX:
            # solid discs. Positions first: every overlap is pushed apart (the lighter one moves more), a few rounds;
            # then each touched disc's velocity is what it really moved by — that is what lets a pile lie still
            # (pushing velocities pair by pair left the pile trembling and sinking into itself) — and the pairs that
            # met at speed get their bounce back
            mass = np.maximum(r * r, 1e-9)
            share = mass[None, :] / (mass[:, None] + mass[None, :])   # how much of a push one takes: the other's weight
            before = vel.copy()
            touched = np.zeros(n, bool)
            met = None
            for it in range(10):
                dx, dy = pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1]
                dist = np.hypot(dx, dy)
                np.fill_diagonal(dist, 1e9)
                over = r[:, None] + r[None, :] - dist
                tch = over > 0
                if not tch.any():
                    break
                ds = np.maximum(dist, 1e-6)
                nx, ny = dx / ds, dy / ds
                same = tch & (dist < 1e-6)                     # born in one spot: pushed apart by their numbers
                if same.any():
                    a = (S['idx'][:, None] * 0.618 + S['idx'][None, :] * 0.382) * 2 * math.pi
                    sgn = np.sign(S['idx'][:, None] - S['idx'][None, :])
                    nx, ny = np.where(same, np.cos(a) * sgn, nx), np.where(same, np.sin(a) * sgn, ny)
                if it == 0:
                    vr = (before[:, None, 0] - before[None, :, 0]) * nx + (before[:, None, 1] - before[None, :, 1]) * ny
                    met = (tch & (vr < -max(rest, 30.0)), vr, nx, ny)
                touched |= tch.any(1)
                cnt = np.maximum(1, tch.sum(1))
                w = np.where(tch, over * share, 0.0)
                pos[:, 0] += (w * nx).sum(1) / np.sqrt(cnt)
                pos[:, 1] += (w * ny).sum(1) / np.sqrt(cnt)
                walls(False)
            if touched.any():
                vel[touched] = (pos[touched] - prev[touched]) / h
                ap, vr, nx, ny = met
                if e > 0 and ap.any():
                    j = np.where(ap, -e * vr * share, 0.0)
                    cnt = np.maximum(1, ap.sum(1))
                    vel[:, 0] += (j * nx).sum(1) / cnt
                    vel[:, 1] += (j * ny).sum(1) / cnt
        if C['kill'] and hit.any():
            S['life'][hit] = 0.0

    def emit(self, S, t0, t1):
        L = self.L
        E, P, h = L['emit'], L['particle'], self.h
        if t0 >= L['out']:
            return
        fr = []
        rate = max(0.0, float(E['rate'].at((t0 + t1) / 2) or 0))
        if rate > 0:
            c0 = self.cum
            c1 = c0 + rate * h
            m = np.arange(math.floor(c0) + 1, math.floor(c1) + 1)
            fr.append((m - c0) / (rate * h))
            self.cum = c1
        for tb, cnt in E['bursts']:
            if t0 <= tb < t1 and cnt > 0:
                fr.append(np.full(cnt, (tb - t0) / h))
        f = np.concatenate(fr) if fr else np.zeros(0)
        k = len(f)
        if not k:
            return
        idx = self.count + np.arange(k, dtype=np.int64)
        self.count += k
        R = self.rand(idx)
        a0, a1 = np.array(E['at'].at(t0)), np.array(E['at'].at(t1))
        c = a0[None] + (a1 - a0)[None] * f[:, None]            # where the emitter was at each one's own moment
        ev = (a1 - a0) / h
        off = np.zeros((k, 2))
        nrm = None
        sh = E['shape']
        if sh == 'line':
            b0, b1 = np.array(E['to'].at(t0)), np.array(E['to'].at(t1))
            b = b0[None] + (b1 - b0)[None] * f[:, None]
            off = (b - c) * R[:, :1]
        elif sh == 'box':
            w, hh = E['size'].at(t0)
            if E['edge']:
                per = R[:, 0] * 2 * (w + hh)
                x = np.where(per < w, per, np.where(per < w + hh, w, np.where(per < 2 * w + hh, 2 * w + hh - per, 0.0)))
                y = np.where(per < w, 0.0, np.where(per < w + hh, per - w, np.where(per < 2 * w + hh, hh, 2 * (w + hh) - per)))
                off = np.stack([x - w / 2, y - hh / 2], -1)
            else:
                off = np.stack([(R[:, 0] - 0.5) * w, (R[:, 1] - 0.5) * hh], -1)
        elif sh == 'circle':
            rad = E['radius'].at(t0) * (1.0 if E['edge'] else np.sqrt(R[:, 1]))
            a = 2 * math.pi * R[:, 0]
            off = np.stack([rad * np.cos(a), rad * np.sin(a)], -1)
        elif sh == 'path':
            pth = E['path']
            q = pth[np.minimum(len(pth) - 1, (R[:, 0] * len(pth)).astype(np.int64))]
            off = q[:, :2]
            nrm = np.arctan2(-q[:, 2], q[:, 3])               # across the curve (to the left of its travel)
        if E['out']:
            if nrm is not None:
                base = nrm
            else:
                far = np.hypot(off[:, 0], off[:, 1]) > 1e-6
                base = np.where(far, np.arctan2(off[:, 1], off[:, 0]), 2 * math.pi * R[:, 10])
        else:
            base = math.radians(E['direction'].at(t0))
        ang = base + math.radians(E['spread'].at(t0)) * (R[:, 2] - 0.5)
        lo, hi = _pair(E['speed'].at(t0))
        sp = lo + (hi - lo) * R[:, 3]
        vel = np.stack([sp * np.cos(ang), sp * np.sin(ang)], -1) + E['inherit'] * ev[None]
        rest = ((1 - f) * h)[:, None]                          # born inside the step: already flew the rest of it
        g = _vec2(L['forces']['gravity'].at(t0))[None]
        pos = c + off + vel * rest + 0.5 * g * rest * rest
        vel = vel + g * rest
        lo, hi = _pair(P['life'].at(t0))
        life = np.maximum(1e-3, lo + (hi - lo) * R[:, 4])
        lo, hi = _pair(P['size'].at(t0))
        size0 = lo + (hi - lo) * R[:, 5]
        lo, hi = _pair(P['rotation'].at(t0))
        rot0 = lo + (hi - lo) * R[:, 7]
        lo, hi = _pair(P['spin'].at(t0))
        spin = lo + (hi - lo) * R[:, 8]
        new = dict(pos=pos, vel=vel, birth=t0 + f * h, life=life, idx=idx, size0=size0, rot0=rot0, spin=spin)
        for kk in self.FIELDS:
            S[kk] = np.concatenate([S[kk], new[kk]])


# ---------------- what to draw ----------------
def emitter_data(L, t, sim, half_shutter):
    """the emitter's particles at t as things to draw: dict(x, y, hx, hy, ang, rgb, alpha) + links"""
    got = sim.at(t)
    if got is None:
        return None
    S, dt = got
    n = len(S['idx'])
    if not n:
        return None
    P = L['particle']
    pos = S['pos'] + S['vel'] * dt
    age = t - S['birth']
    u = np.clip(age / np.maximum(S['life'], 1e-9), 0, 1)
    size = S['size0'] * np.interp(u, *P['size_curve'])
    alpha = np.clip(P['opacity'].at(t) * np.interp(u, *P['opacity_curve']), 0, 1)
    R = sim.rand(S['idx'])
    if P['ramp'] is not None:
        us = np.linspace(0, 1, len(P['ramp']))
        rgb = np.stack([np.interp(u, us, P['ramp'][:, c]) for c in range(3)], -1)
    else:
        rgb = P['colors'][np.minimum(len(P['colors']) - 1, (R[:, 6] * len(P['colors'])).astype(np.int64))]
    speed = np.hypot(S['vel'][:, 0], S['vel'][:, 1])
    heading = np.arctan2(S['vel'][:, 1], S['vel'][:, 0])
    ang = np.radians(S['rot0'] + S['spin'] * age)
    hx = hy = size / 2
    round_ = P['sprite'] in ('disc', 'glow', 'ring', 'streak')
    long = np.zeros(n)
    if P['sprite'] == 'streak':
        long = long + speed * P['stretch']
    if half_shutter > 0 and round_:                            # motion blur: smeared along its flight within the shutter
        long = long + speed * 2 * half_shutter
    if round_ and long.any():
        ang = np.where(long > 0.25, heading, ang)
        hx = size / 2 + long / 2
        if P['sprite'] != 'streak':
            alpha = alpha * size / np.maximum(size + long, 1e-9)
    elif P['align']:
        ang = heading
    keep = (alpha > 1 / 512) & (size > 1e-3)
    D = dict(x=pos[keep, 0], y=pos[keep, 1], hx=(hx[keep] if np.ndim(hx) else hx), hy=(hy[keep] if np.ndim(hy) else hy), ang=ang[keep],
             rgb=rgb[keep], alpha=alpha[keep], sprite=P['sprite'], src=P['src'], add=P['blend'] == 'add', links=None, tris=None)
    K = L['links']
    if K is not None and keep.sum() > 1:
        D['links'] = _linked(np.stack([D['x'], D['y']], -1), K, t, np.ones(int(keep.sum())))
    return D


@functools.lru_cache(32)
def _homes(seed, n, grid, w, h, depth, margin):
    R = np.random.default_rng([seed, 77]).random((n, 10))
    if grid:
        nx, ny = grid
        gx, gy = np.meshgrid(np.linspace(margin, w - margin, nx), np.linspace(margin, h - margin, ny))
        home = np.stack([gx.ravel(), gy.ravel(), np.zeros(n)], -1)
    else:
        home = np.stack([R[:, 0] * w, R[:, 1] * h, (R[:, 2] - 0.5) * depth], -1)
    a, z = 2 * math.pi * R[:, 3], (R[:, 4] * 2 - 1) if depth else np.zeros(n)
    s = np.sqrt(1 - z * z)
    return R, home, np.stack([s * np.cos(a), s * np.sin(a), z], -1)


def _fold(p, lo, hi, wrap):
    span = hi - lo
    if span <= 0:
        return np.full_like(p, lo)
    q = (p - lo) % (span if wrap else 2 * span)
    return lo + (q if wrap else span - np.abs(q - span))


def plexus_data(L, t):
    Q = L['points']
    w, h = L['psize']
    D, n, lt = Q['depth'], Q['count'], t - L['in']
    R, home, dirs = _homes(L['seed'], n, Q['grid'], w, h, D, Q['margin'])
    p = home.copy()
    lo, hi = Q['drift']
    if hi > 0 or lo > 0:
        p = p + dirs * ((lo + (hi - lo) * R[:, 5]) * lt)[:, None]
        wrap = Q['edges'] == 'wrap'
        p[:, 0], p[:, 1], p[:, 2] = _fold(p[:, 0], 0, w, wrap), _fold(p[:, 1], 0, h, wrap), _fold(p[:, 2], -D / 2, D / 2, wrap)
    wd = Q['wander']
    if wd:
        a = wd['amount'].at(t)
        a = (a, a, a) if not isinstance(a, tuple) else tuple(a) + (0.0,) * (3 - len(a))
        u, v, z = home[:, 0] / wd['scale'], home[:, 1] / wd['scale'], home[:, 2] / wd['scale']
        s = t * wd['speed']
        for c, (du, dv) in enumerate(((11.3, 0.0), (0.0, 23.7), (37.1, 5.9))):
            if a[c]:
                p[:, c] += (fbm(u + du + z + (s if c != 0 else 0), v + dv + (s if c == 0 else 0.37 * s), L['seed'] + 3 + 5 * c) - 0.5) * 3 * a[c]
    K = L['links']
    lk = tris = None
    size = _pair(Q['size'].at(t))
    size = size[0] + (size[1] - size[0]) * R[:, 6]
    # the cloud turned about its centre and seen through a lens
    rx, ry, rz = (math.radians(v) for v in L['turn'].at(t))
    q = p - np.array([w / 2, h / 2, 0.0])
    if rx:
        q = q @ np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]]).T
    if ry:
        q = q @ np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]]).T
    if rz:
        q = q @ np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]]).T
    f = L['lens']
    flat = not (D or rx or ry or (wd and q[:, 2].any()))
    s = np.ones(n) if flat else f / np.maximum(f + q[:, 2], 0.1 * f)
    X, Y = w / 2 + q[:, 0] * s, h / 2 + q[:, 1] * s
    zr = math.sqrt(w * w + h * h + D * D) / 2
    dim = np.ones(n) if flat or not L['fade'] else 1 - L['fade'] * np.clip(0.5 + q[:, 2] / (2 * zr), 0, 1)
    if K is not None and n > 1:
        if Q['grid']:
            nx, ny = Q['grid']
            ii = np.arange(n).reshape(ny, nx)
            a = np.concatenate([ii[:, :-1].ravel(), ii[:-1, :].ravel()])
            b = np.concatenate([ii[:, 1:].ravel(), ii[1:, :].ravel()])
            op = K['opacity'].at(t)
            lk = dict(i=a, j=b, alpha=op * np.minimum(dim[a], dim[b]), width=K['width'].at(t), rgb=K['color'])
            if L['triangles']:
                c00, c10, c01, c11 = ii[:-1, :-1].ravel(), ii[:-1, 1:].ravel(), ii[1:, :-1].ravel(), ii[1:, 1:].ravel()
                tri = np.concatenate([np.stack([c00, c10, c11], -1), np.stack([c00, c11, c01], -1)])
                tris = dict(tri=tri, alpha=L['triangles']['opacity'].at(t) * dim[tri].min(1))
        else:
            lk = _linked(p, K, t, dim)
            if L['triangles'] and len(lk['i']):
                tris = _faces(n, lk, L['triangles']['opacity'].at(t), K['opacity'].at(t))
        if tris is not None:
            tris['rgb'] = L['triangles']['color'] if L['triangles']['color'] is not None else K['color']
    order = np.argsort(-q[:, 2], kind='stable')                # far ones first
    rgb = Q['colors'][np.minimum(len(Q['colors']) - 1, (R[:, 7] * len(Q['colors'])).astype(np.int64))]
    alpha = np.clip(Q['opacity'].at(t) * dim, 0, 1)
    hs = size * s / 2
    return dict(x=X[order], y=Y[order], hx=hs[order], hy=hs[order], ang=np.zeros(n), rgb=rgb[order], alpha=alpha[order], sprite=Q['sprite'],
                src=None, add=False, links=lk, tris=tris, lx=X, ly=Y)


def near_pairs(P, dist):
    """every pair i < j of points (n, 2 | 3) closer than dist → (i, j, distance)"""
    n = len(P)
    if n < 2 or dist <= 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0)
    if n <= 1200:
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        i, j = np.nonzero(np.triu(d2 < dist * dist, 1))
        return i, j, np.sqrt(d2[i, j])
    cell = np.floor(P[:, :2] / dist).astype(np.int64)          # binned by cells of `dist`: only neighbours are compared
    bins = collections.defaultdict(list)
    for k, (cx, cy) in enumerate(map(tuple, cell)):
        bins[(cx, cy)].append(k)
    I, J = [], []
    for (cx, cy), a in bins.items():
        a = np.array(a)
        for dx, dy in ((0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            b = a if (dx, dy) == (0, 0) else bins.get((cx + dx, cy + dy))
            if b is None:
                continue
            b = np.asarray(b)
            d2 = ((P[a][:, None, :] - P[b][None, :, :]) ** 2).sum(-1)
            m = d2 < dist * dist
            if (dx, dy) == (0, 0):
                m = np.triu(m, 1)
            ia, ib = np.nonzero(m)
            I.append(a[ia])
            J.append(b[ib])
    i, j = np.concatenate(I), np.concatenate(J)
    i, j = np.minimum(i, j), np.maximum(i, j)
    o = np.lexsort((j, i))
    i, j = i[o], j[o]
    return i, j, np.sqrt(((P[i] - P[j]) ** 2).sum(-1))


def _linked(P, K, t, dim):
    """the lines between near points: fainter the farther apart, at most `max` of the nearest per point"""
    dist = float(K['distance'].at(t))
    i, j, d = near_pairs(P[:LINK_MAX], dist)
    if K['max'] > 0 and len(i):
        m = len(i)
        ends, dd = np.concatenate([i, j]), np.concatenate([d, d])   # every pair stands in the list of both its points
        o = np.lexsort((dd, ends))
        first = np.r_[0, np.nonzero(np.diff(ends[o]))[0] + 1]
        start = np.repeat(first, np.diff(np.r_[first, len(ends)]))
        rank = np.empty(len(ends), np.int64)
        rank[o] = np.arange(len(ends)) - start                 # its place among its point's pairs, nearest first
        keep = (rank[:m] < K['max']) & (rank[m:] < K['max'])
        i, j, d = i[keep], j[keep], d[keep]
    near = 1 - d / dist if dist > 0 else np.zeros(0)
    return dict(i=i, j=j, near=near, alpha=K['opacity'].at(t) * near * np.minimum(dim[i], dim[j]), width=K['width'].at(t), rgb=K['color'])


def _faces(n, lk, opacity, link_opacity):
    """triangles whose three sides are all links; as faint as their faintest side"""
    A = np.zeros((n, n), np.float32)
    al = lk['alpha'] / max(link_opacity, 1e-9)
    A[lk['i'], lk['j']] = A[lk['j'], lk['i']] = np.maximum(al, 1e-6)
    tri, alpha = [], []
    for a, b, w in zip(lk['i'], lk['j'], al):                  # a < b; the third corner beyond b, so each face comes once
        c = np.nonzero((A[a, b + 1:] > 0) & (A[b, b + 1:] > 0))[0] + b + 1
        if len(c):
            tri.append(np.stack([np.full(len(c), a), np.full(len(c), b), c], -1))
            alpha.append(np.minimum(w, np.minimum(A[a, c], A[b, c])))
    if not tri:
        return None
    return dict(tri=np.concatenate(tri), alpha=opacity * np.concatenate(alpha))


# ---------------- drawing ----------------
TEX = 128                       # a sprite's texture, px; its picture sits inside a margin of PAD px (the quad's soft edge)
PAD = 4


@functools.lru_cache(16)
def _texture(kind, src=None, mtime=None):
    """a white sprite (premultiplied: colour = alpha) with its smaller copies → (skia image shader, w, h, how much the
    quad is bigger than the sprite's own size)"""
    import skia
    S = TEX
    if kind == 'image':
        im = Image.open(src).convert('RGBA')
        k = min(1.0, 256 / max(im.size))
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS) if k < 1 else im
        big = Image.new('RGBA', (im.width + 2 * PAD, im.height + 2 * PAD), (0, 0, 0, 0))
        big.paste(im, (PAD, PAD))
        a = np.asarray(big.convert('RGBa'))
        grow = (big.width / im.width, big.height / im.height)
    else:
        ss = 4                                                 # drawn 4× larger and averaged: exact soft edges
        yy, xx = np.mgrid[0:S * ss, 0:S * ss]
        c, rr = S * ss / 2, (S / 2 - PAD) * ss
        x, y = (xx + 0.5 - c) / rr, (yy + 0.5 - c) / rr
        r = np.hypot(x, y)
        if kind == 'glow':                                     # bright in the middle, fading smoothly to nothing at its edge
            q = np.clip(1 - r, 0, 1)
            al = (q * q * (3 - 2 * q)).astype(np.float32)
        else:                                                  # streak: a soft core — stretched along the flight it is a streak
            al = (np.exp(-4.5 * r * r) * np.clip((1 - r) * 4, 0, 1)).astype(np.float32)
        al = al.reshape(S, ss, S, ss).mean((1, 3))
        v = np.clip(np.round(al * 255), 0, 255).astype(np.uint8)
        a = np.dstack([v, v, v, v])
        grow = (S / (S - 2 * PAD),) * 2
    img = skia.Image.fromarray(np.ascontiguousarray(a), colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
    try:
        img = img.withDefaultMipmaps()
    except AttributeError:
        pass
    sh = img.makeShader(skia.TileMode.kClamp, skia.TileMode.kClamp, skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear))
    return sh, a.shape[1], a.shape[0], grow, img


def _sprites(canvas, D):
    """every particle drawn. Soft sprites (glow, streak, a picture) are textured quads, all in one call. Hard-edged ones
    (disc, ring, square, star, spark's rays) are drawn as shapes with exact edge coverage, one by one: as textures
    they twinkled — a 3 px disc covered from 5 to 8 px² depending on where between the pixels it sat, a ring or a
    star twice that; as shapes the disc holds its area within 6 % at 3 px and 1 % at 6"""
    import skia
    n = len(D['x'])
    if not n:
        return
    kind = D['sprite']
    hx, hy = np.broadcast_to(D['hx'], (n,)), np.broadcast_to(D['hy'], (n,))
    mode = skia.BlendMode.kPlus if D['add'] else skia.BlendMode.kSrcOver
    if kind in ('glow', 'streak', 'image', 'spark'):
        E = D if kind != 'spark' else dict(D, alpha=D['alpha'] * 0.55)     # a spark: a soft core under its rays
        _quads(canvas, E, 'glow' if kind == 'spark' else kind, hx * (0.5 if kind == 'spark' else 1.0), hy * (0.5 if kind == 'spark' else 1.0), mode)
        if kind != 'spark':
            return
    paint = skia.Paint(AntiAlias=True, BlendMode=mode)
    rgb = (np.clip(D['rgb'], 0, 255) / 255).tolist()
    al, X, Y, HX, HY, AN = D['alpha'].tolist(), D['x'].tolist(), D['y'].tolist(), hx.tolist(), hy.tolist(), np.degrees(D['ang']).tolist()
    if kind == 'disc':
        cap = skia.Paint(AntiAlias=True, BlendMode=mode, Style=skia.Paint.kStroke_Style, StrokeCap=skia.Paint.kRound_Cap)
        co, si = np.cos(D['ang']).tolist(), np.sin(D['ang']).tolist()
        for k in range(n):
            c4 = skia.Color4f(rgb[k][0], rgb[k][1], rgb[k][2], al[k])
            d = HX[k] - HY[k]
            if d > 0.05:                                       # smeared along its flight: a line with round ends
                cap.setColor4f(c4)
                cap.setStrokeWidth(2 * HY[k])
                canvas.drawLine(X[k] - co[k] * d, Y[k] - si[k] * d, X[k] + co[k] * d, Y[k] + si[k] * d, cap)
            else:
                paint.setColor4f(c4)
                canvas.drawCircle(X[k], Y[k], HY[k], paint)
        return
    if kind == 'ring':
        paint.setStyle(skia.Paint.kStroke_Style)
        for k in range(n):
            paint.setColor4f(skia.Color4f(rgb[k][0], rgb[k][1], rgb[k][2], al[k]))
            paint.setStrokeWidth(0.22 * HY[k])
            canvas.drawCircle(X[k], Y[k], 0.89 * HY[k], paint)
        return
    path = _outline(kind)
    for k in range(n):
        paint.setColor4f(skia.Color4f(rgb[k][0], rgb[k][1], rgb[k][2], al[k]))
        canvas.save()
        canvas.translate(X[k], Y[k])
        if AN[k]:
            canvas.rotate(AN[k])
        canvas.scale(HX[k], HY[k])
        canvas.drawPath(path, paint)
        canvas.restore()


@functools.lru_cache(8)
def _outline(kind):
    """a hard-edged sprite as a path in −1..1"""
    import skia
    p = skia.Path()
    if kind == 'square':
        p.addRect(skia.Rect.MakeLTRB(-1, -1, 1, 1))
        return p
    pts, inner = (10, 0.42) if kind == 'star' else (8, 0.1)    # star: five points; spark: four thin rays
    for q in range(pts):
        rad, a = (1.0 if q % 2 == 0 else inner), -math.pi / 2 + q * 2 * math.pi / pts
        (p.moveTo if q == 0 else p.lineTo)(rad * math.cos(a), rad * math.sin(a))
    p.close()
    return p


def _quads(canvas, D, kind, hx, hy, mode):
    """soft sprites: one textured quad each (turned, stretched, tinted), all in one call"""
    import skia
    n = len(D['x'])
    sh, tw, th, grow, _ = _texture(kind, D['src'], os.path.getmtime(D['src']) if D['src'] else None)
    hx, hy = hx * grow[0], hy * grow[1]
    if kind == 'image':                                        # `size` is the picture's width; its height follows
        hy = hy * (th - 2 * PAD) / max(1, tw - 2 * PAD)
    co, si = np.cos(D['ang']), np.sin(D['ang'])
    ux, uy, vx, vy = co * hx, si * hx, -si * hy, co * hy
    x, y = D['x'], D['y']
    quad = np.stack([np.stack([x - ux - vx, y - uy - vy], -1), np.stack([x + ux - vx, y + uy - vy], -1),
                     np.stack([x + ux + vx, y + uy + vy], -1), np.stack([x - ux + vx, y - uy + vy], -1)], 1)
    rgb = np.clip(np.round(D['rgb']), 0, 255).astype(np.uint32)
    col = (np.clip(np.round(D['alpha'] * 255), 0, 255).astype(np.uint32) << 24) | (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    paint = skia.Paint(Shader=sh, BlendMode=mode)
    corners = [skia.Point(0, 0), skia.Point(tw, 0), skia.Point(tw, th), skia.Point(0, th)]
    for a in range(0, n, 10000):
        b = min(n, a + 10000)
        pts = [skia.Point(px, py) for px, py in quad[a:b].reshape(-1, 2).tolist()]
        idx = (np.arange(b - a)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3])[None, :]).ravel().tolist()
        v = skia.Vertices(skia.Vertices.kTriangles_VertexMode, pts, corners * (b - a), np.repeat(col[a:b], 4).tolist(), idx)
        canvas.drawVertices(v, paint, skia.BlendMode.kModulate)


LEVELS = 32                     # the steps of a link's faintness (each step is one path)


def _lines(canvas, D, add):
    import skia
    lk = D['links']
    if lk is None or not len(lk['i']):
        return
    x, y = D.get('lx', D['x']), D.get('ly', D['y'])
    lev = np.clip(np.ceil(lk['alpha'] * LEVELS), 0, LEVELS).astype(np.int64)
    r, g, b = (int(round(v)) for v in lk['rgb'])
    for q in np.unique(lev):
        if q <= 0:
            continue
        m = lev == q
        path = skia.Path()
        for x0, y0, x1, y1 in zip(x[lk['i'][m]].tolist(), y[lk['i'][m]].tolist(), x[lk['j'][m]].tolist(), y[lk['j'][m]].tolist()):
            path.moveTo(x0, y0)
            path.lineTo(x1, y1)
        canvas.drawPath(path, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=float(lk['width']), StrokeCap=skia.Paint.kRound_Cap,
                                         Color4f=skia.Color4f(r / 255, g / 255, b / 255, float(q) / LEVELS),
                                         BlendMode=skia.BlendMode.kPlus if add else skia.BlendMode.kSrcOver))


def _triangles(canvas, D):
    import skia
    T = D['tris']
    if T is None or not len(T['tri']):
        return
    x, y = D.get('lx', D['x']), D.get('ly', D['y'])
    r, g, b = (int(round(v)) for v in T['rgb'])
    tri = T['tri']
    a8 = np.clip(np.round(T['alpha'] * 255), 0, 255).astype(np.uint32)
    keep = a8 > 0
    tri, a8 = tri[keep], a8[keep]
    if not len(tri):
        return
    col = (a8 << 24) | (r << 16) | (g << 8) | b
    pts = [skia.Point(px, py) for px, py in np.stack([x[tri.ravel()], y[tri.ravel()]], -1).tolist()]
    v = skia.Vertices(skia.Vertices.kTriangles_VertexMode, pts, None, np.repeat(col, 3).tolist())
    canvas.drawVertices(v, skia.Paint(Color=0xFFFFFFFF, AntiAlias=True), skia.BlendMode.kModulate)


def render(L, t, sc, motion_blur, fps, view, sims):
    """the layer at t drawn in its own plane at `sc` px per content px → (x0, y0, w, h of the canvas in content px,
    premultiplied PIL RGBa picture of it) or None when there is nothing to draw"""
    import skia
    w, h = L['psize']
    if L['type'] == 'emitter':
        sim = sims.get(id(L))
        if sim is None or sim.L is not L:
            sim = sims[id(L)] = Sim(L, fps)
        D = emitter_data(L, t, sim, (motion_blur['shutter'] / 360 / fps / 2) if motion_blur else 0.0)
    else:
        D = plexus_data(L, t)
    if D is None or not len(D['x']):
        return None
    ext = np.hypot(np.broadcast_to(D['hx'], D['x'].shape), np.broadcast_to(D['hy'], D['x'].shape)) * 1.1 + 2
    xs, ys = [D['x'] - ext, D['x'] + ext], [D['y'] - ext, D['y'] + ext]
    if 'lx' in D:
        xs += [D['lx'] - 3, D['lx'] + 3]
        ys += [D['ly'] - 3, D['ly'] + 3]
    margin = max(w, h)
    bx0, by0 = max(-margin, min(float(a.min()) for a in xs)), max(-margin, min(float(a.min()) for a in ys))
    bx1, by1 = min(w + margin, max(float(a.max()) for a in xs)), min(h + margin, max(float(a.max()) for a in ys))
    if view is not None:
        bx0, by0, bx1, by1 = max(bx0, view[0]), max(by0, view[1]), min(bx1, view[2]), min(by1, view[3])
    if bx1 <= bx0 or by1 <= by0:
        return None
    ox, oy = math.floor(bx0 * sc), math.floor(by0 * sc)
    W, H = int(math.ceil(bx1 * sc)) - ox, int(math.ceil(by1 * sc)) - oy
    if W <= 0 or H <= 0 or W * H > 40e6:
        return None
    surf = skia.Surface.MakeRaster(skia.ImageInfo.Make(W, H, skia.kRGBA_F16_ColorType, skia.kPremul_AlphaType))
    c = surf.getCanvas()
    c.clear(skia.Color4f(0, 0, 0, 0))
    c.translate(-ox, -oy)
    c.scale(sc, sc)
    _triangles(c, D)
    _lines(c, D, D['add'])
    _sprites(c, D)
    a = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_F32_ColorType, alphaType=skia.kPremul_AlphaType)
    if not a[..., 3].any():
        return None
    im = Image.fromarray(np.clip(np.round(a * 255), 0, 255).astype(np.uint8), 'RGBa')
    return ox / sc, oy / sc, W / sc, H / sc, im
