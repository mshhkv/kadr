"""Particle systems (kadr/psys.py) against known answers: the emitter's physics, that a frame is the same however the
simulation got to it, the sprites' coverage, the plexus's links. python tests/ae/test_psys.py"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film
from kadr import psys

bad = []


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def film(layer, dur=4, mb=False):
    L = dict(id='p', **layer)
    return Film(data={'film': {'size': [800, 600], 'fps': 25, 'duration': dur, 'background': '#000000', **({'motion_blur': True} if mb else {})},
                      'layers': [L]})


def sim_of(layer, dur=4):
    f = film(layer, dur)
    L = f.layers[0]
    return L, psys.Sim(L, f.fps)


# ---- one particle thrown up: where the school formula says
L, s = sim_of(dict(type='emitter', emit=dict(at=[400, 500], rate=0, bursts=[[0, 1]], direction=-90, spread=0, speed=300),
                   particle=dict(life=10, size=4), forces=dict(gravity=400)))
S, dt = s.at(1.0)
y = float(S['pos'][0, 1] + S['vel'][0, 1] * dt)
ok('бросок вверх: высота через 1 с как по формуле', abs(y - (500 - 300 * 1 + 0.5 * 400 * 1)) < 0.5, f'{y:.1f} при {500 - 300 + 200}')
S, dt = s.at(0.76)
ok('бросок вверх: у верхней точки скорость почти ноль', abs(float(S['vel'][0, 1]) - 4) < 0.5, f'{float(S["vel"][0, 1]):.2f} px/с')

# ---- how many live at once: rate × life
L, s = sim_of(dict(type='emitter', emit=dict(rate=200, speed=[10, 20]), particle=dict(life=1.5)))
S, _ = s.at(3.0)
ok('живых частиц rate × life', abs(len(S['idx']) - 300) <= 2, f'{len(S["idx"])} при 300')
S1, _ = s.at(3.4)
ok('номера идут по порядку рождения', bool((np.diff(S1['idx']) > 0).all()))

# ---- the same frame however the simulation got there
A = {k: v.copy() for k, v in s.at(2.0)[0].items()}             # going back: from a snapshot or from the start
s2 = psys.Sim(L, 25)
B = s2.at(2.0)[0]
ok('кадр один и тот же: назад по времени и с чистого листа', all(np.array_equal(A[k], B[k]) for k in A))
s3 = psys.Sim(L, 25)
for tt in (0.4, 1.2, 0.8, 2.0):
    C = s3.at(tt)[0]
ok('кадр один и тот же: вразнобой по времени', all(np.array_equal(A[k], C[k]) for k in A))

# ---- drag: a particle comes to fly with the wind
L, s = sim_of(dict(type='emitter', emit=dict(rate=0, bursts=[[0, 50]], speed=[0, 400], spread=360), particle=dict(life=10),
                   forces=dict(wind=[120, -40], drag=3)))
S, _ = s.at(3.0)
ok('сопротивление воздуха: скорость сходится к ветру', float(np.abs(S['vel'] - [120, -40]).max()) < 1, f'{float(np.abs(S["vel"] - [120, -40]).max()):.3f}')

# ---- the floor: nothing below it, a bounce loses height as `bounce` says
L, s = sim_of(dict(type='emitter', emit=dict(at=[400, 100], rate=0, bursts=[[0, 1]], speed=0), particle=dict(life=20, size=10),
                   forces=dict(gravity=1000), collide=dict(floor=500, bounce=0.6, friction=0)), dur=6)
ys = np.array([float(s.at(k / 50)[0]['pos'][0, 1]) for k in range(1, 290)])
ok('пол: частица не уходит под него', float(ys.max()) <= 495 + 1e-6, f'ниже всего {ys.max():.2f} (пол 500, радиус 5)')
first = int(np.argmax(ys >= 495 - 1e-6))
peak = float(ys[first:first + 60].min())
want = 495 - 0.36 * (495 - 100)                              # the height goes as bounce²
ok('пол: высота отскока как bounce²', abs(peak - want) < 12, f'{peak:.1f} при {want:.1f}')
ok('пол: в конце лежит и не дрожит', float(np.ptp(ys[-20:])) < 0.01, f'размах {float(np.ptp(ys[-20:])):.4f} px')

# ---- friction: on a slope steeper than the friction a ball slides down, on a gentler one it stays
for fr, slides in ((0.1, True), (0.6, False)):
    L, s = sim_of(dict(type='emitter', emit=dict(at=[200, 180], rate=0, bursts=[[0, 1]], speed=0), particle=dict(life=20, size=20),
                       forces=dict(gravity=800), collide=dict(segments=[[100, 200, 700, 400]], bounce=0, friction=fr)))   # slope 1 : 3
    x = float(s.at(3.0)[0]['pos'][0, 0])
    ok(f'трение {fr}: шар на склоне 1 : 3 ' + ('съезжает' if slides else 'лежит'), (x > 450) == slides and (slides or x < 215), f'за 3 с уехал на {x - 200:.0f} px')

# ---- a thin ramp is not flown through
L, s = sim_of(dict(type='emitter', emit=dict(at=[400, 50], rate=300, direction=90, spread=40, speed=[700, 900]), particle=dict(life=3, size=3),
                   collide=dict(segments=[[100, 300, 700, 300]], bounce=0.3)))
S, _ = s.at(2.0)
inside = (S['pos'][:, 0] > 100) & (S['pos'][:, 0] < 700)
ok('отрезок: быстрые мелкие частицы не пролетают насквозь', float(S['pos'][inside, 1].max()) <= 300, f'ниже всего {float(S["pos"][inside, 1].max()):.1f} при отрезке на 300')

# ---- kill: what touched is gone
L, s = sim_of(dict(type='emitter', emit=dict(at=[400, 100], rate=200, direction=90, spread=10, speed=400), particle=dict(life=5, size=4),
                   collide=dict(floor=400, kill=True)))
S, _ = s.at(3.0)
ok('kill: коснувшиеся пола исчезли', len(S['idx']) < 200 and float(S['pos'][:, 1].max()) < 400, f'живых {len(S["idx"])}, ниже всего {float(S["pos"][:, 1].max()):.0f}')

# ---- solid discs: a pile in a box, nobody inside anybody
L, s = sim_of(dict(type='emitter', emit=dict(shape='box', at=[400, 80], size=[300, 20], rate=30, direction=90, spread=20, speed=[50, 150]),
                   particle=dict(life=30, size=[18, 34]), forces=dict(gravity=900),
                   collide=dict(box=[200, 0, 600, 500], bounce=0.2, friction=0.3, each_other=True)), dur=6)
S, _ = s.at(5.0)
r = s.radius(S, 5.0)
d = np.hypot(S['pos'][:, None, 0] - S['pos'][None, :, 0], S['pos'][:, None, 1] - S['pos'][None, :, 1])
np.fill_diagonal(d, 1e9)
over = (r[:, None] + r[None, :] - d)
deep = float((over / np.minimum(r[:, None], r[None, :])).max())
ok('твёрдые шары: в куче никто не сидит в другом', deep < 0.25, f'{len(r)} шаров, глубже всего на {deep * 100:.0f} % радиуса')
ok('твёрдые шары: все в коробке', bool(((S['pos'][:, 0] - r > 199) & (S['pos'][:, 0] + r < 601) & (S['pos'][:, 1] + r < 501)).all()))
old = S['idx'] < 75
speed = np.hypot(S['vel'][old, 0], S['vel'][old, 1])
ok('твёрдые шары: старая куча улеглась', float(np.median(speed)) < 40, f'медианная скорость {float(np.median(speed)):.1f} px/с')

# ---- a point that pulls: the swarm gathers round it
L, s = sim_of(dict(type='emitter', emit=dict(shape='box', at=[400, 300], size=[700, 500], rate=0, bursts=[[0, 300]], speed=0), particle=dict(life=10),
                   forces=dict(points=[dict(at=[400, 300], strength=600)], drag=2)))
far0 = float(np.hypot(*(s.at(0.1)[0]['pos'] - [400, 300]).T).mean())
far1 = float(np.hypot(*(s.at(3.0)[0]['pos'] - [400, 300]).T).mean())
ok('притягивающая точка собирает рой', far1 < 0.3 * far0, f'среднее расстояние {far0:.0f} → {far1:.0f} px')

# ---- the emitter that moves: births fill its path evenly, not in clumps at the steps
L, s = sim_of(dict(type='emitter', emit=dict(at={0: [0, 300], 1: [800, 300], 'ease': 'linear'}, rate=400, speed=0), particle=dict(life=10)))
x = np.sort(s.at(1.0)[0]['pos'][:, 0])
gaps = np.diff(x)
ok('движущийся излучатель: частицы ложатся ровно, без сгустков', float(gaps.max()) < 2.5 * float(np.median(gaps)), f'шаг {float(np.median(gaps)):.2f} px, наибольший {float(gaps.max()):.2f}')

# ---- sprites: a small disc covers its area wherever it sits between pixels
import skia


def area(sprite, size, x, y, half_x=None):
    surf = skia.Surface.MakeRaster(skia.ImageInfo.Make(64, 64, skia.kRGBA_F16_ColorType, skia.kPremul_AlphaType))
    c = surf.getCanvas()
    c.clear(skia.Color4f(0, 0, 0, 0))
    psys._sprites(c, dict(x=np.array([x]), y=np.array([y]), hx=np.array([half_x or size / 2]), hy=np.array([size / 2]), ang=np.zeros(1),
                          rgb=np.array([[255.0, 255, 255]]), alpha=np.ones(1), sprite=sprite, src=None, add=False))
    return float(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_F32_ColorType, alphaType=skia.kPremul_AlphaType)[..., 3].sum())


for kind, size, tol, spread in (('disc', 3.0, 0.12, 0.08), ('disc', 6.0, 0.04, 0.02), ('disc', 20.0, 0.03, 0.02), ('ring', 8.0, None, 0.04),
                                ('star', 8.0, None, 0.04), ('spark', 24.0, None, 0.08), ('glow', 4.0, None, 0.04), ('square', 5.0, 0.01, 0.01)):
    got = np.array([area(kind, size, 30 + dx, 30 + dy) for dx in (0, 0.25, 0.5, 0.75) for dy in (0, 0.33, 0.66)])
    want = {'disc': math.pi * size * size / 4, 'square': size * size}.get(kind)
    ok(f'{kind} {size:g} px не мерцает: площадь не зависит от места между пикселями' + (' и равна своей' if want else ''),
       float(np.ptp(got)) / got.mean() < spread and (want is None or abs(got.mean() / want - 1) < tol),
       f'размах {float(np.ptp(got)) / got.mean() * 100:.1f} %' + (f', {got.mean():.2f} при {want:.2f}' if want else ''))
long = area('disc', 6.0, 30.3, 30.6, half_x=15.0)
ok('смазанный диск — полоса с круглыми концами своей площади', abs(long / (math.pi * 9 + 6 * 24) - 1) < 0.03, f'{long:.1f} при {math.pi * 9 + 6 * 24:.1f}')

# ---- motion blur: a fast particle is smeared along its flight and is as much fainter as it is longer
f = film(dict(type='emitter', emit=dict(at=[100, 300], rate=0, bursts=[[0, 1]], direction=0, spread=0, speed=1000), particle=dict(life=5, size=6, opacity_over_life=[1, 1])), mb=True)
L = f.layers[0]
D0 = psys.emitter_data(L, 0.5, psys.Sim(L, 25), 0.0)
D1 = psys.emitter_data(L, 0.5, psys.Sim(L, 25), f.motion_blur['shutter'] / 360 / 25 / 2)
ok('смаз: частица вытянута на путь за время затвора', abs(2 * float(D1['hx'][0]) - (6 + 1000 * 0.02)) < 0.01 and abs(float(D1['ang'][0])) < 1e-6, f'длина {2 * float(D1["hx"][0]):.1f} px при 26')
ok('смаз: во столько же раз бледнее', abs(float(D1['alpha'][0]) * 26 - float(D0['alpha'][0]) * 6) < 0.01, f'{float(D1["alpha"][0]):.3f} при {6 / 26:.3f}')

# ---- plexus
f = film(dict(type='plexus', points=dict(count=300, depth=300, drift=[10, 40]), links=dict(distance=90, max=3), triangles=0.1))
L = f.layers[0]
D = psys.plexus_data(L, 1.7)
deg = np.bincount(np.r_[D['links']['i'], D['links']['j']], minlength=300)
ok('plexus: у точки не больше max связей', int(deg.max()) <= 3, f'наибольшее {int(deg.max())}')
L['links']['max'] = 0
R, home, dirs = psys._homes(L['seed'], 300, None, 800.0, 600.0, 300.0, 0.0)
D = psys.plexus_data(L, 1.7)
P = np.random.default_rng(5).random((2500, 3)) * [800, 600, 300]
i, j, d = psys.near_pairs(P, 40)                               # many points: binned — against every pair looked at
d2 = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
bi, bj = np.nonzero(np.triu(d2 < 40, 1))
ok('plexus: пары по ячейкам те же, что перебором', len(i) == len(bi) and set(zip(i.tolist(), j.tolist())) == set(zip(bi.tolist(), bj.tolist())), f'{len(i)} пар')
ok('plexus: связь тем слабее, чем дальше точки', bool((np.diff(D['links']['alpha'][np.argsort(-D['links']['near'])]) <= 1e-9).all() or L['fade'] > 0))
tri = D['tris']['tri']
linked = set(zip(D['links']['i'].tolist(), D['links']['j'].tolist()))
ok('plexus: у каждой грани все три стороны — связи', all((a, b) in linked and (a, c) in linked and (b, c) in linked for a, b, c in tri.tolist()), f'{len(tri)} граней')
f = film(dict(type='plexus', points=dict(count=200, drift=[100, 300]), links=False), dur=20)
D = psys.plexus_data(f.layers[0], 17.3)
ok('plexus: точки отражаются от краёв и остаются в рамке', bool((D['x'] >= 0).all() and (D['x'] <= 800).all() and (D['y'] >= 0).all() and (D['y'] <= 600).all()))
f = film(dict(type='plexus', points=dict(grid=[9, 6]), triangles=0.2))
D = psys.plexus_data(f.layers[0], 0.5)
ok('сетка: связей и граней сколько положено', len(D['links']['i']) == 9 * 5 + 6 * 8 and len(D['tris']['tri']) == 2 * 8 * 5, f'{len(D["links"]["i"])} связей, {len(D["tris"]["tri"])} граней')
f = film(dict(type='plexus', points=dict(count=60, depth=200, drift=0), turn=[0, 360, 0]))
g = film(dict(type='plexus', points=dict(count=60, depth=200, drift=0)))
A, B = psys.plexus_data(f.layers[0], 1.0), psys.plexus_data(g.layers[0], 1.0)
ok('plexus: полный оборот возвращает облако на место', float(np.abs(np.sort(A['x']) - np.sort(B['x'])).max()) < 1e-6)

# ---- the document says what is wrong
from kadr.doc import DocError
try:
    film(dict(type='emitter', emit=dict(rate=10, speeed=5)))
    ok('опечатка в поле излучателя названа', False)
except DocError as e:
    ok('опечатка в поле излучателя названа', 'speeed' in str(e) and 'speed' in str(e), str(e).splitlines()[0][:90])

print('ВСЁ СХОДИТСЯ' if not bad else f'ЕСТЬ РАСХОЖДЕНИЯ: {bad}')
sys.exit(1 if bad else 0)
