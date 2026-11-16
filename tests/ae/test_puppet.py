"""Known answers for a picture bent by pins (kadr/puppet.py): pins moved together as a whole move the whole picture the
same way, exactly; pins that stay leave the picture as it was; a pin is where it is led; a small move of a pin is a
small move of everything; parts that do not touch do not pull each other; what the frame shows is where the pin went;
and the mistakes are named. python tests/ae/test_puppet.py"""
import math, os, sys, tempfile
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr.render import Renderer, to_u8
from kadr.gpu import GpuRenderer
from kadr import puppet as pp
from kadr import check as ck

bad = []
D = tempfile.mkdtemp()
MAN = os.path.join(HERE, 'man.png')


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def film(layers, size=(600, 600), dur=2):
    return Film(data={'film': {'size': list(size), 'fps': 25, 'duration': dur, 'background': '#000000'}, 'layers': layers})


def errs(layer):
    try:
        film([dict(dict(id='p', type='puppet', position=[300, 300]), **layer)])
        return ''
    except DocError as e:
        return str(e)


alpha = np.asarray(Image.open(MAN).convert('RGBA'), np.float32)[..., 3] / 255
PINS = [(356, 290), (64, 290), (210, 86), (250, 172), (170, 172), (175, 330), (245, 330), (140, 522), (280, 522)]   # the hand first; the body held at shoulders, hips and feet
P0 = pp.Puppet(alpha, 8, PINS)

# ---- pins moved together as a whole
worst = 0.0
for deg, sh in ((0, (30, -20)), (25, (0, 0)), (70, (60, -20)), (180, (5, 5)), (-140, (-80, 40))):
    th = math.radians(deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    move = lambda X: (X - [210, 300]) @ R.T + [210 + sh[0], 300 + sh[1]]
    worst = max(worst, float(np.abs(P0.solve(move(P0.at)) - move(P0.V)).max()))
ok('булавки, уведённые все вместе (сдвиг и поворот), уводят так же всю картинку', worst < 1e-6, f'пять переносов, поворот до 180°: ни одна из {len(P0.V)} вершин сетки не мимо больше чем на {worst:.1e} px')

# ---- a bend: the pins are where they are led, everything follows smoothly, nothing is torn off
to = np.array(PINS, float)
to[0] = (330, 60)                                              # the hand raised
P = P0.solve(to)
held = float(np.abs(P0.C @ P - to).max())
tri_area = lambda Q: np.abs((Q[P0.T[:, 1], 0] - Q[P0.T[:, 0], 0]) * (Q[P0.T[:, 2], 1] - Q[P0.T[:, 0], 1]) - (Q[P0.T[:, 1], 1] - Q[P0.T[:, 0], 1]) * (Q[P0.T[:, 2], 0] - Q[P0.T[:, 0], 0])).sum() / 2
far = np.hypot(*(P0.V - [150, 480]).T) < 40                     # the left leg, far from the hand
ok('булавка стоит там, куда её привели', held < .05, f'рука поднята на 230 px: все девять булавок на местах с точностью {held:.3f} px')
ok('картинка гнётся, а не растягивается; далёкое от булавки остаётся на месте', abs(tri_area(P) / tri_area(P0.V) - 1) < .05 and float(np.abs(P - P0.V)[far].max()) < 5 and float(np.abs(P - P0.V).max()) > 200,
   f'площадь всей сетки изменилась на {100 * (tri_area(P) / tri_area(P0.V) - 1):+.2f} %; левая нога сдвинулась не больше чем на {float(np.abs(P - P0.V)[far].max()):.2f} px')
steps, ratio = 240, 0.0
prev = None
for i in range(steps + 1):                                    # the hand led round the shoulder, through every pose on the way
    a = math.radians(48 - 150 * i / steps)
    to[0] = (250 + 158.6 * math.cos(a), 172 + 158.6 * math.sin(a))
    P = P0.solve(to)
    if prev is not None:
        ratio = max(ratio, float(np.abs(P - prev[0]).max()) / float(np.abs(to[0] - prev[1]).max()))
    prev = (P, to[0].copy())
ok('малый ход булавки — малый ход всей картинки (ни рывка, ни перескока)', ratio < 3, f'рука обведена вокруг плеча на 150° за {steps} шагов: за шаг ни одна вершина не ушла дальше чем на {ratio:.2f} хода булавки')

# ---- parts that do not touch
isl = np.zeros((120, 300), np.float32)
isl[30:90, 20:120] = 1
isl[30:90, 180:280] = 1
Q = pp.Puppet(isl, 10, [(40, 60), (100, 60)])                  # two pins in the left one, none in the right
to2 = np.array([(40, 60), (100, 20)], float)
P = Q.solve(to2)
right = Q.V[:, 0] > 150
ok('часть без булавки стоит, часть с булавками гнётся', float(np.abs(P - Q.V)[right].max()) == 0 and float(np.abs(P - Q.V)[~right].max()) > 20,
   f'правый остров сдвинут на {float(np.abs(P - Q.V)[right].max()):g} px, левый — до {float(np.abs(P - Q.V)[~right].max()):.0f} px')
Q = pp.Puppet(isl, 10, [(40, 60), (100, 60), (230, 60)])       # …and one pin in the right one
P = Q.solve(np.array([(40, 60), (100, 60), (245, 35)], float))
ok('часть с одной булавкой идёт за ней целиком, как есть', float(np.abs(P[right] - Q.V[right] - [15, -25]).max()) < 1e-9 and float(np.abs(P - Q.V)[~right].max()) < 1e-9,
   f'правый остров весь сдвинут на (15, −25) с точностью {float(np.abs(P[right] - Q.V[right] - [15, -25]).max()):.1e} px; левый не тронут')

# ---- in the frame
BAR = os.path.join(D, 'bar.png')
yy, xx = np.mgrid[0:60, 0:320]
bar = np.zeros((60, 320, 4), np.uint8)
bar[10:50, 10:310] = (255, 0, 0, 255)
bar[np.hypot(xx - 290, yy - 30) < 14] = (0, 255, 0, 255)       # a green disc where the pin will be
Image.fromarray(bar).save(BAR)
LAY = dict(id='p', type='puppet', src=BAR, position=[300, 300], mesh=10)
still = to_u8(Renderer(film([dict(id='i', type='image', src=BAR, position=[300, 300])])).frame(0)).astype(int)
same = to_u8(Renderer(film([dict(LAY, pins={'a': {'at': [30, 30]}, 'b': {'at': [290, 30], 'move': [0, 0]}})])).frame(0)).astype(int)
inside = np.zeros(still.shape[:2], bool)
inside[270 + 13:270 + 47, 140 + 13:140 + 270] = True                # the bar less 3 px of its edge and less the disc (a picture drawn in its own plane is laid by a soft cubic: its edge is a pixel softer)
ok('булавки на месте — картинка та же, что без них', np.abs(still - same)[inside].max() <= 2 and abs(same.sum() / still.sum() - 1) < .003 and np.abs(still - same)[~inside & (still.max(-1) < 9)].max() <= 60,
   f'слой puppet против слоя image: внутри разница до {np.abs(still - same)[inside].max()} из 255, всего краски {100 * (same.sum() / still.sum() - 1):+.2f} %')
# the end goes round the middle pin at the distance it has from it (140 px), 50° a second: the bar turns there, it is not pulled
f = film([dict(LAY, pins={'a': {'at': [30, 30]}, 'm': {'at': [150, 30]}, 'b': {'at': [290, 30], 'to': '=[150 + 140 * cos(radians(50 * t)), 30 + 140 * sin(radians(50 * t))]'}})])
R, G = Renderer(f), GpuRenderer(f)
a = to_u8(R.frame(1.0)).astype(int)
green = (a[..., 1] > 180) & (a[..., 0] < 90)
gy, gx = np.nonzero(green)
spot = lambda t: (140 + 150 + 140 * math.cos(math.radians(50 * t)), 270 + 30 + 140 * math.sin(math.radians(50 * t)))   # the layer's corner in the frame plus where the pin is led
want = spot(1.0)
ok('в кадре: зелёный круг там, куда привели его булавку (круг у самого сгиба чуть мнётся: до 3 px)', abs(gx.mean() - want[0]) < 3 and abs(gy.mean() - want[1]) < 3 and abs(green.sum() - math.pi * 14 ** 2) < .12 * math.pi * 14 ** 2,
   f'середина круга ({gx.mean():.1f}, {gy.mean():.1f}), булавка ({want[0]:.1f}, {want[1]:.1f}); площадь {int(green.sum())} px (круг радиуса 14 — {math.pi * 14 ** 2:.0f})')
a5 = to_u8(R.frame(.5)).astype(int)
g5 = np.nonzero((a5[..., 1] > 180) & (a5[..., 0] < 90))
ok('на полпути булавка на полпути', abs(g5[1].mean() - spot(.5)[0]) < 3 and abs(g5[0].mean() - spot(.5)[1]) < 3, f'середина круга ({g5[1].mean():.1f}, {g5[0].mean():.1f}), булавка ({spot(.5)[0]:.1f}, {spot(.5)[1]:.1f})')
d = np.abs(a - G.frame_u8(1.0).astype(int)).max()
ok('оба рендера рисуют согнутую картинку одинаково', d <= 3, f'наибольшая разница {d} из 255')
red = (a[..., 0] > 150).sum() + green.sum()
ok('согнутая полоса той же площади', abs(red / (300 * 40) - 1) < .04, f'{red} px против {300 * 40} у прямой')

# ---- what kadr check counts
f = film([dict(LAY, pins={'a': {'at': [30, 30]}, 'b': {'at': [290, 30], 'to': {0: [290, 30], 1.5: [560, 30]}}})])
w = [str(i) for i in ck.run(f)]
ok('kadr check: картинку растянули — сказано', any('растягивают' in x for x in w), (w or ['—'])[0][:120])
f = film([dict(id='p', type='puppet', src=MAN, position=[300, 300], pins={'a': {'at': [150, 200]}, 'b': {'at': [270, 200]}, 'c': {'at': [210, 320], 'to': {0: [210, 320], 1.5: [210, 60]}}})])
w = [str(i) for i in ck.run(f)]
ok('kadr check: булавку протащили сквозь картинку — складка названа', any('складкой' in x for x in w), ' | '.join(x[:70] for x in w) or '—')
# a free end swinging far and fast on a turning forearm moves many times further than its pins — smoothly: not a jerk
f = film([dict(LAY, pins={'a': {'at': [30, 30]}, 'm': {'at': [150, 30], 'move': '=[0, 14 * sin(5 * t)]'}, 'b': {'at': [200, 30], 'move': '=[0, -20 * sin(5 * t)]'}})])
w = [str(i) for i in ck.run(f)]
L_ = f.layers[0]
tip = max(float(np.abs(pp.puppet_of(L_).solve(pp.targets(L_, k / 25)) - pp.puppet_of(L_).solve(pp.targets(L_, (k + 1) / 25))).max()) for k in range(10, 30))
ok('kadr check: свободный конец, который быстро, но плавно машет, — не рывок', not any('дёргается' in x for x in w) and tip > 8, f'конец полосы проходит до {tip:.1f} px за кадр; замечаний о рывке нет')
demo = Film(os.path.join(HERE, '53_puppet.yaml'))
w = [str(i) for i in ck.run(demo)]
ok('kadr check: у человечка, машущего рукой, замечаний нет', not w, f'{demo.frames} кадров' + (': ' + w[0][:100] if w else ''))

# ---- a person's joints as pins
from kadr import matte as mt
nm = mt.name_joints({'nose': (50, 10, .9), 'shoulder_a': (70, 40, .8), 'shoulder_b': (30, 42, .8), 'elbow_b': (20, 70, .6), 'neck': (50, 38, .9)})
ok('суставы названы по месту на картинке: из пары _l — тот, что левее; одиночный — без стороны',
   nm.get('shoulder_l') == (30, 42, .8) and nm.get('shoulder_r') == (70, 40, .8) and nm.get('elbow') == (20, 70, .6) and 'nose' in nm and 'neck' in nm, ', '.join(sorted(nm)))
if sys.platform == 'darwin':
    ok('на рисованной фигуре человека нет — так и сказано, а не выдуманы суставы', pp.joints(MAN) == {}, f'найдено суставов: {len(pp.joints(MAN))}')

# ---- the mistakes named
cases = [('нет файла', dict(src='nope.png', pins={'a': {'at': [1, 1]}}), 'нет файла'), ('нет булавок', dict(src=MAN), 'хотя бы одна булавка'),
         ('булавка на пустом месте', dict(src=MAN, pins={'a': {'at': [20, 20]}}), 'на прозрачном месте'), ('булавка без места', dict(src=MAN, pins={'a': {'to': [1, 2]}}), 'at: [x, y]'),
         ('и to, и move', dict(src=MAN, pins={'a': {'at': [210, 200], 'to': [1, 2], 'move': [0, 0]}}), 'одно из двух'),
         ('опечатка', dict(src=MAN, pins={'a': {'at': [210, 200], 'mov': [0, 0]}}), 'может быть, move'), ('слишком мелкая сетка', dict(src=MAN, mesh=3, pins={'a': {'at': [210, 200]}}), 'слишком мелкая'),
         ('вырезки нет', dict(src=MAN, cutout=True, pins={'a': {'at': [210, 200]}}), 'kadr matte'), ('не то свойство', dict(src=MAN, pin={'a': {'at': [210, 200]}}), 'pins')]
miss = [n for n, lay, want in cases if want not in errs(lay)]
ok('ошибки названы по-человечески', not miss, f'{len(cases)} случаев' + (': не названы ' + ', '.join(miss) if miss else ''))

print('\nВСЁ СХОДИТСЯ' if not bad else f'\nНЕ СХОДИТСЯ: {bad}')
sys.exit(1 if bad else 0)
