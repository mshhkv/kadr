"""Known answers for vector files (kadr/svg.py): the `svg` layer — colours where the file puts them, sharp when enlarged,
fitted into its box, in one colour; and a file's shape traced for 3D — a disc's outline, the logo's volume, a PNG's own
transparency. python tests/ae/test_svg.py"""
import math, os, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from PIL import Image, ImageDraw
from kadr.doc import Film, DocError
from kadr.render import Renderer, to_u8
from kadr.gpu import GpuRenderer
from kadr import svg, extrude

bad = []
D = tempfile.mkdtemp()
LOGO = os.path.join(HERE, 'logo.svg')


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def frame(layers, R=Renderer, size=(400, 240)):
    f = Film(data={'film': {'size': list(size), 'fps': 25, 'duration': 1, 'background': '#000000'}, 'layers': layers})
    r = R(f)
    return (to_u8(r.frame(0)) if R is Renderer else r.frame_u8(0)).astype(int)


# the file is 200 × 120: an orange-to-red square with a white triangle at the left, a blue ring with a dot at the right
a = frame([dict(id='s', type='svg', src=LOGO, position=[200, 120])])
px = lambda im, x, y: im[int(120 - 60 + y), int(200 - 100 + x)]            # a point of the file's own px
ok('цвета там, где их ставит файл', px(a, 60, 58)[0] > 240 and px(a, 60, 58)[2] > 240 and px(a, 14, 14)[1] > 180 and px(a, 14, 14)[2] < 60 and px(a, 108, 108)[1] < 90
   and px(a, 166, 60)[2] > 180 and px(a, 166, 60)[0] < 90 and px(a, 150, 60).max() < 12 and px(a, 124, 60).max() < 12,
   f'треугольник {px(a, 60, 58).tolist()}, верх градиента {px(a, 14, 14).tolist()}, низ {px(a, 108, 108).tolist()}, точка кольца {px(a, 166, 60).tolist()}, между кольцом и точкой {px(a, 150, 60).tolist()}')
b = frame([dict(id='s', type='svg', src=LOGO, position=[400, 240], scale=4)], size=(800, 480))
row = b[240 + int((30 - 60) * 4), :, 0].astype(float)                       # across the square's left edge, 4× larger
edge = np.nonzero((row > 25) & (row < 230))[0]
edge = edge[edge < 100]
ok('увеличенный вчетверо — резкий: рисуется заново, а не растягивается', len(edge) <= 3 and row[:100].min() < 25 and row[:100].max() > 230, f'край квадрата занимает {len(edge)} px (у растянутой картинки было бы около 6)')
c = frame([dict(id='s', type='svg', src=LOGO, position=[200, 120], size=[100, 200])])
ys, xs = np.nonzero(c.max(2) > 30)
ok('size — рамка, рисунок вписан с пропорциями', abs((xs.max() - xs.min() + 1) - 100) <= 3 and abs((ys.min() + ys.max()) / 2 - 120) <= 2 and (ys.max() - ys.min() + 1) < 62,
   f'рисунок {xs.max() - xs.min() + 1} × {ys.max() - ys.min() + 1} px посередине рамки 100 × 200')
d = frame([dict(id='s', type='svg', src=LOGO, position=[200, 120], color='#00ff00')])
on = d[d.max(2) > 60]
ok('color — рисунок одним цветом', len(on) > 5000 and on[:, 0].max() < 8 and on[:, 2].max() < 8 and px(d, 60, 58)[1] > 250, f'закрашено {len(on)} пикселей, все зелёные')
g = frame([dict(id='s', type='svg', src=LOGO, position=[200, 120], scale=1.7, rotation=12)], GpuRenderer)
e = frame([dict(id='s', type='svg', src=LOGO, position=[200, 120], scale=1.7, rotation=12)])
ok('слой одинаков на обоих рендерах', (np.abs(g - e) > 3).mean() < .003, f'расходится {(np.abs(g - e) > 3).mean() * 100:.2f} % пикселей')

# --- a shape traced out of a picture
yy, xx = np.mgrid[0:400, 0:400]
disc = np.clip(150.5 - np.hypot(xx - 200, yy - 200), 0, 1).astype(np.float32)
disc[np.hypot(xx - 200, yy - 200) < 60] = 0                                 # a ring: radius 150, a hole of 60
cs = svg.contours(disc)
areas = sorted(abs(extrude._area(c)) for c in cs)
ok('обводка: кольцо — две замкнутые линии нужной площади', len(cs) == 2 and abs(areas[1] - math.pi * 150 ** 2) < .004 * math.pi * 150 ** 2 and abs(areas[0] - math.pi * 60 ** 2) < .01 * math.pi * 60 ** 2,
   f'линий {len(cs)}; площади {areas[1]:.0f} и {areas[0]:.0f} (ждём {math.pi * 150 ** 2:.0f} и {math.pi * 60 ** 2:.0f})')
sp = svg.simplify(max(cs, key=len), .6)
ok('лишние точки убраны, форма цела', len(sp) < len(max(cs, key=len)) / 8 and abs(abs(extrude._area(sp)) - math.pi * 150 ** 2) < .004 * math.pi * 150 ** 2, f'{len(max(cs, key=len))} точек → {len(sp)}')
m = svg.mask(LOGO)
p = svg.outline(LOGO, 40)
v, ix = extrude.mesh(p, 10, 0)
P, t = v[:, :3].astype(float), ix.reshape(-1, 3)
vol = float(np.einsum('ij,ij->i', P[t[:, 0]], np.cross(P[t[:, 1]], P[t[:, 2]])).sum() / 6)
k = 40 / (112 / 120 * m.shape[0])
ok('форма логотипа из SVG — в объёме', abs(vol - float(m.sum()) * k * k * 10) < .003 * vol, f'объём {vol:.0f}, площадь рисунка × толщина {float(m.sum()) * k * k * 10:.0f}')
png = os.path.join(D, 'star.png')
im = Image.new('RGBA', (300, 300), (0, 0, 0, 0))
pts = [(150 + (130 if i % 2 == 0 else 55) * math.sin(i * math.pi / 5), 150 - (130 if i % 2 == 0 else 55) * math.cos(i * math.pi / 5)) for i in range(10)]
ImageDraw.Draw(im).polygon(pts, fill=(255, 200, 0, 255))
im.save(png)
true = abs(sum(pts[i][0] * pts[(i + 1) % 10][1] - pts[(i + 1) % 10][0] * pts[i][1] for i in range(10))) / 2
po = svg.outline(png, 100)
v2, ix2 = extrude.mesh(po, 5, 0)
P2, t2 = v2[:, :3].astype(float), ix2.reshape(-1, 3)
vol2 = float(np.einsum('ij,ij->i', P2[t2[:, 0]], np.cross(P2[t2[:, 1]], P2[t2[:, 2]])).sum() / 6)
h_px = max(p_[1] for p_ in pts) - min(p_[1] for p_ in pts)
ok('и из прозрачности PNG: звезда', abs(vol2 - true * (100 / h_px) ** 2 * 5) < .015 * vol2, f'объём {vol2:.0f} (по площади звезды {true * (100 / h_px) ** 2 * 5:.0f})')
f3 = frame([dict(id='s', type='scene3d', scene=dict(camera=dict(position=[0, 0, 4000], look_at=[0, 0, 0], fov=2.8648), lights=[], look=dict(raw=True, glow=0),
                                                    nodes=[dict(shape='extrude', src=png, height=100, depth=4, color='#ffffff', unlit=True)]), samples=16)], GpuRenderer, size=(400, 400))
got = f3[..., 0] > 127
want = np.asarray(Image.open(png).resize((int(300 * 200 / h_px), int(300 * 200 / h_px)), Image.LANCZOS))[..., 3] > 127
ys, xs = np.nonzero(got)
ok('выдавленная звезда в кадре — та же звезда', abs(got.sum() - want.sum()) < .02 * want.sum() and abs((ys.max() - ys.min() + 1) - 200) <= 3, f'площадь {got.sum()} px (у рисунка в том же масштабе {want.sum()}), высота {ys.max() - ys.min() + 1} px (ждём 200)')
for what, layer in (('нет файла — названо', dict(id='s', type='svg', src='nope.svg')), ('не SVG — названо', dict(id='s', type='svg', src=png))):
    try:
        frame([layer])
        ok(what, False, 'документ принят')
    except DocError as e_:
        ok(what, True, str(e_).splitlines()[0][:100])

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
