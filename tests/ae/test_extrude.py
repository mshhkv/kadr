"""Known answers for outlines given thickness (kadr/extrude.py; `shape: text`, `shape: extrude` of a 3D scene): the
outline's area, which contour is a hole in which, a closed surface of the right volume, normals that point out, and the
letter's silhouette in 3D against the same letter drawn flat. python tests/ae/test_extrude.py"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import skia
from kadr.doc import Film, DocError
from kadr import extrude, scene3d

bad = []


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def volume(v, ix):
    P, t = v[:, :3].astype(float), ix.reshape(-1, 3)
    return float(np.einsum('ij,ij->i', P[t[:, 0]], np.cross(P[t[:, 1]], P[t[:, 2]])).sum() / 6)


def open_edges(v, ix):
    """how many edges of the surface do not have exactly one neighbour going the other way (0: closed, no cracks)"""
    key = np.round(v[:, :3].astype(float) * 1e4).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    t = inv[ix.reshape(-1, 3)]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    fwd = {}
    for a, b in e.tolist():
        fwd[(a, b)] = fwd.get((a, b), 0) + 1
    return sum(1 for (a, b), n in fwd.items() if n != 1 or fwd.get((b, a), 0) != 1)


def area_of(path, tol):
    cs = extrude.flatten(path, tol)
    _, depth = extrude.nest(cs)
    return sum(abs(extrude._area(c)) * (1 if d % 2 == 0 else -1) for c, d in zip(cs, depth))


# --- the outline
circ = skia.Path()
circ.addCircle(0, 0, 10)
a = area_of(circ, .005)
ok('круг разбит на отрезки точно', abs(a - math.pi * 100) < .002 * math.pi * 100, f'площадь {a:.3f} (πr² = {math.pi * 100:.3f})')
ring = skia.Path()
ring.addCircle(0, 0, 10); ring.addCircle(0, 0, 6, skia.PathDirection.kCCW); ring.addCircle(0, 0, 2)
cs = extrude.flatten(ring, .005)
groups, depth = extrude.nest(cs)
ok('кто в ком дырка: кольцо и точка внутри него', sorted(depth) == [0, 1, 2] and sorted(len(h) for _, h in groups) == [0, 1],
   f'глубины контуров {sorted(depth)}; сплошных {len(groups)}, дырок у них {sorted(len(h) for _, h in groups)}')

# --- a closed surface of the right volume
for name, path, depth_, bevel in (('круг без фаски', circ, 5.0, 0.0), ('кольцо с точкой без фаски', ring, 4.0, 0.0), ('круг с фаской', circ, 6.0, 1.5), ('кольцо с фаской', ring, 4.0, .8)):
    v, ix = extrude.mesh(path, depth_, bevel, 4, tol=.01)
    vol, flat = volume(v, ix), area_of(path, .01) * depth_
    ln = np.linalg.norm(v[:, 3:6], axis=1)
    good = open_edges(v, ix) == 0 and abs(ln - 1).max() < 1e-4 and abs(float(v[:, 2].max() - v[:, 2].min()) - depth_) < 1e-4
    if bevel == 0:
        ok(f'{name}: замкнутая поверхность, объём = площадь × толщина', good and abs(vol - flat) < .002 * flat, f'объём {vol:.2f}, площадь × толщина {flat:.2f}; незакрытых рёбер {open_edges(v, ix)}')
    else:
        # a quarter-round edge takes away (1 − π/4) b² along the outline's length, on both faces
        cs_ = extrude.flatten(path, .01)
        length = sum(float(np.hypot(*(np.roll(c, -1, 0) - c).T).sum()) for c in cs_)
        want = flat - 2 * (1 - math.pi / 4) * bevel ** 2 * length
        ok(f'{name}: замкнута, край скруглён на свой объём', good and abs(vol - want) < .02 * flat and abs(float(np.abs(v[:, :2]).max()) - 10) < .02,
           f'объём {vol:.2f} (ждём около {want:.2f} из {flat:.2f}); незакрытых рёбер {open_edges(v, ix)}; силуэт не сжался: {float(np.abs(v[:, :2]).max()):.3f}')
v, ix = extrude.mesh(circ, 5.0, 1.0, 4, tol=.01)
P, N = v[:, :3], v[:, 3:6]
wall = np.abs(N[:, 2]) < 1e-6
rad = P[wall, :2] / np.linalg.norm(P[wall, :2], axis=1, keepdims=True)
face = np.abs(P[:, 2]) > 2.5 - 1e-6
ok('нормали: стенка — наружу, лицо — вперёд и назад, фаска — между', float((rad * N[wall, :2]).sum(1).min()) > .999 and np.allclose(np.abs(N[face, 2]), 1, atol=1e-5)
   and float((np.sign(P[face, 2]) * N[face, 2]).min()) > .999 and bool(((np.abs(N[:, 2]) > .05) & (np.abs(N[:, 2]) < .95)).any()),
   f'стенка: наименьшее совпадение с радиусом {float((rad * N[wall, :2]).sum(1).min()):.4f}')
for txt in ('KADR', 'Жё8', 'Bg%@'):
    p = extrude.text_outline(txt, 'Arial', 30)
    for bevel in (0.0, .5):
        v, ix = extrude.mesh(p, 6.0, bevel)
        flat = area_of(p, max(p.getBounds().width(), p.getBounds().height()) / 1200) * 6
        if bevel == 0:
            ok(f'текст «{txt}»: замкнутая поверхность нужного объёма', open_edges(v, ix) == 0 and abs(volume(v, ix) - flat) < .003 * flat, f'{len(ix) // 3} треугольников, объём {volume(v, ix):.1f} из {flat:.1f}')
        else:
            ok(f'текст «{txt}» с фаской: замкнутая поверхность', open_edges(v, ix) == 0 and .85 * flat < volume(v, ix) < flat, f'{len(ix) // 3} треугольников, объём {volume(v, ix):.1f} из {flat:.1f}; незакрытых рёбер {open_edges(v, ix)}')

# --- in a scene: the letter's silhouette is the letter
W = 400
scene = dict(camera=dict(position=[0, 0, 4000], look_at=[0, 0, 0], fov=2.8648), lights=[], look=dict(raw=True, glow=0),      # 200 units across the frame: 2 px a unit
             nodes=[dict(shape='text', text='R8', font='Arial', size=60, depth=4, color='#ffffff', unlit=True)])
f = Film(data={'film': {'size': [W, W], 'fps': 25, 'duration': 1, 'background': '#000000'}, 'layers': [dict(id='s', type='scene3d', scene=scene, samples=64)]})
from kadr.gpu import GpuRenderer
gray = GpuRenderer(f).frame_u8(0)[..., 0].astype(float) / 255
got = gray > .5
p = extrude.text_outline('R8', 'Arial', 60)
m = skia.Matrix()
m.setScale(2, -2)
m.postTranslate(W / 2, W / 2)
flat2 = skia.Path()
p.transform(m, flat2)
surf = skia.Surface(W, W)
surf.getCanvas().clear(skia.ColorBLACK)
surf.getCanvas().drawPath(flat2, skia.Paint(Color=skia.ColorWHITE, AntiAlias=True))
flat_gray = surf.makeImageSnapshot().toarray()[..., 0].astype(float) / 255
true = flat_gray > .5
iou = float((got & true).sum() / (got | true).sum())
yy, xx = np.mgrid[0:W, 0:W]
mid = lambda a: np.array([(a * xx).sum() / a.sum(), (a * yy).sum() / a.sum()])
off = float(np.hypot(*(mid(gray) - mid(flat_gray))))
ok('силуэт объёмной буквы — сама буква', iou > .96 and off < .3 and abs(gray.sum() / flat_gray.sum() - 1) < .025,
   f'совпадение площади с плоским рисунком {iou * 100:.2f} %, середина сдвинута на {off:.2f} px, площадь отличается на {abs(gray.sum() / flat_gray.sum() - 1) * 100:.2f} %')
for what, node in (('пустой текст назван', dict(shape='text', text='', size=20)), ('extrude без контура назван', dict(shape='extrude', height=20))):
    try:
        Film(data={'film': {'size': [64, 64], 'fps': 25, 'duration': 1}, 'layers': [dict(id='s', type='scene3d', scene=dict(nodes=[node]))]})
        ok(what, False, 'документ принят')
    except DocError as e:
        ok(what, True, str(e).splitlines()[0][:110])

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
