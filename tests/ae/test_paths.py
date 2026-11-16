"""Known answers for curves as data (kadr/paths.py): areas after merging and offsetting, morphs, the repeater, text
outlines. python tests/ae/test_paths.py"""
import math, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import skia, yaml
from kadr.doc import Film
from kadr import paths
from kadr.render import text_layout

bad = []


def film(layer):
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'f.yaml')
    yaml.safe_dump({'film': {'size': [400, 400], 'fps': 25, 'duration': 2, 'background': '#000000'}, 'layers': [dict(id='s', type='shape', **layer)]},
                   open(p, 'w'), default_flow_style=None)
    f = Film(p)
    errs = [i for i in f.issues if i.level == 'error']
    assert not errs, '\n'.join(map(str, errs))
    return f.layers[0]


def area(p, size=420, pad=60):
    """the filled area of a path, px² (anti-aliased coverage summed)"""
    s = skia.Surface(size, size)
    c = s.getCanvas()
    c.clear(skia.Color4f(0, 0, 0, 0))
    c.translate(pad, pad)
    c.drawPath(p, skia.Paint(AntiAlias=True, Color=skia.ColorWHITE))
    return float(s.makeImageSnapshot().toarray()[..., 3].sum()) / 255


def check(what, got, want, tol):
    ok = abs(got - want) <= tol
    print(f'{"OK  " if ok else "FAIL"} {what}: {got:.1f}, нужно {want:.1f} (допуск {tol})')
    if not ok:
        bad.append(what)


SQ = 'M0 0 H100 V100 H0 Z'
# ---- Merge Paths: two 100 × 100 squares overlapping by half
for op, want in (('union', 15000), ('subtract', 5000), ('intersect', 5000), ('xor', 10000)):
    L = film({'paths': [{'d': SQ}, {'d': SQ, 'at': [50, 0], 'op': op}], 'fill': '#ffffff'})
    check(f'paths {op}: площадь двух квадратов внахлёст', area(paths.final(L, 0)), want, 30)
L = film({'paths': [{'shape': 'rect', 'size': [100, 100]}, {'shape': 'ellipse', 'size': [60, 60], 'at': [20, 20], 'op': 'subtract'}], 'fill': '#ffffff'})
check('paths: квадрат минус круг', area(paths.final(L, 0)), 10000 - math.pi * 900, 30)

# ---- Offset Paths: out by 10 (round corners), in by 10
L = film({'d': SQ, 'offset': 10, 'fill': '#ffffff'})
check('offset +10: квадрат вырос, углы круглые', area(paths.final(L, 0)), 10000 + 4 * 1000 + math.pi * 100, 40)
L = film({'d': SQ, 'offset': -10, 'fill': '#ffffff'})
check('offset −10: квадрат 80 × 80', area(paths.final(L, 0)), 6400, 30)
L = film({'d': SQ, 'offset': {0: 0, 2: 20}, 'fill': '#ffffff'})
print('OK  ' if paths.animated(L) else 'FAIL', 'анимированный offset считается движением')

# ---- a morph: at the keys the keys' own outlines; a rectangle on its way to a rectangle is a rectangle
L = film({'d': {0: 'M0 0 H10 V200 H0 Z', 2: 'M0 0 H300 V200 H0 Z', 'ease': 'linear'}, 'fill': '#ffffff'})
check('morph: первый ключ — его фигура', area(paths.final(L, 0.0)), 2000, 10)
check('morph: последний ключ — его фигура', area(paths.final(L, 2.0)), 60000, 20)
mid = paths.final(L, 1.0)
b = mid.computeTightBounds()
check('morph: на полпути прямоугольник 155 × 200 (вершина в вершину)', area(mid), 155 * 200, 40)
check('morph: его ширина', b.width(), 155, 0.6)
# different builds (a triangle and a circle): sampled; the area moves monotonically between the two
L = film({'d': {0: 'M100 0 L200 180 L0 180 Z', 2: 'M100 0 C210 0 210 200 100 200 C-10 200 -10 0 100 0 Z', 'ease': 'linear'}, 'fill': '#ffffff'})
a0, a1, a2 = (area(paths.final(L, t)) for t in (0.0, 1.0, 2.0))
ok = min(a0, a2) - 50 <= a1 <= max(a0, a2) + 50
print('OK  ' if ok else 'FAIL', f'morph разных фигур: площадь идёт от {a0:.0f} к {a2:.0f} через {a1:.0f}')
if not ok:
    bad.append('morph sampled')
# the same circle drawn from another start point: nothing must twist on the way
c1 = 'M100 0 C155 0 200 45 200 100 C200 155 155 200 100 200 C45 200 0 155 0 100 C0 45 45 0 100 0 Z'
c2 = 'M0 100 C0 45 45 0 100 0 C155 0 200 45 200 100 C200 155 155 200 100 200 C45 200 0 155 0 100 Z M300 300 L310 300 L310 310 Z'
L = film({'d': {0: c1, 2: c2, 'ease': 'linear'}, 'fill': '#ffffff'})
check('morph: круг в тот же круг с другой начальной точкой не схлопывается', area(paths.final(L, 1.0), 520), math.pi * 100 ** 2, 600)

# ---- Repeater
L = film({'shape': 'rect', 'size': [100, 40], 'fill': '#ffffff', 'repeat': {'count': 3, 'offset': [120, 0], 'opacity': [1, 0.2]}})
b = paths.final(L, 0).computeTightBounds()
check('repeat: три копии через 120 px — ширина', b.width(), 340, 0.5)
cs = paths.copies(L, 0)
ok = len(cs) == 3 and abs(cs[0][1] - 1) < 1e-6 and abs(cs[1][1] - 0.6) < 1e-6 and abs(cs[2][1] - 0.2) < 1e-6
print('OK  ' if ok else 'FAIL', f'repeat: прозрачность копий от первой к последней {[round(o, 2) for _, o in cs]}')
if not ok:
    bad.append('repeat opacity')
L = film({'shape': 'rect', 'size': [100, 40], 'fill': '#ffffff', 'repeat': {'count': 2.5, 'offset': [120, 0]}})
cs = paths.copies(L, 0)
ok = len(cs) == 3 and abs(cs[2][1] - 0.5) < 1e-6
print('OK  ' if ok else 'FAIL', 'repeat: дробное число копий — последняя проявляется наполовину')
if not ok:
    bad.append('repeat fraction')
L = film({'shape': 'rect', 'size': [40, 10], 'anchor': [0, 5], 'fill': '#ffffff', 'repeat': {'count': 4, 'rotation': 90, 'anchor': [0, 5]}})
b = paths.final(L, 0).computeTightBounds()
check('repeat: четыре копии по кругу — крест 80 × 80', b.width() + b.height(), 160, 1.0)

# ---- a text's outlines: as wide as the text layer lays it out; along a straight line it is the same text
L = film({'text': 'Kadr', 'font': 'Arial', 'font_size': 80, 'fill': '#ffffff'})
p = paths.final(L, 0)
_, (w, h) = text_layout('Arial', 80, 'Kadr', 'left', 1.2)
check('text: контуры по ширине как у текстового слоя', p.computeTightBounds().right(), w, 6)
check('text: площадь букв не пустая и не сплошная', area(p), 0.33 * w * 60, 0.2 * w * 60)
L2 = film({'text': 'Kadr', 'font': 'Arial', 'font_size': 80, 'fill': '#ffffff', 'along': 'M0 0 H600'})
b1, b2 = p.computeTightBounds(), paths.final(L2, 0).computeTightBounds()
check('along: по прямой линии текст тот же (ширина)', b2.width(), b1.width(), 0.5)
check('along: и высота', b2.height(), b1.height(), 0.5)
L3 = film({'text': 'Kadr', 'font': 'Arial', 'font_size': 80, 'fill': '#ffffff', 'along': 'M0 0 H600', 'along_offset': {0: 0, 2: 100, 'ease': 'linear'}})
check('along_offset: текст уехал по кривой на 50 px за половину пути', paths.final(L3, 1.0).computeTightBounds().left() - b2.left(), 50, 0.5)

# ---- a curve mask is a hidden matte shape
d = tempfile.mkdtemp()
yaml.safe_dump({'film': {'size': [200, 200], 'fps': 25, 'duration': 1, 'background': '#000000'},
                'layers': [{'id': 'a', 'type': 'solid', 'color': '#ffffff', 'size': [200, 200], 'position': [100, 100],
                            'mask': {'shape': 'rect', 'size': [100, 50], 'position': [50, 75]}}]}, open(os.path.join(d, 'm.yaml'), 'w'), default_flow_style=None)
from kadr.render import Renderer
f = Film(os.path.join(d, 'm.yaml'))
img = Renderer(f).frame(0.5)
check('mask-кривая: видно ровно её площадь', float(img[..., 0].sum()) / 255, 5000, 5)
ok = [L['id'] for L in f.layers] == ['a', 'a.mask'] and f.layers[0]['matte']['src'] == 'a.mask'
print('OK  ' if ok else 'FAIL', 'mask-кривая стала скрытым слоем-матом')
if not ok:
    bad.append('mask layer')

print('ВСЁ СХОДИТСЯ' if not bad else f'ЕСТЬ РАСХОЖДЕНИЯ: {bad}')
sys.exit(1 if bad else 0)
