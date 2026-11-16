"""A 3D scene looked at from all sides and checked (kadr/scenecheck.py): the distances each body is measured by against
known answers; planted mistakes — a leg through a seat, a book through the wall, a thing swallowed, a mug in the air,
two sheets in one plane — are named, what is right is left alone; the six views show what they say they show.
python tests/ae/test_scenecheck.py"""
import math, os, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import yaml
from kadr.doc import Film, DocError
from kadr import scene3d, scenecheck as sk

bad = []


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


# ---- 1. distances
P = lambda *rows: np.array(rows, float)
f, kind = sk._distance(('box', [5, 3, 2], 0))
d = f(P([0, 0, 0], [5, 0, 0], [8, 0, 0], [8, 7, 0], [0, 2.5, 0]))
ok('коробка: в середине — глубина до ближней грани, снаружи — расстояние, у угла — по диагонали', np.allclose(d, [-2, 0, 3, 5, -.5]) and kind == 'solid', str(d.round(3)))
f, _ = sk._distance(('box', [5, 3, 2], 1))
ok('скруглённая коробка: по грани то же, на углу дальше на срезанное', abs(f(P([8, 0, 0]))[0] - 3) < 1e-9 and abs(f(P([5, 3, 2]))[0] - (math.sqrt(3) - 1)) < 1e-9)
f, _ = sk._distance(('ball', 4))
ok('шар', np.allclose(f(P([0, 0, 0], [0, 6, 0])), [-4, 2]))
f, _ = sk._distance(('torus', 10, 2))
ok('кольцо: в середине дыры — снаружи, в толще — внутри', np.allclose(f(P([0, 0, 0], [10, 0, 0], [10, 0, 3], [0, 13, 0])), [8, -2, 1, 1]))
f, kind = sk._distance(('lathe', [[0, -5], [3, -5], [3, 5], [0, 5]]))
d = f(P([0, 0, 0], [0, 0, 2.5], [0, 4.5, 0], [0, 7, 0], [6, 0, 0], [3, 0, 4]))
ok('цилиндр как тело вращения: на оси глубина до стенки, а не ноль', np.allclose(d, [-3, -.5, -.5, 2, 3, 2]) and kind == 'solid', str(d.round(3)))
f, kind = sk._distance(('shell', [[9, 14], [11, 8], [15, 0]]))
ok('незамкнутый профиль (абажур) — оболочка: внутри неё не бывает', kind == 'shell' and (f(P([0, 5, 0], [15, 0, 0], [0, 0, 0])) >= 0).all() and abs(f(P([15, 0, 0]))[0]) < 1e-9)
f, kind = sk._distance(('sheet', 10, 5))
ok('лист: расстояние до прямоугольника', kind == 'sheet' and np.allclose(f(P([0, 0, 3], [13, 0, 4], [0, 0, 0])), [3, 5, 0]))

# ---- 2. planted mistakes
D = tempfile.mkdtemp()


def scene(nodes, **more):
    p = os.path.join(D, f'{len(os.listdir(D))}.yaml')
    open(p, 'w').write(yaml.safe_dump(dict(camera=dict(position=[0, 30, 200], look_at=[0, 20, 0], fov=30), lights=[dict(type='sky', color='#ffffff', ground='#888888', intensity=1)], nodes=nodes, **more), sort_keys=False))
    return p


def found(nodes, **more):
    F, L = sk.open_scene(scene(nodes, **more))
    sc = scene3d.layer_scene(L)
    world, seen = sk.pose(sc, L, 0.0)
    return sk.audit(sc, world, seen)


floor = dict(id='floor', shape='box', size=[200, 4, 200], position=[0, -2, 0])
chair = dict(id='chair', children=[dict(id='seat', shape='rbox', size=[44, 6.5, 30], radius=3, position=[0, 20, 0]), dict(id='back', shape='rbox', size=[44, 58, 7], radius=3, position=[0, 40, -14]),
                                  dict(id='post', shape='cylinder', radius=3, height=17, position=[0, 8.5, 0])])
man = lambda shin_z: dict(id='man', position=[0, 23.25, 0], children=[dict(id='hips', shape='rbox', size=[22, 8, 12], radius=2, position=[0, 4, 0]),
                                                                    dict(id='shin', shape='rbox', size=[9, 18, 7], radius=1.5, position=[5, -5, shin_z])])
r = found([floor, chair, man(8)])
ok('голень сквозь сиденье названа — с тем, сколько её толщины ушло в него', len(r) == 1 and r[0][0] == 'through' and set(r[0][1]) == {'shin', 'seat'} and '%' in r[0][2], r[0][2] if r else 'ничего')
r = found([floor, chair, man(19)])
ok('голень перед краем сиденья — замечаний нет (человек сидит на сиденье, ножка кресла стоит на полу)', r == [], '; '.join(w for _, _, w in r))
r = found([floor, chair, man(8)], accept=[['man', 'chair']])
ok('пара, о которой сцена сама говорит «так задумано», не называется', r == [])
r = found([floor, dict(id='card', shape='box', size=[30, .6, 20], position=[0, .1, 0])])
ok('карточка, на волос утопленная в пол, — не «сквозь пол»', r == [], '; '.join(w for _, _, w in r))
r = found([floor, dict(id='plank', shape='box', size=[30, .6, 20], position=[0, 10, 0], rotation=[0, 0, 90])])
ok('а доска, стоящая сквозь пол, — да', len(r) == 1 and r[0][0] == 'through' and set(r[0][1]) == {'plank', 'floor'}, r[0][2] if r else 'ничего')
wall = dict(id='wall', shape='plane', size=[200, 120], position=[0, 60, -50])
r = found([floor, wall, dict(id='shelf', shape='box', size=[40, 2, 12], position=[0, 60, -44]), dict(id='book', shape='box', size=[3, 14, 10], position=[0, 68, -50])])
ok('книга, наполовину ушедшая за стену, названа; полка, прижатая к стене, — нет', [(k, n[0]) for k, n, _ in r] == [('pokes', 'book')], '; '.join(w for _, _, w in r))
r = found([floor, dict(id='crate', shape='box', size=[20, 20, 20], position=[0, 10, 0]), dict(id='coin', shape='cylinder', radius=3, height=1, position=[2, 8, 3])])
ok('тело целиком внутри другого названо спрятанным (и только так)', [(k, n) for k, n, _ in r] == [('hidden', ('coin', 'crate'))], '; '.join(w for _, _, w in r))
r = found([floor, dict(id='jar', shape='box', size=[20, 20, 20], position=[0, 10, 0], opacity=.4), dict(id='coin', shape='cylinder', radius=3, height=1, position=[2, .5, 3])])
ok('внутри прозрачного тела — видно, не названо', not [k for k, _, _ in r if k == 'hidden'], '; '.join(w for _, _, w in r))
mug = lambda y: dict(id='mug', position=[30, y, 0], children=[dict(id='cup', shape='lathe', profile=[[0, 0], [3.4, 0], [3.9, 8], [3.6, 7.2], [0, 7.2]]), dict(id='handle', shape='torus', radius=1.8, tube=.42, position=[4.1, 4, 0])])
r = found([floor, mug(5)])
ok('кружка над столом висит в воздухе — названа вещь целиком, с зазором', [(k, n) for k, n, _ in r] == [('floats', ('mug',))] and '5,0' in r[0][2], '; '.join(w for _, _, w in r))
r = found([floor, dict(id='boot', position=[0, 0, 0], children=[dict(id='leg', shape='box', size=[6, 20, 6], position=[0, 10, 0]), dict(id='shoe', shape='box', size=[6, 4, 10], position=[0, 2, 30])])])
ok('тело, оторванное от своей вещи, названо, хотя сама вещь стоит на полу', [(k, n) for k, n, _ in found([floor, dict(id='boot', children=[dict(id='leg', shape='box', size=[6, 20, 6], position=[0, 10, 0]), dict(id='shoe', shape='box', size=[6, 4, 10], position=[0, 12, 30])])])] == [('floats', ('shoe',))] and r == [], '; '.join(w for _, _, w in r))
ok('кружка на столе — нет', found([floor, mug(0)]) == [])
ok('тело, которому положено висеть, не называется', found([floor, mug(5)], accept=[['mug']]) == [])
r = found([wall, dict(id='poster', shape='plane', size=[30, 40], position=[10, 60, -50]), dict(id='photo', shape='plane', size=[20, 20], position=[-60, 60, -49.8])])
ok('два листа в одной плоскости один на другом — мерцание; лист чуть впереди стены — нет', [(k, set(n)) for k, n, _ in r] == [('flicker', {'wall', 'poster'})], '; '.join(w for _, _, w in r))
r = found([floor, dict(id='pile', children=[dict(id='a', shape='box', size=[10, 10, 10], position=[0, 5, 0]), dict(id='b', shape='sphere', radius=6, position=[0, 9, 0])])])
ok('тела одной вещи могут входить друг в друга', r == [], '; '.join(w for _, _, w in r))

# ---- 3. the six views
F, L = sk.open_scene(scene([dict(id='thing', children=[dict(id='body', shape='box', size=[20, 30, 10], position=[0, 15, 0], color='#808080'),
                                                       dict(id='nose', shape='box', size=[4, 4, 6], position=[0, 20, 8], color='#ff0000', unlit=True),
                                                       dict(id='cap', shape='box', size=[20, 2, 10], position=[0, 31, 0], color='#0000ff', unlit=True),
                                                       dict(id='ear', shape='box', size=[4, 6, 4], position=[12, 20, 0], color='#00ff00', unlit=True)]),
                            dict(id='far', shape='sphere', radius=5, position=[300, 5, 0], color='#ffff00', unlit=True)]))
sc = scene3d.layer_scene(L)
pic, (lo, hi) = sk.sheet(sc, L, 0.0, node='thing', tile=200)
a = np.asarray(pic).astype(int)
tile = lambda k: a[(k // 3) * 234 + 34:(k // 3) * 234 + 234, (k % 3) * 200:(k % 3) * 200 + 200]
red = lambda t: int(((t[..., 0] > 200) & (t[..., 1] < 60) & (t[..., 2] < 60)).sum())
blue = lambda t: int(((t[..., 2] > 200) & (t[..., 0] < 60) & (t[..., 1] < 60)).sum())
green = lambda t: ((t[..., 1] > 200) & (t[..., 0] < 60) & (t[..., 2] < 60))
yellow = lambda t: int(((t[..., 0] > 200) & (t[..., 1] > 200) & (t[..., 2] < 60)).sum())
gx = lambda t: float(np.nonzero(green(t))[1].mean()) if green(t).any() else -1
ok('спереди нос виден, сзади — нет', red(tile(0)) > 100 and red(tile(1)) == 0, f'красных точек {red(tile(0))} и {red(tile(1))}')
ok('правое ухо: спереди — справа, сзади — слева, справа — в середине кадра, слева его не видно', gx(tile(0)) > 120 and 0 < gx(tile(1)) < 80 and abs(gx(tile(3)) - 100) < 25 and gx(tile(2)) < 0,
   f'{gx(tile(0)):.0f}, {gx(tile(1)):.0f}, {gx(tile(3)):.0f}, {gx(tile(2)):.0f}')
ok('сверху видна макушка во всю ширину, спереди — только её край', blue(tile(4)) > 4 * blue(tile(0)) > 0, f'синих точек {blue(tile(4))} и {blue(tile(0))}')
ok('часть показана одна: постороннего тела нет ни на одном виде, рамка — её собственная', sum(yellow(tile(k)) for k in range(6)) == 0 and np.allclose(lo, [-10, 0, -5]) and np.allclose(hi, [14, 32, 11]), f'{lo} … {hi}')
h = np.nonzero((np.abs(tile(0) - np.array([58, 61, 69])).sum(-1) > 12).any(1))[0]
ok('вид занимает кадр: часть стоит от края до края с полем', 150 < h.max() - h.min() < 196, f'по высоте {h.max() - h.min()} из 200')
pic2, _ = sk.sheet(sc, L, 0.0, tile=200)
ok('вся сцена: далёкое тело в кадре', sum(yellow(np.asarray(pic2).astype(int)[(k // 3) * 234 + 34:(k // 3) * 234 + 234, (k % 3) * 200:(k % 3) * 200 + 200]) for k in range(6)) > 0)
try:
    sk.look(scene([floor]), node='flor')
    ok('опечатка в имени части названа', False)
except DocError as e:
    ok('опечатка в имени части названа', 'floor' in str(e), str(e).splitlines()[0][:80])

# ---- 4. a film's layer in the pose of a moment; kadr check
fp = os.path.join(D, 'film.yaml')
open(fp, 'w').write(yaml.safe_dump({'film': {'size': [200, 200], 'fps': 25, 'duration': 4}, 'layers': [dict(id='room', type='scene3d', src=scene([floor, mug(0)]), nodes={'mug': {'position': {0: [30, 0, 0], 2: [30, 20, 0]}}})]}))
_, w0 = sk.look(fp, at='0')
_, w2 = sk.look(fp, at='2')
ok('слой фильма проверяется в позе названного момента: на 0 с кружка стоит, на 2 с — висит', not any('висит' in x for x in w0) and any('mug' in x and 'висит' in x for x in w2), w2[-1])
from kadr import check
fb = os.path.join(D, 'film_bad.yaml')
open(fb, 'w').write(yaml.safe_dump({'film': {'size': [200, 200], 'fps': 25, 'duration': 1}, 'layers': [dict(id='room', type='scene3d', src=scene([floor, chair, man(8)]))]}))
got = [str(i) for i in check.run(Film(fb))]
ok('kadr check говорит то же о сцене фильма', any('shin' in g and 'сквозь' in g for g in got) and not [str(i) for i in check.run(Film(fp)) if 'сцена' in str(i)], got[0][:110] if got else 'ничего')

HERO = os.path.join(HERE, '..', '..', 'examples', 'hero3d', 'scene.yaml')
if os.path.exists(HERO):
    F, L = sk.open_scene(HERO)
    sc = scene3d.layer_scene(L)
    world, seen = sk.pose(sc, L, 0.0)
    ok('комната героя: замечаний нет (три пары она сама называет задуманными)', sk.audit(sc, world, seen) == [] and len(sc.J['accept']) == 3)
    sc.J['accept'] = []
    r = sk.audit(sc, world, seen)
    ok('без этих пар названы они и только они: полка и флаг в шторе, кружка в крышке ноутбука',
       {frozenset((sc.nodes[next(i for i, n in enumerate(sc.nodes) if n['name'] == x)]['name'] and next(b for b in ('shelfOnWall', 'flagOnWall', 'window', 'macbook', 'mug3') if x in sk.subtree_names(sc, b))) for x in n) for k, n, _ in r}
       == {frozenset(('shelfOnWall', 'window')), frozenset(('flagOnWall', 'window')), frozenset(('macbook', 'mug3'))} and all(k == 'through' for k, _, _ in r), f'{len(r)} замечаний')

print()
print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ СХОДИТСЯ: {len(bad)} — ' + '; '.join(bad))
sys.exit(1 if bad else 0)
