"""A whole room written in Kadr's own scene language (kadr/scene3d.py): the bodies added for it — a lathe, a ring, a
disc, a cone's slice, a dome, a bent sheet, a rounded box with a thin radius, a glow — against known answers; a room of
lamps as the surroundings; a scene kept in a file of its own and led by a rig; and a character's room — a private scene of the owner's, skipped without it
(examples/hero3d/scene.yaml) against the scene its first renderer made (examples/hero3d/first).
python tests/ae/test_scene_words.py"""
import json, math, os, sys, tempfile, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr import scene3d, envmap
from kadr.gpu import GpuRenderer

bad = []


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def volume(m):
    v, ix = m
    t = v[ix.reshape(-1, 3), :3].astype(float)
    return float(np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)


def outward(m):
    """the share of faces whose corners' normals look the way the face does"""
    v, ix = m
    t = v[ix.reshape(-1, 3)].astype(float)
    fn = np.cross(t[:, 1, :3] - t[:, 0, :3], t[:, 2, :3] - t[:, 0, :3])
    keep = np.linalg.norm(fn, axis=1) > 1e-9
    return float((np.einsum('ij,ij->i', fn[keep], t[keep, :, 3:6].mean(1)) > 0).mean())


# ---- 1. the bodies
m = scene3d.lathe_mesh([[0, 0], [5, 0], [5, 10], [0, 10]], 96)
ok('тело вращения из прямоугольника — цилиндр', abs(volume(m) / (math.pi * 25 * 10) - 1) < .002 and outward(m) == 1, f'объём {volume(m):.1f} при {math.pi * 250:.1f}')
m = scene3d.lathe_mesh([[0, 0], [6, 0], [0, 9]], 96)
ok('из треугольника — конус', abs(volume(m) / (math.pi * 36 * 9 / 3) - 1) < .002, f'объём {volume(m):.1f} при {math.pi * 108:.1f}')
m = scene3d.torus_mesh(8, 2, 96, 48)
ok('кольцо: объём 2π²Rr², лежит в плоскости xy', abs(volume(m) / (2 * math.pi ** 2 * 8 * 4) - 1) < .004 and outward(m) == 1 and abs(m[0][:, 2].max() - 2) < 1e-5 and abs(m[0][:, 0].max() - 10) < 1e-5,
   f'объём {volume(m):.1f} при {2 * math.pi ** 2 * 32:.1f}')
v, ix = scene3d.disc_mesh(5, 96)
t = v[ix.reshape(-1, 3), :3].astype(float)
area = float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)
ok('диск: площадь πr², смотрит на +z', abs(area / (math.pi * 25) - 1) < .002 and (v[:, 5] == 1).all() and np.cross(t[0, 1] - t[0, 0], t[0, 2] - t[0, 0])[2] > 0, f'площадь {area:.2f} при {math.pi * 25:.2f}')
m = scene3d.cylinder_mesh((2, 6), 8, 96)
want = math.pi * 8 / 3 * (4 + 12 + 36)
side = m[0][:2 * 97]
ok('срез конуса: объём, стенка смотрит вверх на свой наклон', abs(volume(m) / want - 1) < .002 and outward(m) == 1 and abs(side[:, 4].mean() - (4 / 8) / math.hypot(1, 4 / 8)) < 1e-5,
   f'объём {volume(m):.1f} при {want:.1f}; наклон нормали {side[:, 4].mean():.3f}')
m = scene3d.cylinder_mesh(3, 8, 96)
ok('цилиндр остался цилиндром', abs(volume(m) / (math.pi * 72) - 1) < .002 and abs(m[0][:2 * 97, 4]).max() == 0)
v, ix = scene3d.sphere_mesh(4, 48, 24, .5)
ok('купол — верхняя половина шара', abs(v[:, 1].min()) < 1e-6 and abs(v[:, 1].max() - 4) < 1e-6 and abs(np.linalg.norm(v[:, :3], axis=1) - 4).max() < 1e-5)
v, ix = scene3d.plane_mesh(20, 10, (40, 20), 'sin(x * .5) * 2 + y * 0')
ok('лист гнётся по формуле: высота — та, что в формуле, нормаль наклонена по склону', abs(v[:, 2] - np.sin(v[:, 0] * .5) * 2).max() < 1e-5 and abs(v[:, 0].min() + 10) < 1e-6 and
   abs(v[20 * 41 // 2 + 20, 3] + math.cos(0) / math.hypot(1, 1)) < .02, f'в середине нормаль по x {v[20 * 41 // 2 + 20, 3]:.3f} при −0,707')
try:
    scene3d.plane_mesh(20, 10, (4, 4), 'open("x")')
    ok('в формуле листа нет ничего, кроме счёта', False)
except ValueError as e:
    ok('в формуле листа нет ничего, кроме счёта', 'relief' in str(e), str(e)[:70])
for size, r in (((10, 6, 4), 1), ((230, 72, 208), 2), ((900, 3, 2), .6)):
    m = scene3d.box_mesh(*size, radius=r)
    w, h, d = size
    exact = w * h * d - (4 - math.pi) * r * r * (w + h + d - 6 * r) - (8 - 4 / 3 * math.pi) * r ** 3
    cut = w * h * d - exact                                # what the rounding takes off the sharp box
    v = m[0]
    flat = v[np.abs(v[:, 4] - 1) < 1e-9]                   # the top face's flat middle
    ok(f'скруглённая коробка {size}, радиус {r}: срезано столько, сколько срезает радиус; грань плоская до самого скругления',
       abs((w * h * d - volume(m)) / cut - 1) < .03 and outward(m) == 1 and abs(np.abs(flat[:, 0]).max() - (w / 2 - r)) < 1e-4 and np.allclose(v[:, :3].max(0), [w / 2, h / 2, d / 2], atol=1e-4),
       f'срезано {w * h * d - volume(m):.3f} при {cut:.3f}')
dot = scene3d.soft_dot()
ok('пятно свечения: плотное в середине, пустое у края', dot[64, 64, 3] > 250 and dot[0, 0, 3] == 0 and 150 < dot[64, 64 + 35, 3] < 230)

# ---- 2. the same bodies in a document
def shot(scene, w=200, h=200, t=0.0, **layer):
    scene.setdefault('look', {}).update(raw=True, glow=0)
    f = Film(data={'film': {'size': [w, h], 'fps': 25, 'duration': 5, 'background': '#000000'}, 'layers': [dict(id='s', type='scene3d', scene=scene, samples=1, **layer)]})
    L = f.layers[0]
    sc = scene3d.layer_scene(L)
    G, insts, mats, draws, lk = sc.frame(t, w, h, 25, samples=1, flags={'still': True}, over=L.get('over'), lit=L.get('lit'), mts=L.get('mts'))
    a = sc.gpu().render(w, h, np.ascontiguousarray(G), np.ascontiguousarray(insts), np.ascontiguousarray(mats), np.ascontiguousarray(draws), lk, [0.0, 0.0, 0.0, 0.0]).download()
    return a[..., :3] / 255


cam = dict(position=[0, 0, 100], look_at=[0, 0, 0], fov=30)
sky = [dict(type='sky', color='#ffffff', ground='#ffffff', intensity=1)]
a = shot(dict(camera=cam, lights=sky, nodes=[dict(shape='torus', radius=15, tube=4, color='#ffffff', roughness=1)]))
ok('кольцо в кадре: в середине дыра', a[100, 100].max() < .01 and a[100, 100 + 56].min() > .2, f'середина {a[100, 100].max():.2f}, обод {a[100, 100 + 56].min():.2f}')
a = shot(dict(camera=cam, lights=sky, nodes=[dict(shape='disc', radius=20, color='#ffffff', roughness=1)]))
ok('диск в кадре — круг своего радиуса', abs((a[..., 0] > .1).sum() / (math.pi * (20 / (100 * math.tan(math.radians(15))) * 100) ** 2) - 1) < .03)
b0 = shot(dict(camera=cam, lights=[], nodes=[dict(shape='plane', size=[60, 60], position=[0, 0, -5], color='#202020', unlit=True)]))
b1 = shot(dict(camera=cam, lights=[], nodes=[dict(shape='plane', size=[60, 60], position=[0, 0, -5], color='#202020', unlit=True), dict(id='g', shape='glow', size=30, color='#ff8000', glow=.5)]))
ok('свечение прибавляется к тому, что за ним, и сходит к краю', b1[100, 100, 0] - b0[100, 100, 0] > .4 and b1[100, 100, 2] - b0[100, 100, 2] < .01 and abs(b1[100, 100 + 54, 0] - b0[100, 100 + 54, 0]) < .03 and
   b1[100, 100 + 20, 0] < b1[100, 100, 0], f'в середине +{b1[100, 100, 0] - b0[100, 100, 0]:.2f} красного, у края +{b1[100, 100 + 54, 0] - b0[100, 100 + 54, 0]:.3f}')
b2 = shot(dict(camera=cam, lights=[], nodes=[dict(shape='plane', size=[60, 60], position=[0, 0, -5], color='#202020', unlit=True), dict(id='g', shape='glow', size=30, color='#ff8000', glow=.5)]),
          materials={'g': {'shade': .5}})
ok('свечение приглушается из документа по имени', abs((b2[100, 100, 0] - b0[100, 100, 0]) / (b1[100, 100, 0] - b0[100, 100, 0]) - .5) < .02)
room = dict(room='#000000', lamps=[dict(color='#ffffff', intensity=5, position=[10, 0, 0], radius=3)])
a = shot(dict(camera=cam, lights=[], environment=room, nodes=[dict(shape='sphere', radius=20, color='#ffffff', metal=1, roughness=0)]))
xs = np.nonzero(a[100, :, 0] > .9)[0]
ok('окружение из ламп: лампа справа отражается в правой половине зеркального шара', len(xs) > 0 and xs.min() > 100 and abs(xs.mean() - 153) < 6 and a[100, 60].max() < .02, f'блик на {xs.min() if len(xs) else "—"}…{xs.max() if len(xs) else "—"} из 200')
eq = envmap.room((0, 0, 0), [((1, 1, 1), (0, 0, -10), 2)], 512)
want = 2 * math.pi * (1 - math.cos(math.asin(.2))) / (4 * math.pi)
got = float((eq[..., 0] * np.cos((np.arange(256) + .5 - 128) / 256 * math.pi)[:, None]).sum() / (512 * 256 * 2 / math.pi))
ok('шар-лампа занимает на небе свой телесный угол, стоит прямо впереди', abs(got / want - 1) < .02 and eq[128, 256, 0] == 1 and eq[128, 0, 0] == 0, f'доля неба {got:.5f} при {want:.5f}')
eq2 = envmap.room((0, 0, 0), [((1, 1, 1), (0, 0, -10), 2)], 512, soft=2)
ok('мягкий край не меняет света лампы', abs(float(eq2.sum() / eq.sum()) - 1) < .03 and 0 < eq2[128, 256 + 16, 0] < 1 and eq2[128, 256, 0] == 1, f'света {float(eq2.sum() / eq.sum()):.4f} от прежнего, на краю {eq2[128, 256 + 16, 0]:.2f}')
eq = envmap.room((0, 0, 0), [((1, 0, 0), (0, 0, -10), 2), ((0, 1, 0), (0, 0, -5), .5), ((0, 0, 1), (0, 1, 0), 3)], 512)
ok('ближняя лампа закрывает дальнюю; лампа, внутри которой стоит взгляд, не видна', tuple(eq[128, 256]) == (0, 1, 0) and tuple(eq[128, 256 + 12]) == (1, 0, 0) and eq[..., 2].max() == 0)
a = shot(dict(camera=cam, lights=[], environment=dict(room='#808080', intensity=1), nodes=[dict(shape='sphere', radius=20, color='#ffffff', roughness=1, env=0)]))
b = shot(dict(camera=cam, lights=[], environment=dict(room='#808080', intensity=1), nodes=[dict(shape='sphere', radius=20, color='#ffffff', roughness=1, env=1)]))
ok('env: 0 — тело не берёт света окружения', a[100, 100, 0] < .01 and abs(b[100, 100, 0] - .2158) < .03, f'{a[100, 100, 0]:.2f} без окружения, {b[100, 100, 0]:.2f} с ним')
for what, node, word in (('опечатка в форме названа', dict(shape='lahte'), 'lathe'), ('тело вращения без профиля названо', dict(shape='lathe'), 'profile'), ('неверная формула листа названа', dict(shape='plane', size=[4, 4], relief='x +'), 'relief')):
    try:
        Film(data={'film': {'size': [64, 64], 'fps': 25, 'duration': 1}, 'layers': [dict(id='s', type='scene3d', scene=dict(nodes=[node]))]})
        ok(what, False)
    except DocError as e:
        ok(what, word in str(e), str(e).splitlines()[0][:90])

# ---- 3. a scene in a file of its own, led from the film
D = tempfile.mkdtemp()
from PIL import Image
pic = np.zeros((8, 8, 4), np.uint8); pic[..., 0] = 255; pic[:, :4, 3] = 255          # red, its right half see-through
os.makedirs(os.path.join(D, 'room', 'pics'))
Image.fromarray(pic).save(os.path.join(D, 'room', 'pics', 'half.png'))
open(os.path.join(D, 'room', 'scene.yaml'), 'w').write('''camera: {position: [0, 0, 100], look_at: [0, 0, 0], fov: 30}
look: {raw: true, glow: 0}
lights:
  - {id: fill, type: sky, color: "#ffffff", ground: "#ffffff", intensity: 1}
nodes:
  - {id: back, shape: plane, size: [80, 80], position: [0, 0, -5], color: "#0000ff", unlit: true}
  - {id: sign, shape: plane, size: [20, 20], map: pics/half.png, alpha: true, unlit: true}
  - {id: ball, shape: sphere, radius: 4, position: [0, 18, 0], color: "#ffffff", roughness: 1}
''')
import yaml as _yaml
open(os.path.join(D, 'film.yaml'), 'w').write(_yaml.safe_dump({'film': {'size': [200, 200], 'fps': 25, 'duration': 2, 'background': '#000000'}, 'layers': [dict(id='s', type='scene3d', src='room/scene.yaml', samples=1)]}))
f = Film(os.path.join(D, 'film.yaml'))
a = GpuRenderer(f).frame_u8(0).astype(int)
ok('сцена из своего файла: картинка ищется рядом с ней, её прозрачная половина пропускает то, что сзади',
   a[100, 85, 0] > 200 and a[100, 85, 2] < 30 and a[100, 115, 2] > 200 and a[100, 115, 0] < 30, f'левая половина знака {a[100, 85, :3]}, правая {a[100, 115, :3]}')
ok('фильм помнит файл сцены (сервер перечитает фильм, когда тот изменится)', any(p.endswith('room/scene.yaml') for p in getattr(f, 'sources', ())))
base = {'film': {'size': [200, 200], 'fps': 25, 'duration': 2, 'background': '#000000'}}
open(os.path.join(D, 'film2.yaml'), 'w').write(_yaml.safe_dump({**base, 'layers': [dict(id='s', type='scene3d', src='room/scene.yaml', samples=1, nodes={'ball': {'position': {0: [0, 18, 0], 1: [20, 18, 0]}}})]}))
open(os.path.join(D, 'film3.yaml'), 'w').write(_yaml.safe_dump({**base, 'layers': [dict(id='s', type='scene3d', src='room/scene.yaml', samples=1, lights={'fill': {'intensity': 0}})]}))
b1 = GpuRenderer(Film(os.path.join(D, 'film2.yaml'))).frame_u8(1).astype(int)
b0 = GpuRenderer(Film(os.path.join(D, 'film3.yaml'))).frame_u8(0).astype(int)
row = 100 - 67                                             # the ball's height in the frame (18 cm up, 26.8 cm to the frame's edge)
ok('её названные тела ведутся из фильма: шар уехал вправо', a[row, 100, 1] > 60 and b1[row, 100, 1] < 10 and b1[row, 175, 1] > 60, f'на прежнем месте {a[row, 100, :3]} → {b1[row, 100, :3]}, на новом {b1[row, 175, :3]}')
ok('и её лампы: свет погашен по имени', b0[row, 100, 1] < 10 and b0[100, 85, 0] > 200, f'шар без света {b0[row, 100, :3]}')
try:
    Film(data={'film': {'size': [64, 64], 'fps': 25, 'duration': 1}, 'layers': [dict(id='s', type='scene3d', src=os.path.join(D, 'room', 'scene.yaml'), lights={'fil': {'intensity': 0}})]})
    ok('опечатка в имени лампы своей сцены названа', False)
except DocError as e:
    ok('опечатка в имени лампы своей сцены названа', 'fill' in str(e), str(e).splitlines()[0][:90])

# ---- 4. a character's room (private; skipped without it) against the scene its first renderer made
HERO = os.path.join(HERE, '..', '..', 'examples', 'hero3d')
FIRST = os.path.join(HERO, 'first')
if os.path.exists(os.path.join(HERO, 'scene.yaml')) and os.path.exists(os.path.join(FIRST, 'scene.json')):
    ex = scene3d.Scene(os.path.join(FIRST, 'scene.json'))
    F = Film(data={'film': {'size': [270, 480], 'fps': 30, 'duration': 1}, 'layers': [dict(id='s', type='scene3d', src=os.path.join(HERO, 'scene.yaml'))]})
    L = F.layers[0]
    mk = scene3d.layer_scene(L)

    def boxes(sc, P, Q, S):
        W, res = {}, []
        for i, n in enumerate(sc.nodes):
            m = scene3d.trs(P[i], Q[i], S[i])
            W[i] = m if n['parent'] < 0 else W[n['parent']] @ m
            if n['kind'] == 'Mesh':
                first, cnt, base = sc.geos[n['geo']]
                v = sc.verts[base + np.unique(sc.indices[first:first + cnt]), :3]
                w = (W[i] @ np.concatenate([v, np.ones((len(v), 1))], 1).T).T[:, :3]
                res.append((n.get('name', ''), w.min(0), w.max(0), sc.J['mats'][n['mat']]))
        return res
    Pm, Qm, Sm = mk.p0.copy(), mk.q0.copy(), mk.s0.copy()
    for nid, key, pr in L['made']['props']:
        v = pr.at(0.0)
        if key == 'position':
            Pm[nid] = v
        elif key == 'rotation':
            Qm[nid] = scene3d.euler(v)
        elif key == 'scale':
            Sm[nid] = v if np.ndim(v) else (v, v, v)
    byname = {n['name']: i for i, n in enumerate(mk.nodes)}
    # the hero is a block figure of the kit now (kadr/figure.py): his parts are called hero.torso, hero.head, …; a few
    # have new names, and he is placed by where he sits, not by a point under the floor
    KIT = {'root': 'hero', 'hoodie': 'hero.top', 'dollarL': 'hero.signL', 'dollarR': 'hero.signR'}
    for i, n in enumerate(ex.nodes):                           # the pose the first scene was stored in, onto the named parts
        nm = n.get('name')
        nm = nm if nm in byname else KIT.get(nm, f'hero.{nm}')
        if nm in byname and nm != 'hero.body':
            j = byname[nm]
            Pm[j], Qm[j], Sm[j] = ex.p0[i] + (np.array([0, .72 * 11, 0]) if nm == 'hero' else 0), ex.q0[i], ex.s0[i]
    E, Mk = boxes(ex, ex.p0, ex.q0, ex.s0), boxes(mk, Pm, Qm, Sm)
    used, off, paint = set(), [], []
    for name, lo, hi, mat in E:
        d, j = min((max(np.abs(lo - lo2).max(), np.abs(hi - hi2).max()), j) for j, (nm, lo2, hi2, m2) in enumerate(Mk) if j not in used)
        used.add(j)
        m2 = Mk[j][3]
        if d > .03:
            off.append((Mk[j][0], round(float(d), 3)))
        if mat['kind'] != m2['kind'] or (mat['kind'] != 'basic' and max(np.abs(np.array(mat['color']) - np.array(m2['color'])).max(), abs(mat['roughness'] - m2['roughness']), abs(mat['metalness'] - m2['metalness']),
                                                                       abs(mat['clearcoat'] - m2['clearcoat'])) > .004):
            paint.append(Mk[j][0])
    new = sorted(Mk[j][0] for j in range(len(Mk)) if j not in used)
    ok('комната героя словами Kadr: каждое из 127 тел первой сцены стоит там же (до 0,03 см)', len(E) == 127 and not off, f'расходятся: {off}' if off else '')
    ok('и покрашено так же', not paint, f'не так: {paint}' if paint else '')
    ok('сверх первой сцены — только ноги и сиденье', new == ['hero.pantL', 'hero.pantR', 'hero.shoeL', 'hero.shoeR', 'seat'], ', '.join(new))
    rig = _yaml.safe_load(open(os.path.join(HERO, 'rig.yaml')))
    lost = [k for part, have in (('nodes', byname), ('lights', {l.get('name') for l in mk.J['lights']}), ('materials', {m_.get('name') for m_ in mk.J['mats']})) for k in rig.get(part) or {} if k not in have]
    ok('все «нитки» рига находят в ней свою часть, лампу и материал', not lost, f'нет: {lost}' if lost else f"{len(rig['nodes'])} частей, {len(rig['lights'])} ламп, {len(rig['materials'])} материала")
    legs = byname['hero.legs']
    ok('ноги висят на теле (прыгают с героем, не наклоняются с туловищем), сиденье — часть кресла',
       mk.nodes[legs]['parent'] == byname['hero.body'] and mk.nodes[byname['hero.torso']]['parent'] == byname['hero.body'] and mk.nodes[byname['seat']]['parent'] == byname['chair'])
    ok('герой в сцене — одна запись набора «блочная фигурка», без единого тела, написанного руками',
       len(L['made']['figures']) == 1 and sum(1 for ln in open(os.path.join(HERO, 'scene.yaml'), encoding='utf-8') if 'hero.' in ln and 'id:' in ln) == 0,
       f"частей у фигурки {sum(1 for n in mk.nodes if n['name'].startswith('hero.'))}, «ниток» {sum(1 for n_, k_, p_ in L['over'] if n_.startswith('hero.'))}")
    # the same poses on both scenes: a frame from each kind of shot
    sys.path.insert(0, HERO)
    import hero as kit
    P = json.load(open(os.path.join(HERO, 'gta6_pose.json')))
    quiet = os.path.join(D, 'quiet.wav')
    with wave.open(quiet, 'wb') as w_:
        w_.setnchannels(1); w_.setsampwidth(2); w_.setframerate(8000); w_.writeframes(b'\0\0' * 8000)

    def film(src, vars=None, rest=None, **more):
        Lh = kit.layer(P['shots'], P['total'], [270, 480], **more)
        Lh['src'] = src
        poses = kit.poses([[k['t'], k['d'], k['p']] for k in P['keys']], day=.7)
        poses['hero']['defaults'].update(rest or {})
        return GpuRenderer(Film(data={'film': {'size': [270, 480], 'fps': 60, 'duration': P['total'], 'audio': [dict(id='voice', src=quiet)], 'vars': vars or {}, 'poses': poses}, 'layers': [Lh]}))
    # the first scene has no figure of the kit in it — only bodies with plain names. The kit's own strings are tied to
    # them by those names (its root stood under the floor and hopped in centimetres; it had no legs)
    from kadr import figure
    St = figure.strings(L['made']['figures'][0], 'voice')
    tied = {{'signL': 'dollarL', 'signR': 'dollarR'}.get(k[5:], k[5:]): v for k, v in St['nodes'].items() if k not in ('hero.body', 'hero.legL', 'hero.legR')}
    tied['root'] = {'position': "=[0, -3 + pose('hero', 'y') * 11 - pose('hero', 'sag') * 1.2, -22]"}
    old = film(os.path.join(FIRST, 'scene.json'), vars=St['vars'], rest={k: L['made']['figures'][0]['rest'][k] for k in ('armLf', 'armRf')}, nodes=tied, materials={'lampBulb': {'shade': "=(1 - .8 * pose('hero', 'day')) / .44"}})     # that scene was stored at day .7
    now = film(os.path.join(HERO, 'scene.yaml'))
    for t, what in ((3.2, 'крупный план'), (10.8, 'общий')):
        x, y = old.frame_u8(t)[..., :3].astype(int), now.frame_u8(t)[..., :3].astype(int)
        d = np.abs(x - y)
        ok(f'кадр на {t} с ({what}) — как с первой сценой', d.mean() < 2.5 and (d.max(-1) > 30).mean() < .003, f'средняя разница {d.mean():.2f} из 255, заметно отличается {(d.max(-1) > 30).mean() * 100:.3f} % точек')
    x, y = old.frame_u8(34.3)[..., :3].astype(int), now.frame_u8(34.3)[..., :3].astype(int)
    blue = lambda a: int(((a[..., 2] > a[..., 0] + 50) & (a[..., 2] > 100) & (a[..., 1] < 120))[240:].sum())
    ok('на плане три четверти видна нога в джинсах, которой в первой сцене не было', blue(y) > blue(x) + 150, f'синих точек внизу кадра {blue(x)} → {blue(y)}')
else:
    print('ПРОПУЩЕНО: нет examples/hero3d')

print()
print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ СХОДИТСЯ: {len(bad)} — ' + '; '.join(bad))
sys.exit(1 if bad else 0)
