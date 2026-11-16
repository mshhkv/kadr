"""Known answers for models made elsewhere (kadr/gltf.py; `shape: model` of a 3D scene): a .glb is written here by hand
— a tree of nodes, two meshes, a picture, two animations — and what Kadr draws of it is compared with what the file
says. python tests/ae/test_gltf.py"""
import io, json, math, os, struct, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from PIL import Image
from kadr.doc import Film, DocError
from kadr.gpu import GpuRenderer
from kadr.render import Renderer, to_u8
from kadr import gltf

bad = []
D = tempfile.mkdtemp()


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


# ---- the file
blob, views, accs = bytearray(), [], []


def put(a, kind, comp, target=None, minmax=False):
    a = np.ascontiguousarray(a)
    while len(blob) % 4:
        blob.append(0)
    views.append(dict(buffer=0, byteOffset=len(blob), byteLength=a.nbytes, **({'target': target} if target else {})))
    blob.extend(a.tobytes())
    acc = dict(bufferView=len(views) - 1, componentType=comp, count=len(a), type=kind)
    if minmax:
        acc.update(min=a.min(0).tolist(), max=a.max(0).tolist())
    accs.append(acc)
    return len(accs) - 1


def cube():
    P, N, I = [], [], []
    for ax in range(3):
        for sg in (1, -1):
            u, v = [(1, 2), (2, 0), (0, 1)][ax]
            base = len(P)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0, 0, 0]
                p[ax], p[u], p[v] = sg, a * sg, b
                P.append(p)
                n = [0, 0, 0]
                n[ax] = sg
                N.append(n)
            I += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(P, np.float32), np.array(N, np.float32), np.array(I, np.uint16)


P, N, I = cube()
a_pos, a_nrm, a_idx = put(P, 'VEC3', 5126, 34962, True), put(N, 'VEC3', 5126, 34962), put(I, 'SCALAR', 5123, 34963)
QP = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32)
QUV = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)             # glTF: a picture's row 0 is v = 0
q_pos, q_nrm = put(QP, 'VEC3', 5126, 34962, True), put(np.tile([0, 0, 1], (4, 1)).astype(np.float32), 'VEC3', 5126, 34962)
q_uv, q_idx = put(QUV, 'VEC2', 5126, 34962), put(np.array([0, 1, 2, 0, 2, 3], np.uint16), 'SCALAR', 5123, 34963)
S = (P[I] * .5).astype(np.float32)                                       # a small cube with no normals and no indices
s_pos = put(S, 'VEC3', 5126, 34962, True)
pic = np.zeros((2, 2, 3), np.uint8)
pic[0, 0], pic[0, 1], pic[1, 0], pic[1, 1] = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
png = io.BytesIO()
Image.fromarray(pic).resize((64, 64), Image.NEAREST).save(png, 'PNG')
while len(blob) % 4:
    blob.append(0)
views.append(dict(buffer=0, byteOffset=len(blob), byteLength=len(png.getvalue())))
blob.extend(png.getvalue())
img_view = len(views) - 1
t_move, v_move = put(np.array([0, 1], np.float32), 'SCALAR', 5126, None, True), put(np.array([[3, 0, 0], [3, 4, 0]], np.float32), 'VEC3', 5126)
h = math.sin(math.pi / 2)
t_turn, v_turn = put(np.array([0, 2], np.float32), 'SCALAR', 5126, None, True), put(np.array([[0, 0, 0, 1], [0, 0, h, math.cos(math.pi / 2)]], np.float32), 'VEC4', 5126)
J = dict(asset=dict(version='2.0'), scene=0, scenes=[dict(nodes=[0])],
         nodes=[dict(name='rig', children=[1, 2, 3]),
                dict(name='cube', mesh=0, translation=[3, 0, 0], scale=[1, 2, 1]),
                dict(name='quad', mesh=1, translation=[-3, 0, 0], skin=0),
                dict(name='small', mesh=2, matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 5, 0, 1]),
                dict(name='stray', mesh=0, translation=[8, 0, 0])],      # not in the scene: must not be drawn
         meshes=[dict(primitives=[dict(attributes=dict(POSITION=a_pos, NORMAL=a_nrm), indices=a_idx, material=0)]),
                 dict(primitives=[dict(attributes=dict(POSITION=q_pos, NORMAL=q_nrm, TEXCOORD_0=q_uv), indices=q_idx, material=1)]),
                 dict(primitives=[dict(attributes=dict(POSITION=s_pos))])],
         materials=[dict(pbrMetallicRoughness=dict(baseColorFactor=[1, 0, 0, 1], metallicFactor=0, roughnessFactor=1)),
                    dict(pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0), doubleSided=True, normalTexture=dict(index=0))],
         textures=[dict(source=0, sampler=0)], images=[dict(bufferView=img_view, mimeType='image/png')], samplers=[dict(wrapS=33071, wrapT=33071)],
         skins=[dict(joints=[0])],
         animations=[dict(name='move', channels=[dict(sampler=0, target=dict(node=1, path='translation'))], samplers=[dict(input=t_move, output=v_move, interpolation='LINEAR')]),
                     dict(name='turn', channels=[dict(sampler=0, target=dict(node=2, path='rotation'))], samplers=[dict(input=t_turn, output=v_turn, interpolation='LINEAR')])],
         accessors=accs, bufferViews=views, buffers=[dict(byteLength=len(blob))])
js = json.dumps(J).encode()
js += b' ' * (-len(js) % 4)
while len(blob) % 4:
    blob.append(0)
GLB = os.path.join(D, 'rig.glb')
with open(GLB, 'wb') as f_:
    f_.write(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(js) + 8 + len(blob)) + struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(blob), 0x004E4942) + bytes(blob))

# ---- what was read
M = gltf.load(GLB)
lo, hi = gltf.bounds(M)
ok('файл прочитан: узлы, сетки, материалы, картинка, анимации', len(M['nodes']) == 5 and len(M['meshes']) == 3 and len(M['mats']) == 3 and len(M['texs']) == 2 and [a['name'] for a in M['anims']] == ['move', 'turn'],
   f"узлов {len(M['nodes'])}, сеток {len(M['meshes'])}, анимаций {[a['name'] for a in M['anims']]}")
ok('рамка модели в покое — по дереву узлов', np.allclose(lo, [-4, -2, -1]) and np.allclose(hi, [4, 5.5, 1]), f'{lo.tolist()} … {hi.tolist()} (ждём [-4, -2, -1] … [4, 5.5, 1])')
ok('чего в модели не прочитано — названо', any('скелеты' in x for x in M['left_out']), ', '.join(M['left_out']))
v, ix = M['meshes'][2]
ok('нормалей в файле нет — сделаны по треугольникам', len(v) == 36 and np.allclose(np.abs(v[:, 3:6]).max(1), 1) and float((v[:, :3] * v[:, 3:6]).sum(1).min()) > .49, f'{len(v)} вершин, нормали смотрят наружу')

# ---- what is drawn: 20 units across a 400 px frame, seen from far away
W = 400
U = W / 20


def frame(node, t=0.0, engine=GpuRenderer):
    scene = dict(camera=dict(position=[0, 0, 2000], look_at=[0, 0, 0], fov=math.degrees(2 * math.atan(10 / 2000))), lights=[], look=dict(raw=True, glow=0),
                 nodes=[dict(dict(shape='model', src=GLB, unlit=True), **node)])
    f = Film(data={'film': {'size': [W, W], 'fps': 25, 'duration': 4, 'background': '#000000'}, 'layers': [dict(id='s', type='scene3d', scene=scene, samples=4)]})
    R = engine(f)
    return (R.frame_u8(t) if engine is GpuRenderer else to_u8(R.frame(t))).astype(int), f


def box(mask):
    ys, xs = np.nonzero(mask)
    return (W / 2 - (ys.max() + 1)) / U * -1 if False else ((xs.min() - W / 2) / U, (xs.max() + 1 - W / 2) / U, (W / 2 - ys.max() - 1) / U, (W / 2 - ys.min()) / U)


red = lambda a: (a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] < 60)
a, f0 = frame(dict(origin='model', animation=False))
x0, x1, y0, y1 = box(red(a) & (np.arange(W)[None] > W / 2))
ok('куб стоит и вытянут, как велит его узел', max(abs(x0 - 2), abs(x1 - 4), abs(y0 + 2), abs(y1 - 2)) < .08, f'x {x0:.2f} … {x1:.2f}, y {y0:.2f} … {y1:.2f} (ждём 2 … 4, −2 … 2)')
at = lambda im, x, y: im[int(W / 2 - y * U), int(W / 2 + x * U)]
quad = [at(a, -3.5, .5), at(a, -2.5, .5), at(a, -3.5, -.5), at(a, -2.5, -.5)]
ok('картинка на модели стоит как в файле', [tuple((c > 127).astype(int)) for c in quad] == [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)], 'левый верх, правый верх, левый низ, правый низ: ' + ' '.join(str(c.tolist()) for c in quad))
ok('узел с матрицей и узел вне сцены', a[int(W / 2 - 5 * U), W // 2].max() > 200 and a[:, int(W / 2 + 6.5 * U):].max() < 30, f'малый куб на высоте 5: {a[int(W / 2 - 5 * U), W // 2].tolist()}; справа, где узел вне сцены, пусто')
a, _ = frame(dict(origin='model', animation='move'), t=.52)          # frame 13 of 25 a second
x0, x1, y0, y1 = box(red(a) & (np.arange(W)[None] > W / 2))
ok('анимация «move»: на 0,52 с куб поднялся на 2,08', abs(y0 - .08) < .1 and abs(y1 - 4.08) < .1, f'y {y0:.2f} … {y1:.2f} (ждём 0.08 … 4.08; кадр смазан движением на 0,08)')
a, _ = frame(dict(origin='model', animation='move'), t=1.52)
x0, x1, y0, y1 = box(red(a) & (np.arange(W)[None] > W / 2))
ok('она идёт по кругу (loop)', abs(y0 - .08) < .1, f'на 1,52 с y {y0:.2f} … {y1:.2f} (как на 0,52 с)')
a, _ = frame(dict(origin='model', animation='move', loop=False, speed=.5), t=1.04)
x0, x1, y0, y1 = box(red(a) & (np.arange(W)[None] > W / 2))
ok('speed: 0.5 — вдвое медленнее', abs(y0 - .08) < .1, f'на 1,04 с y {y0:.2f} … {y1:.2f} (как на 0,52 с при обычной скорости)')
a, _ = frame(dict(origin='model', animation='turn'), t=1.0)
quad = [at(a, -3.5, .5), at(a, -2.5, .5), at(a, -3.5, -.5), at(a, -2.5, -.5)]
ok('анимация «turn»: четверть оборота — картинка повернулась', [tuple((c > 127).astype(int)) for c in quad] == [(0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1)],
   'левый верх, правый верх, левый низ, правый низ: ' + ' '.join(str(c.tolist()) for c in quad))
a, _ = frame(dict(height=15, animation=False, position=[0, -7.5, 0]))
ys, xs = np.nonzero(a.max(2) > 60)
ok('height и origin: base — модель нужной высоты стоит на своём узле', abs((W / 2 - ys.min()) / U - 7.5) < .1 and abs((W / 2 - ys.max() - 1) / U + 7.5) < .1 and abs((xs.min() + xs.max() + 1) / 2 - W / 2) < 1.5,
   f'от {(W / 2 - ys.max() - 1) / U:.2f} до {(W / 2 - ys.min()) / U:.2f} (ждём −7.5 … 7.5), по ширине посередине')
a, _ = frame(dict(origin='model', animation=False, color='#00ff00'))
ok('цвет на узле перекрашивает модель', at(a, 3, 0)[1] > 250 and at(a, 3, 0)[0] < 5, str(at(a, 3, 0).tolist()))
a, f1 = frame(dict(origin='model', animation='move'), t=.5)
b, _ = frame(dict(origin='model', animation='move'), t=.5, engine=Renderer)
ok('модель одинакова на обоих рендерах', np.abs(a - b).max() <= 3, f'наибольшая разница {np.abs(a - b).max()}')
ok('в документе сказано, чего в модели не прочитано', any(i.level == 'warn' and 'скелеты' in i.msg for i in f1.issues), next((str(i)[:110] for i in f1.issues if i.level == 'warn'), 'предупреждения нет'))
open(os.path.join(D, 'junk.glb'), 'wb').write(b'not a model at all')
for what, node in (('нет файла модели — названо', dict(src='nope.glb')), ('не модель — названо', dict(src=os.path.join(D, 'junk.glb'))), ('нет такой анимации — названо', dict(animation='jump')), ('опечатка в origin названа', dict(origin='bottom'))):
    try:
        frame(node)
        ok(what, False, 'документ принят')
    except DocError as e:
        ok(what, True, str(e).splitlines()[0][:120])
    except Exception as e:
        ok(what, False, f'падение: {type(e).__name__}: {e}')

# ---- a material's other pictures: relief, roughness / metal / surrounding light, glow — three squares side by side
blob, views, accs = bytearray(), [], []
q_pos, q_nrm = put(QP, 'VEC3', 5126, 34962, True), put(np.tile([0, 0, 1], (4, 1)).astype(np.float32), 'VEC3', 5126, 34962)
q_uv, q_idx = put(QUV, 'VEC2', 5126, 34962), put(np.array([0, 1, 2, 0, 2, 3], np.uint16), 'SCALAR', 5123, 34963)


def image(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, 'PNG')
    while len(blob) % 4:
        blob.append(0)
    views.append(dict(buffer=0, byteOffset=len(blob), byteLength=len(b.getvalue())))
    blob.extend(b.getvalue())
    return dict(bufferView=len(views) - 1, mimeType='image/png')


tilt = lambda x, y: [int(round((v * .5 + .5) * 255)) for v in (x * .5, y * .5, math.cos(math.radians(30)))]      # leaning 30° towards (x, y)
nrm = np.zeros((32, 64, 3), np.uint8)
nrm[:, :32], nrm[:, 32:] = tilt(1, 0), tilt(0, 1)                         # left half leans right, right half leans up
orm = np.zeros((32, 64, 3), np.uint8)
orm[:, :32], orm[:, 32:] = (128, 255, 255), (255, 255, 0)                 # left: half the surrounding light, metal; right: all of it, not metal
glow = np.zeros((64, 32, 3), np.uint8)
glow[:32] = 255                                                           # the upper half glows
J = dict(asset=dict(version='2.0'), scene=0, scenes=[dict(nodes=[0, 1, 2])],
         nodes=[dict(name='relief', mesh=0, translation=[-3, 0, 0]), dict(name='metal', mesh=1), dict(name='glow', mesh=2, translation=[3, 0, 0])],
         meshes=[dict(primitives=[dict(attributes=dict(POSITION=q_pos, NORMAL=q_nrm, TEXCOORD_0=q_uv), indices=q_idx, material=k)]) for k in range(3)],
         materials=[dict(pbrMetallicRoughness=dict(metallicFactor=0, roughnessFactor=1), normalTexture=dict(index=0)),
                    dict(pbrMetallicRoughness=dict(metallicFactor=1, roughnessFactor=1, metallicRoughnessTexture=dict(index=1)), occlusionTexture=dict(index=1)),
                    dict(pbrMetallicRoughness=dict(baseColorFactor=[0, 0, 0, 1], metallicFactor=0), emissiveFactor=[1, 1, 1], emissiveTexture=dict(index=2))],
         textures=[dict(source=0), dict(source=1), dict(source=2)], images=[image(nrm), image(orm), image(glow)],
         accessors=accs, bufferViews=views, buffers=[dict(byteLength=len(blob))])
js = json.dumps(J).encode()
js += b' ' * (-len(js) % 4)
while len(blob) % 4:
    blob.append(0)
MAPS = os.path.join(D, 'maps.glb')
with open(MAPS, 'wb') as f_:
    f_.write(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(js) + 8 + len(blob)) + struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(blob), 0x004E4942) + bytes(blob))
ok('у этой модели прочитано всё', gltf.load(MAPS)['left_out'] == [], str(gltf.load(MAPS)['left_out']))
Image.fromarray(np.full((32, 64, 3), 255, np.uint8)).save(os.path.join(D, 'white.png'))


def lit(lights, env=None, flat=False):
    """the three squares in linear light (0 … 1) under these lights; flat: the same squares without their pictures"""
    scene = dict(camera=dict(position=[0, 0, 2000], look_at=[0, 0, 0], fov=math.degrees(2 * math.atan(10 / 2000))), lights=lights, look=dict(raw=True, glow=0),
                 nodes=[dict(shape='model', src=MAPS, origin='model')] if not flat else [dict(shape='plane', size=[2, 2], position=[x, 0, 0], color='#ffffff', roughness=1) for x in (-3, 0, 3)])
    if env:
        scene['environment'] = env
    f = Film(data={'film': {'size': [W, W], 'fps': 25, 'duration': 1, 'background': '#000000'}, 'layers': [dict(id='s', type='scene3d', scene=scene, samples=4)]})
    return GpuRenderer(f).frame_u8(0).astype(float) / 255


sun = lambda p: [dict(type='sun', position=p, target=[0, 0, 0], intensity=math.pi * .8)]     # a white matte face-on shows as 0.8
front, plain = lit(sun([0, 0, 100])), lit(sun([0, 0, 100]), flat=True)
px = lambda im, x, y: float(im[int(W / 2 - y * U), int(W / 2 + x * U), 1])
ok('рельеф: наклон на 30° темнит на косинус', abs(px(front, -3.5, 0) / px(plain, -3, 0) - math.cos(math.radians(30))) < .03 and abs(px(front, -2.5, 0) / px(plain, -3, 0) - math.cos(math.radians(30))) < .03,
   f'ровное {px(plain, -3, 0):.3f}; наклонённые половины {px(front, -3.5, 0):.3f} и {px(front, -2.5, 0):.3f} (ждём × {math.cos(math.radians(30)):.3f})')
side, up = lit(sun([100, 0, 0])), lit(sun([0, 100, 0]))
ok('рельеф: куда наклонено — оттуда и светло', abs(px(side, -3.5, 0) - .4) < .05 and px(side, -2.5, 0) < .03 and abs(px(up, -2.5, 0) - .4) < .05 and px(up, -3.5, 0) < .03,
   f'свет справа: левая половина (наклон вправо) {px(side, -3.5, 0):.3f}, правая (наклон вверх) {px(side, -2.5, 0):.3f}; свет сверху: {px(up, -3.5, 0):.3f} и {px(up, -2.5, 0):.3f} (ждём 0.4 и 0)')
ok('металл по картинке: рассеянного света не отдаёт', .2 < px(front, -.5, 0) < .6 * px(front, .5, 0) and abs(px(front, .5, 0) - px(plain, 0, 0)) < .03,
   f'металлическая половина {px(front, -.5, 0):.3f}, неметаллическая {px(front, .5, 0):.3f} (как ровная белая: {px(plain, 0, 0):.3f})')
amb = lit([], env=dict(src=os.path.join(D, 'white.png'), intensity=.5))
ok('затенение по картинке: половина окружающего света', abs(px(amb, .5, 0) - .5) < .03 and abs(px(amb, -.5, 0) - .25) < .03, f'{px(amb, -.5, 0):.3f} и {px(amb, .5, 0):.3f} при окружении 0.5 (ждём 0.25 и 0.5)')
dark = lit([])
ok('свечение по картинке', px(dark, 3, .5) > .97 and px(dark, 3, -.5) < .02 and px(dark, 0, 0) < .02, f'верхняя половина {px(dark, 3, .5):.3f}, нижняя {px(dark, 3, -.5):.3f}')

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
