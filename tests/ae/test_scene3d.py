"""The 3D layer (kadr/scene3d.py, core/src/scene3d.rs) against known answers: where a point lands on the frame, how
bright a surface is under a lamp (inverse squares, the cosine law, the textbook value), where a shadow falls, how wide
a thing out of focus or in motion smears — and the hero scene against the frames its first renderer drew.
python tests/ae/test_scene3d.py"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr import scene3d

bad = []


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def shot(scene, t=0.0, w=400, h=400, fps=25, samples=1, still=True, **layer):
    """the scene's picture in linear light (no curve, no glow): (h, w, 3) 0..1"""
    scene.setdefault('look', {}).update(raw=True, glow=0)
    f = Film(data={'film': {'size': [w, h], 'fps': fps, 'duration': 5, 'background': '#000000'}, 'layers': [dict(id='s', type='scene3d', scene=scene, **layer)]})
    L = f.layers[0]
    sc = scene3d.layer_scene(L)
    G, insts, mats, draws, lk = sc.frame(t, w, h, fps, samples=samples, flags={'still': still})
    a = sc.gpu().render(w, h, np.ascontiguousarray(G), np.ascontiguousarray(insts), np.ascontiguousarray(mats), np.ascontiguousarray(draws), lk, [0.0, 0.0, 0.0, 0.0]).download()
    return a[..., :3] / 255


LAMP = 0.0318                                              # a white glowing ball in linear light: its emissive × glow shows as itself


def ball(pos, r=2.0, **kw):
    return dict(shape='sphere', radius=r, position=pos, color='#000000', emissive='#ffffff', glow=1, cast=False, **kw)


# ---- 1. a point of the world lands where perspective says
cam = dict(position=[0, 0, 100], look_at=[0, 0, 0], fov=40)
a = shot(dict(camera=cam, nodes=[ball([12, 7, 20], 1.0)]))
ys, xs = np.nonzero(a[..., 0] > .5)
f = 200 / math.tan(math.radians(20))                       # focal length in px of a 400 px frame with a 40° view
ex, ey = 200 + 12 * f / 80, 200 - 7 * f / 80
ok('точка мира попадает в кадр туда, куда велит перспектива', abs(xs.mean() + .5 - ex) < .6 and abs(ys.mean() + .5 - ey) < .6, f'({xs.mean() + .5:.1f}, {ys.mean() + .5:.1f}) при ({ex:.1f}, {ey:.1f})')

# ---- 2. a matt wall under a lamp: the textbook brightness, inverse squares, the cosine law
def wall(d, tilt=0.0, rough=1.0):
    sc = dict(camera=dict(position=[0, 0, 300], look_at=[0, 0, 0], fov=10),
              lights=[dict(type='point', position=[0, 0, d], color='#ffffff', intensity=20000)],
              nodes=[dict(shape='plane', size=[400, 400], rotation=[tilt, 0, 0], color='#bcbcbc', roughness=rough, cast=False, receive=False)])
    return float(shot(sc, w=64, h=64)[32, 32, 1])
rho = float(scene3d.srgb([188, 188, 188])[0])
b1, b2 = wall(100), wall(200)
want = rho / math.pi * 20000 / 100 ** 2                    # Lambert: albedo / π × illuminance
ok('матовая стена под лампой: яркость как по учебнику (альбедо / π × освещённость)', want <= b1 <= want * 1.05, f'{b1:.4f} при {want:.4f} (сверх того — слабый блик)')
ok('вдвое дальше лампа — вчетверо темнее', abs(b1 / b2 - 4) < .04, f'в {b1 / b2:.3f} раза')
b60 = wall(100, tilt=60)
ok('стена, повёрнутая на 60° к лампе, вдвое темнее (закон косинуса)', abs(b60 / b1 - .5) < .03, f'в {b1 / b60:.3f} раза')

# ---- 3. a shadow falls where the lamp's rays put it
sc = dict(camera=dict(position=[0, 400, 0.01], look_at=[0, 0, 0], fov=20),
          lights=[dict(type='spot', position=[0, 200, 0], target=[0, 0, 0], intensity=40000, angle=40, penumbra=0, shadow=True)],
          nodes=[dict(shape='box', size=[300, 2, 300], position=[0, -1, 0], color='#bbbbbb', roughness=1, cast=False),
                 dict(shape='box', size=[20, 20, 20], position=[0, 30, 0], color='#bbbbbb', roughness=1)])
a = shot(sc, w=600, h=600)
row = a[300, :, 1]
f = 300 / math.tan(math.radians(10))
px = lambda x: 300 + x * f / 400                           # a floor point's column
lit = float(row[int(px(40))])
edge = next(c for c in range(int(px(11.7)) + 2, 600) if row[c] > lit * .5)     # past the box's own outline (11.5 on the floor)
want = 12.5                                                # the top edge (10, 40) thrown from 200 high: 10 × 200 / 160
got = (edge - 300) * 400 / f
ok('тень коробки кончается там, куда её бросает лампа', abs(got - want) < .5, f'край тени на {got:.2f} при {want} (по полу, от центра)')
sc['lights'][0]['soft'] = 12
b = shot(sc, w=600, h=600, samples=32, still=False)[300, :, 1]
ramp = np.nonzero((b[int(px(11.7)) + 2:] > lit * .15) & (b[int(px(11.7)) + 2:] < lit * .85))[0]
ok('широкая лампа (soft) даёт мягкий край тени', len(ramp) >= 4, f'переход шириной {len(ramp)} px, был резкий')

# ---- 4. depth of field: what is off the focus spreads into a disc of the size geometry gives
def blob(a):
    cols = np.nonzero(a[..., 0].max(0) > a[..., 0].max() * .5)[0]
    return cols[-1] - cols[0] + 1
cam = dict(position=[0, 0, 100], look_at=[0, 0, 0], fov=20, lens=3, focus=100)
f = 200 / math.tan(math.radians(10))
sharp = blob(shot(dict(camera=cam, nodes=[ball([0, 0, 0], 2)]), samples=64, still=False))
far = blob(shot(dict(camera=cam, nodes=[ball([0, 0, -100], 2)]), samples=64, still=False))
coc = 2 * 3 * (200 - 100) / 100 * f / 200                  # the lens seen from the focus point, carried on to the ball's distance, in px there
ok('в фокусе шар резкий', abs(sharp - 4 * f / 100) < 2.5, f'{sharp} px при {4 * f / 100:.1f}')
ok('вне фокуса шар расплывается в кружок по геометрии объектива', abs(far - coc) < coc * .2, f'{far} px при кружке {coc:.1f} px')

# ---- 5. motion blur: a thing smears by what it travels while the shutter is open
sc = dict(camera=dict(position=[0, 0, 100], look_at=[0, 0, 0], fov=20), nodes=[ball('=[-200 + 400 * t, 0, 0]', 2)])
mv = blob(shot(sc, t=.5, samples=64, still=False))
want = 400 * .5 / 25 * f / 100                            # the path of half a frame: a ball smeared further than its own
ok('движущийся шар смазан на путь за время затвора', abs(mv - want) < 3, f'{mv} px при {want:.1f}')   # width is half as bright over just that length

# ---- 6. the same frame twice; a mistake in the document is said in words
a1 = shot(dict(camera=cam, nodes=[ball([3, 2, 0], 2)]), samples=16, still=False)
a2 = shot(dict(camera=cam, nodes=[ball([3, 2, 0], 2)]), samples=16, still=False)
ok('один и тот же кадр дважды — одна и та же картинка', np.array_equal(a1, a2))
try:
    Film(data={'film': {'size': [100, 100], 'fps': 25, 'duration': 1}, 'layers': [dict(id='s', type='scene3d', scene=dict(nodes=[dict(shape='spere', radius=2)]))]})
    ok('опечатка в сцене названа', False)
except DocError as e:
    ok('опечатка в сцене названа', 'spere' in str(e) and 'sphere' in str(e), str(e).splitlines()[0][:80])

# ---- 7. a character's room (a private scene of the owner's; skipped without it): the scene exported from its first renderer, against that renderer's frames
EX = os.path.join(HERE, '..', '..', 'examples', 'hero3d', 'first')      # the scene as its first renderer made it; the living one is ../scene.yaml (test_scene_words.py)
if os.path.exists(os.path.join(EX, 'scene.json')):
    from PIL import Image
    hero = scene3d.load(os.path.join(EX, 'scene.json'), os.path.join(EX, 'clip.bin'), 0)
    for name in sorted(os.listdir(os.path.join(EX, 'ref'))):
        t = float(name[1:-4])
        r = np.asarray(Image.open(os.path.join(EX, 'ref', name)).convert('RGB')).astype(int)
        a = np.clip(hero.render(round(t * 60) / 60, r.shape[1], r.shape[0], 60).download()[..., :3], 0, 255).astype(int)
        d = np.abs(a - r)
        ok(f'герой на {t:g} с — как у прежнего рендера', d.mean() < 3 and (d.max(-1) > 30).mean() < .002, f'средняя разница {d.mean():.2f} из 255, заметно отличается {(d.max(-1) > 30).mean() * 100:.3f} % пикселей')
else:
    print('ПРОПУЩЕНО: нет examples/hero3d')

# ---- pictures on bodies and the environment (kadr/envmap.py)
import tempfile
from PIL import Image
from kadr import envmap
TD = tempfile.mkdtemp()
q4 = np.zeros((200, 300, 3), np.uint8)
q4[:100, :150], q4[:100, 150:], q4[100:, :150], q4[100:, 150:] = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
Image.fromarray(q4).save(os.path.join(TD, 'quad.png'))
far = dict(position=[0, 0, 1000], look_at=[0, 0, 0], fov=6)       # nearly parallel rays: a body's normal is where it is on screen
R = 40 / (1000 * math.tan(math.radians(3))) * 200                  # a 40-unit radius on a 400 px frame
a = shot(dict(camera=far, lights=[], nodes=[dict(shape='plane', size=[90, 60], map=os.path.join(TD, 'quad.png'), unlit=True)]))
px = lambda im, x, y: im[int(round(200 - y * R / 40)), int(round(200 + x * R / 40))]
quad = [px(a, -20, 15), px(a, 20, 15), px(a, -20, -15), px(a, 20, -15)]
ok('картинка на плоскости стоит как в файле', all(np.allclose(c, e, atol=.02) for c, e in zip(quad, ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)))),
   'левый верх, правый верх, левый низ, правый низ: ' + ' '.join(str(np.round(c, 2).tolist()) for c in quad))
a = shot(dict(camera=far, lights=[], nodes=[dict(shape='sphere', radius=40, map=os.path.join(TD, 'quad.png'), unlit=True)]))
ok('на шаре середина картинки смотрит на зрителя', np.allclose(px(a, -8, 8), (1, 0, 0), atol=.02) and np.allclose(px(a, 8, 8), (0, 1, 0), atol=.02) and np.allclose(px(a, 8, -8), (1, 1, 0), atol=.02),
   f'слева вверху от середины {np.round(px(a, -8, 8), 2).tolist()}, справа вверху {np.round(px(a, 8, 8), 2).tolist()}')
a = shot(dict(camera=far, lights=[], nodes=[dict(shape='plane', size=[90, 60], map=os.path.join(TD, 'quad.png'), tile=[3, 2], unlit=True)]))
row = a[int(200 - 14 * R / 40), int(200 - 44 * R / 40):int(200 + 44 * R / 40)]
ok('tile: [3, 2] повторяет картинку', int((np.abs(np.diff((row[:, 0] > .5).astype(int))) > 0).sum()) == 5, f'смен красного и зелёного по строке {int((np.abs(np.diff((row[:, 0] > .5).astype(int))) > 0).sum())} (ждём 5)')
try:
    shot(dict(camera=far, lights=[], nodes=[dict(shape='plane', size=[9, 6], map='нет_такого.png')]))
    ok('нет файла картинки — названо', False)
except DocError as e:
    ok('нет файла картинки — названо', 'нет файла' in str(e), str(e).splitlines()[0][:100])

d = envmap.directions(512, 256)                                    # a panorama whose every side has its own colour
ax = np.abs(d).argmax(-1)
sg = np.take_along_axis(d, ax[..., None], -1)[..., 0] > 0
eq = np.zeros(d.shape, np.float32)
SIDES = {(0, True): (1, 0, 0), (0, False): (0, 1, 1), (1, True): (1, 1, 1), (1, False): (0, 0, 0), (2, True): (0, 1, 0), (2, False): (0, 0, 1)}
for (a_, s_), c in SIDES.items():
    eq[(ax == a_) & (sg == s_)] = c
Image.fromarray((eq * 255).astype(np.uint8)).save(os.path.join(TD, 'sides.png'))
ok('середина панорамы — прямо впереди (−z), правее — справа (+x)', tuple(eq[128, 256]) == (0, 0, 1) and tuple(eq[128, 384]) == (1, 0, 0))
mirror = [dict(shape='sphere', radius=40, color='#ffffff', metal=1, roughness=0)]
a = shot(dict(camera=far, lights=[], environment=os.path.join(TD, 'sides.png'), nodes=mirror))
s45 = 40 * math.sin(math.radians(45))                              # a normal at 45° sends the look a quarter turn away
got = [px(a, 0, 0), px(a, s45, 0), px(a, -s45, 0), px(a, 0, s45), px(a, 0, -s45)]
ok('зеркальный шар отражает каждую сторону на своём месте', all(np.allclose(c, e, atol=.03) for c, e in zip(got, ((0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)))),
   'середина (сзади камеры, +z), вправо (+x), влево (−x), вверх, вниз: ' + ' '.join(str(np.round(c, 2).tolist()) for c in got))
a = shot(dict(camera=far, lights=[], environment=dict(src=os.path.join(TD, 'sides.png'), turn=90), nodes=mirror))
ok('turn: 90 поворачивает окружение', np.allclose(px(a, s45, 0), (0, 0, 1), atol=.03) and np.allclose(px(a, 0, 0), (1, 0, 0), atol=.03),
   f'справа теперь {np.round(px(a, s45, 0), 2).tolist()} (то, что было впереди), за камерой {np.round(px(a, 0, 0), 2).tolist()} (то, что было справа)')
Image.fromarray(np.full((64, 128, 3), 255, np.uint8)).save(os.path.join(TD, 'white.png'))
res = []
for name, node in (('матовое', dict(color='#ffffff', roughness=1)), ('гладкое', dict(color='#ffffff', roughness=.2)), ('металл', dict(color='#ffffff', metal=1, roughness=.5)), ('лак', dict(color='#ffffff', roughness=.6, clearcoat=1))):
    a = shot(dict(camera=far, lights=[], environment=dict(src=os.path.join(TD, 'white.png'), intensity=.5), nodes=[dict(node, shape='sphere', radius=40)]))
    res.append((name, float(px(a, 0, 0)[1]), float(px(a, 34, 0)[1])))
ok('белое тело в ровном окружении — само окружение (свет не теряется и не берётся ниоткуда)', all(abs(c - .5) < .03 and abs(e - .5) < .04 for _, c, e in res),
   '; '.join(f'{n}: середина {c:.3f}, у края {e:.3f}' for n, c, e in res) + ' (окружение 0.5)')
a = shot(dict(camera=far, lights=[], environment=dict(src=os.path.join(TD, 'white.png'), intensity=.5), nodes=[dict(shape='sphere', radius=40, color='#bcbcbc', roughness=1)]))
ok('серое матовое отдаёт свою долю света', abs(float(px(a, 0, 0)[1]) - .5 * (.5029 * .96 + .04)) < .02, f'{float(px(a, 0, 0)[1]):.3f} при окружении 0.5 и сером 0.503 (ждём около {.5 * (.5029 * .96 + .04):.3f})')
# a small lamp straight overhead in a dark surround: the blurred copies of the environment are bells of the widths the shader expects
lamp = np.zeros((512, 1024, 3), np.float32)
dd = envmap.directions(1024, 512)
lamp[np.degrees(np.arccos(np.clip(dd[..., 1], -1, 1))) < 2.0] = 1.0
at = envmap.bake(lamp).view(np.float16).astype(np.float32)
fits = []
for k, sig in ((1, .125), (3, .35), (6, .582)):                    # three of the six lods at the 16 px face
    fd = envmap._face_dirs(16)[1]                                  # the +y face
    tile = at[4 * (envmap.CUBE - 16):4 * (envmap.CUBE - 16) + 16, 48 * k + 16:48 * k + 32, 1]
    th = np.arccos(np.clip(fd[..., 1], -1, 1)).ravel()
    v = tile.ravel()
    m = v > v.max() * .1
    est = float(np.sqrt(-1 / (2 * np.polyfit(th[m] ** 2, np.log(v[m]), 1)[0])))     # ln v = ln A − θ² / 2σ²
    fits.append((sig, est))
ok('размытые копии окружения — колокола нужной ширины', all(abs(e - s_) < .18 * s_ + .02 for s_, e in fits), '; '.join(f'σ {s_}: вышло {e:.3f}' for s_, e in fits))
h1 = envmap.bake(envmap.studio()).tobytes()
ok('студия печётся одинаково от раза к разу', h1 == envmap.bake(envmap.studio()).tobytes())
import struct
hp = os.path.join(TD, 'two.hdr')                                    # a tiny Radiance picture written by hand: 1.0, 4.0 / 0.25, 0
with open(hp, 'wb') as f_:
    f_.write(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n' + bytes([128, 128, 128, 129, 128, 128, 128, 131, 128, 128, 128, 127, 0, 0, 0, 0]))
hd = envmap.read_hdr(hp)
ok('.hdr читается с настоящей яркостью', np.allclose(hd[..., 0], [[1.0, 4.0], [.25, 0.0]], rtol=.01), f'{hd[..., 0].tolist()} (ждём [[1, 4], [0.25, 0]])')

print('ВСЁ СХОДИТСЯ' if not bad else f'ЕСТЬ РАСХОЖДЕНИЯ: {bad}')
sys.exit(1 if bad else 0)
