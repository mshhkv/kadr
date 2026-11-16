"""kadr depth: a depth map comes out beside the photo, as large as the photo, using its whole range, the same on a
second run — when a Python with torch and transformers and the model are on this machine (else the test is skipped).
python tests/ae/test_depth.py"""
import os, shutil, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from PIL import Image
from kadr import depth

bad = []


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


# ---- --smooth, without the model: a tilted wall with a dome of «relief» on it (a printed bowl) in front of a far background
sys.path.insert(0, os.path.join(HERE, '..', '..', 'kadr'))
from _depth_smooth import smooth
yy, xx = np.mgrid[0:300, 0:400].astype(np.float64)
wall = 0.55 + 0.0006 * (xx - 200) - 0.0004 * (yy - 150)        # the true face
n = np.full((300, 400), 0.05)                                  # far background
face = (xx >= 60) & (xx < 340) & (yy >= 40) & (yy < 260)
n[face] = wall[face]
dome = np.clip(1 - np.hypot(xx - 200, yy - 170) / 80, 0, 1)
n[face] += 0.3 * dome[face]                                    # the relief the model invents under a print
scene = n
for spill in (10, 4, 0):                                       # how far the region runs over the face's edge
    n = scene.copy()
    share = smooth(n, [[60 - spill, 40 - spill], [339 + spill, 40 - spill], [339 + spill, 259 + spill], [60 - spill, 259 + spill]], 2, np.random.default_rng(7))
    dev = np.abs(n - wall)[face].max()
    ok(f'--smooth, область шире грани на {spill} px: рельеф печати убран, осталась гладкая грань', dev < 0.02, f'отклонение от настоящей грани до {dev:.3f} (было {np.abs(scene - wall)[face].max():.2f})')
    ok(f'--smooth, область шире грани на {spill} px: фон за краем предмета не тронут', np.abs(n - scene)[~face].max() < 1e-9)
    ok(f'--smooth, область шире грани на {spill} px: говорит, по какой доле области нашёл поверхность', 0.3 < share < 0.9, f'{share:.2f}')

if depth.ml_python() is None:
    print('ПРОПУЩЕНО (сама модель): нет Python с torch и transformers (KADR_ML_PYTHON или ~/.kadr/ml_python)')
    sys.exit(1 if bad else 0)
d = tempfile.mkdtemp()
src = os.path.join(d, 'photo.png')
# a scene the model cannot misread: a bright ball in front of a wall that fades into the distance
y, x = np.mgrid[0:360, 0:480].astype(np.float32)
wall = np.clip(200 - y * 0.35, 0, 255)
img = np.stack([wall * 0.9, wall, wall * 1.05], -1)
r = np.hypot(x - 240, y - 220)
ball = np.clip(1 - r / 90, 0, 1) ** 0.5
img = np.where((r < 90)[..., None], np.stack([255 * ball, 120 * ball + 60, 40 + 0 * ball], -1), img)
Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(src)
try:
    rep = depth.run([src])
except RuntimeError as e:
    print('ПРОПУЩЕНО (сама модель):', e)
    sys.exit(1 if bad else 0)
out = depth.out_path(src)
a = np.asarray(Image.open(out))
ok('карта лежит рядом с фото и того же размера', os.path.exists(out) and a.shape == (360, 480), str(a.shape))
ok('карта занимает весь диапазон от далёкого к близкому', a.min() <= 8 and a.max() >= 247, f'{a.min()}…{a.max()}')
ok('шар перед стеной ближе стены вокруг него', a[220, 240] > a[220, 60] + 20 and a[220, 240] > a[60, 240] + 20, f'шар {a[220, 240]}, стена {a[220, 60]} и {a[60, 240]}')
shutil.copy(out, out + '.first')
depth.run([src])
ok('второй запуск даёт ту же карту', np.abs(np.asarray(Image.open(out)).astype(int) - np.asarray(Image.open(out + '.first')).astype(int)).max() <= 1)
print(f"     модель загрузилась за {rep['load_seconds']} с, карта за {rep['maps'][0]['seconds']} с, считала на {rep['device']}")
print('ВСЁ СХОДИТСЯ' if not bad else f'ЕСТЬ РАСХОЖДЕНИЯ: {bad}')
sys.exit(1 if bad else 0)
