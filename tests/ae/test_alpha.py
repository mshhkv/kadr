"""Known answers for export with transparency (kadr frame / render --alpha): the film drawn over nothing, laid back over
the film's own background, is the film — for a PNG, a ProRes 4444 .mov, a VP9 .webm and a folder of PNG frames.
python tests/ae/test_alpha.py"""
import glob, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import yaml
from PIL import Image
from kadr.doc import Film
from kadr.render import Renderer, to_u8
from kadr.gpu import GpuRenderer
from kadr import cli

bad = []
D = tempfile.mkdtemp()
BG = (0x20, 0x30, 0x50)
DOC = {'film': {'size': [320, 180], 'fps': 25, 'duration': 1.2, 'background': '#203050'}, 'layers': [
    dict(id='plate', type='shape', shape='rect', size=[150, 40], radius=10, position={0: [-160, 70], .6: [85, 70], 'ease': 'outCubic'}, fill='#ee3124cc'),
    dict(id='name', type='text', text='KADR', font='Arial', size=28, color='#ffffff', position=[160, 92]),
    dict(id='glow', type='shape', shape='ellipse', size=[60, 60], anchor=[30, 30], position=[260, 45], fill='#ffd60a', blur=9, opacity=.8)]}
Y = os.path.join(D, 'f.yaml')
yaml.safe_dump(DOC, open(Y, 'w'), allow_unicode=True)


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def over(rgba):
    """a picture with transparency laid over the film's background"""
    a = rgba[..., 3:].astype(float) / 255
    return rgba[..., :3].astype(float) * a + np.array(BG, float) * (1 - a)


T = .8
for name, R in (('эталонный рендер', Renderer), ('видеокарта', GpuRenderer)):
    full = to_u8(R(Film(Y)).frame(T)).astype(float) if R is Renderer else R(Film(Y)).frame_u8(T).astype(float)
    cut = cli._rgba(R(Film(Y)), T)
    d = np.abs(over(cut) - full)
    ok(f'кадр без фона, положенный на фон, — тот же кадр ({name})', d.max() <= 2.5 and cut[5, 5, 3] == 0 and cut[84:100, 120:200, 3].max() == 255 and 195 < cut[76, 100, 3] < 213,
       f'наибольшая разница {d.max():.1f} из 255; пусто — прозрачно, буква — глухая, плашка — {cut[76, 100, 3]} из 255 (в документе cc = 204)')
want = cli._rgba(GpuRenderer(Film(Y)), 20 / 25 + cli.SAMPLE)
run = lambda *a: subprocess.run([sys.executable, '-m', 'kadr', *a], capture_output=True, text=True, cwd=os.path.join(HERE, '..', '..'))
r = run('frame', Y, '--at', str(T), '--alpha', '--gpu', '-o', os.path.join(D, 'one.png'))
one = np.asarray(Image.open(os.path.join(D, 'one.png')))
ok('kadr frame --alpha пишет PNG с прозрачностью', r.returncode == 0 and one.shape == (180, 320, 4) and one[5, 5, 3] == 0, r.stderr.strip()[-100:])
for ext, codec, tol in (('.mov', None, 3.0), ('.webm', 'libvpx-vp9', 25.0)):
    out = os.path.join(D, 'f' + ext)
    r = run('render', Y, '--gpu', '--alpha', '-o', out)
    raw = subprocess.run(['ffmpeg', '-v', 'error', *(['-c:v', codec] if codec else []), '-i', out, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 180, 320, 4)
    got = fr[20] if len(fr) > 20 else np.zeros((180, 320, 4), np.uint8)
    d = np.abs(over(got) - over(want))
    ok(f'{ext}: все кадры, прозрачность на месте', r.returncode == 0 and len(fr) == 30 and np.percentile(d, 99.5) <= tol and got[5, 5, 3] == 0 and got[84:100, 120:200, 3].max() >= 250,
       f'кадров {len(fr)} из 30; кадр 20 на фоне отличается от своего оригинала не больше чем на {np.percentile(d, 99.5):.1f} (99,5 % пикселей), альфа в пустом месте {got[5, 5, 3]}')
r = run('render', Y, '--gpu', '--alpha', '-o', os.path.join(D, 'seq', '%04d.png'))
files = sorted(glob.glob(os.path.join(D, 'seq', '*.png')))
ok('папка кадров PNG: по файлу на кадр, без потерь', r.returncode == 0 and len(files) == 30 and np.array_equal(np.asarray(Image.open(files[20])), want), f'файлов {len(files)}')
# the straight colour and 8 bits are made on the GPU (straight8.wgsl): the same bytes as the numpy arithmetic it replaced,
# give or take 1 where a division lands on a half (semi-transparent edges: a glow's soft rim is mostly that; the GPU's
# division is within a few ulp, not correctly rounded). 1 of 255 is under ProRes 4444's own quantisation
G = GpuRenderer(Film(Y))
G.f.background = None
diffs = []
for tt in (.3, .6, .8, 1.1):
    a = G.frame_comp(tt).download()
    rgb, al = a[..., :3], a[..., 3:]
    old = np.concatenate([np.clip(np.round(np.where(al > 1e-5, rgb / np.maximum(al, 1e-5), 0)), 0, 255), np.clip(np.round(al * 255), 0, 255)], -1)
    diffs.append(np.abs(old - G.frame_rgba_start(tt).get().astype(float)))
d = np.stack(diffs)
ok('прозрачность на видеокарте = прежний расчёт на процессоре', d.max() <= 1 and (d > 0).mean() < 0.05,
   f'наибольшая разница {d.max():.0f} из 255, отличаются {(d > 0).mean() * 100:.3f} % значений')
r = run('render', Y, '--alpha', '-o', os.path.join(D, 'f.mp4'))
ok('в .mp4 прозрачности не бывает — сказано, какие файлы годятся', r.returncode != 0 and '.mov' in r.stderr, r.stderr.strip()[:110])

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
