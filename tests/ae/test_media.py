"""Known answers for reading footage (kadr/media.py): a clip a phone filmed upright — stored on its side with a note to
turn it — comes out upright, the same picture a player shows. python tests/ae/test_media.py"""
import os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.media import probe, VideoReader
from kadr.doc import Film
from kadr.render import Renderer, to_u8

bad = []
D = tempfile.mkdtemp()


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


# four quarters of different colours, 320 × 180, stored — and the same stored with «turn me a quarter» / «three quarters»
a = np.zeros((180, 320, 3), np.uint8)
a[:90, :160], a[:90, 160:], a[90:, :160], a[90:, 160:] = (220, 30, 30), (30, 200, 30), (30, 30, 220), (220, 220, 30)
flat = os.path.join(D, 'flat.mp4')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '320x180', '-r', '25', '-i', '-', '-c:v', 'libx264', '-crf', '10', '-pix_fmt', 'yuv420p', flat],
               input=a.tobytes() * 10, check=True)
p0 = probe(flat)
r = VideoReader(flat)
f0 = r.at(0).copy()
r.close()
name = lambda c: {(1, 0, 0): 'красный', (0, 1, 0): 'зелёный', (0, 0, 1): 'синий', (1, 1, 0): 'жёлтый'}.get(tuple(int(v > 127) for v in c), '?')
quart = lambda f: [name(f[f.shape[0] // 4, f.shape[1] // 4]), name(f[f.shape[0] // 4, 3 * f.shape[1] // 4]), name(f[3 * f.shape[0] // 4, f.shape[1] // 4]), name(f[3 * f.shape[0] // 4, 3 * f.shape[1] // 4])]
ok('обычный клип: размер и четверти на месте', (p0['w'], p0['h']) == (320, 180) and quart(f0) == ['красный', 'зелёный', 'синий', 'жёлтый'], f"{p0['w']}×{p0['h']}, {quart(f0)}")
for turn in (90, 270, 180):
    side = os.path.join(D, f'turn{turn}.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-display_rotation:v:0', str(turn), '-i', flat, '-c', 'copy', side], check=True)
    want = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', side, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout, np.uint8)
    p = probe(side)
    r = VideoReader(side)
    f = r.at(0).copy()
    r.close()
    upright = (p['w'], p['h']) == ((180, 320) if turn != 180 else (320, 180))
    same = f.shape == (p['h'], p['w'], 3) and np.array_equal(f.ravel(), want)
    turned = any(np.rot90(f0, k).shape == f.shape and np.abs(f.astype(int) - np.rot90(f0, k)).mean() < 6 for k in (1, 2, 3))
    ok(f'клип с пометкой «повернуть на {turn}°» читается повёрнутым', upright and same and turned, f"{p['w']}×{p['h']}, четверти {quart(f)}")
    if turn == 90:
        film = Film(data={'film': {'size': [180, 320], 'fps': 25, 'duration': .4, 'background': '#000000'}, 'layers': [dict(id='v', type='video', src=side, position=[90, 160])]})
        g = to_u8(Renderer(film).frame(0))
        ok('…и в фильме стоит так же', quart(g) == quart(f), str(quart(g)))

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
