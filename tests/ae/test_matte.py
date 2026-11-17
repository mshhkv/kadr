"""Known answers for cutting things out (AE 47, kadr/matte.py): the `cutout` property against a matte made here (a disc
whose every position is known), and the matte the Mac's own subject finder makes of a thing drawn by Kadr against the
thing's true outline. python tests/ae/test_matte.py"""
import math, os, shutil, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import yaml
from kadr.doc import Film, DocError
from kadr.render import Renderer, to_u8
from kadr.gpu import GpuRenderer
from kadr import matte as mt

bad = []
D = tempfile.mkdtemp()
W, H, N, FPS, RAD = 320, 180, 20, 25, 30


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def clip(name, frames_, gray=False):
    p = os.path.join(D, name)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray' if gray else 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                    '-c:v', 'libx264', '-crf', '8', '-pix_fmt', 'gray' if gray else 'yuv420p', p], input=b''.join(f.tobytes() for f in frames_), check=True)
    return p


# a white clip, and its matte: a disc that moves 10 px a frame
yy, xx = np.mgrid[0:H, 0:W]
disc = lambda k: (np.clip(RAD + .5 - np.hypot(xx - (60 + 10 * k), yy - 90), 0, 1) * 255).astype(np.uint8)
CLIP = clip('shot.mp4', [np.full((H, W, 3), 255, np.uint8)] * N)
clip('shot.matte.mp4', [disc(k) for k in range(N)], gray=True)


def film(layers, size=(W, H)):
    return Film(data={'film': {'size': list(size), 'fps': FPS, 'duration': N / FPS, 'background': '#0000ff'}, 'layers': layers})


def white(R, k):
    a = to_u8(R.frame(k / FPS)) if type(R) is Renderer else R.frame_u8(k / FPS)
    a = a[..., 0].astype(np.float32) / 255                    # white over blue: the red channel is the cut-out's alpha
    return np.where(a < .02, 0, a)                             # (a compressed matte's black is 0 or 1 of 255: over a whole frame that pulls a centroid)


V = dict(id='v', type='video', src=CLIP, position=[W / 2, H / 2])
f = film([dict(V, cutout=True)])
R, G = Renderer(f), GpuRenderer(f)
worst, area = 0.0, []
for k in (0, 7, 15):
    a = white(R, k)
    worst = max(worst, abs(float((a * xx).sum() / a.sum()) - (60 + 10 * k)), abs(float((a * yy).sum() / a.sum()) - 90))
    area.append(float(a.sum()))
ok('вырезка — кадр в кадр там, где её мат', worst < .5 and all(abs(s - math.pi * RAD ** 2) < .03 * math.pi * RAD ** 2 for s in area),
   f'середина круга мимо не больше чем на {worst:.2f} px на кадрах 0, 7, 15; площадь {area[0]:.0f} (круг радиуса 30 — {math.pi * RAD ** 2:.0f})')
ok('вырезка одинакова на обоих рендерах', np.abs(white(R, 7) - white(G, 7)).max() < .02, f'наибольшая разница {np.abs(white(R, 7) - white(G, 7)).max() * 255:.1f} из 255')
a = white(Renderer(film([dict(V, cutout={'invert': True})])), 7)
ok('invert — всё, кроме вырезанного', abs(float((1 - a).sum()) - math.pi * RAD ** 2) < .03 * math.pi * RAD ** 2 and a[5, 5] > .98, f'дыра площадью {float((1 - a).sum()):.0f}')
hard, soft = white(R, 7)[90], white(Renderer(film([dict(V, cutout={'feather': 8})])), 7)[90]
wd = lambda row: int(((row > .1) & (row < .9)).sum())
ok('feather смягчает край', wd(soft) > wd(hard) + 6 and abs(float(soft.sum()) - float(hard.sum())) < 4, f'переход по краю {wd(hard)} → {wd(soft)} px, ширина та же')
f = film([dict(V, cutout=True, size='native', scale=.5, position=[400, 250], **{'in': .2})], size=(640, 360))
a = to_u8(Renderer(f).frame(.2 + 7 / FPS))[..., 0].astype(np.float32) / 255
a = np.where(a < .02, 0, a)
gy, gx = np.mgrid[0:360, 0:640]
cx, cy = float((a * gx).sum() / a.sum()), float((a * gy).sum() / a.sum())
ok('вырезка идёт за слоем: сдвиг, масштаб, начало позже', abs(cx - (400 + (130 - 160) * .5)) < .6 and abs(cy - 250) < .6 and abs(a.sum() - math.pi * 15 ** 2) < 40,
   f'середина ({cx:.1f}, {cy:.1f}), ждём (385, 250); площадь {a.sum():.0f}, ждём {math.pi * 225:.0f}')
for what, layers in (('cutout без вырезки назван', [dict(V, src=os.path.join(HERE, 'clip.mp4'), cutout=True)]),
                     ('cutout не у видео назван', [dict(id='t', type='text', text='а', font='Arial', size=20, cutout=True)]),
                     ('опечатка в cutout названа', [dict(V, cutout={'fether': 2})])):
    try:
        film(layers)
        ok(what, False, 'документ принят')
    except DocError as e:
        ok(what, True, str(e).splitlines()[0][:120])
    except Exception as e:
        ok(what, False, f'падение: {type(e).__name__}: {e}')

# --- another system's matte program: whatever speaks the language answers (Windows and Linux have no Vision; the
# stand-in here calls «the thing» what is brighter than middle grey — the plumbing is what is tested)
EX = os.path.join(HERE, 'matte_tool_example.py')
os.environ['KADR_MATTE_TOOL'] = EX
try:
    yy, xx = np.mgrid[0:H, 0:W]
    disc = lambda cx, cy: np.where(((xx - cx) ** 2 + (yy - cy) ** 2 <= RAD ** 2)[..., None], 255, 20).astype(np.uint8).repeat(3, 2)
    cx_of = lambda i: 60 + 8 * i
    C = clip('own.mp4', [disc(cx_of(i), 90) for i in range(N)])
    ok('своя программа вырезания запускается вместо системной', mt.tool()[-1] == EX and len(mt.tool()) == 2, ' '.join(os.path.basename(v) for v in mt.tool()))
    rep = mt.run(C, 'subject')
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', rep['out'], '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], capture_output=True).stdout
    M = np.frombuffer(raw, np.uint8).reshape(-1, H, W) > 127
    true = lambda i: (xx - cx_of(i)) ** 2 + (yy - 90) ** 2 <= RAD ** 2
    iou = min(float((true(i) & M[i]).sum() / (true(i) | M[i]).sum()) for i in (0, 9, 19))
    ok('клип через неё: вырезка каждого кадра на своём месте', len(M) == N and rep['frames'] == N and rep['empty'] == 0 and iou > .9 and abs(rep['cover'] - math.pi * RAD ** 2 / (W * H)) < .01,
       f'кадров {len(M)}, совпадение площади не меньше {iou * 100:.1f} %, доля кадра {rep["cover"] * 100:.1f} %')
    rep = mt.run(C, 'subject', point=(5, 5), out=os.path.join(D, 'own_none.mp4'))
    ok('под точкой ничего нет — файл не пишется', rep['out'] is None and rep['empty'] == N and not os.path.exists(os.path.join(D, 'own_none.mp4')), f'пустых кадров {rep["empty"]} из {rep["frames"]}')
    from PIL import Image
    P = os.path.join(D, 'own.png')
    Image.fromarray(disc(200, 70)).save(P)
    rep = mt.run(P, 'subject')
    got = np.asarray(Image.open(rep['out']).convert('L')) > 127
    tr = (xx - 200) ** 2 + (yy - 70) ** 2 <= RAD ** 2
    ok('фотография через неё', rep['out'].endswith('own.matte.png') and float((tr & got).sum() / (tr | got).sum()) > .97, f'совпадение площади {float((tr & got).sum() / (tr | got).sum()) * 100:.1f} %')
    j = mt.pose(P)
    ok('суставы приходят в пикселях самой фотографии', set(j) == {'nose'} and abs(j['nose'][0] - 200) < 1.5 and abs(j['nose'][1] - 70) < 1.5, str({k: tuple(round(float(v), 1) for v in q) for k, q in j.items()}))
    from kadr import reframe as rf
    rep = rf.run(C, 'subject')
    rows = __import__('json').load(open(rep['out'], encoding='utf-8'))['rows']
    ok('кадрирование под вертикаль через неё: предмет найден там, где он есть', rep['seen'] == rep['samples'] == len(rows) and abs(rows[0][0] - cx_of(0)) < 3 and abs(rows[-1][0] - cx_of((len(rows) - 1) * rf.STEP)) < 3,
       f'замеров {rep["samples"]}, середина по x: {rows[0][0]:.0f} → {rows[-1][0]:.0f} (на самом деле {cx_of(0)} → {cx_of((len(rows) - 1) * rf.STEP)})')
    os.environ['KADR_MATTE_TOOL'] = os.path.join(D, 'nope.py')
    try:
        mt.tool()
        ok('названной программы нет — сказано', False)
    except RuntimeError as e:
        ok('названной программы нет — сказано', 'KADR_MATTE_TOOL' in str(e), str(e)[:60])
finally:
    del os.environ['KADR_MATTE_TOOL']

# a system that is not macOS: its own program is kadr/matte_tool.py, and until it is written Kadr says what to do
real, own = sys.platform, mt.OWN
try:
    sys.platform = 'win32'
    mt.OWN = os.path.join(D, 'matte_tool.py')
    try:
        mt.tool()
        ok('не macOS и программы ещё нет — сказано, что написать и где описание', False)
    except RuntimeError as e:
        ok('не macOS и программы ещё нет — сказано, что написать и где описание', all(w in str(e) for w in ('matte_tool.py', 'AGENTS.md', 'matte_tool_example.py')), str(e)[:70])
    shutil.copy(EX, mt.OWN)
    ok('не macOS, программа написана — работает она', mt.tool()[-1] == mt.OWN and mt.run(P, 'subject', out=os.path.join(D, 'own2.png'))['out'] is not None)
finally:
    sys.platform, mt.OWN = real, own

# --- the Mac's own subject finder on a thing drawn by Kadr, against the thing's true outline
try:
    mt.tool()
    have = True
except RuntimeError as e:
    have = False
    print('     пропущено (системная программа вырезания есть только на macOS с инструментами Xcode):', str(e).splitlines()[0][:100])
if have:
    S = os.path.join(D, 'subject.mp4')
    shutil.copy(os.path.join(HERE, 'subject.mp4'), S)
    rep = mt.run(S, 'subject')
    doc = yaml.safe_load(open(os.path.join(HERE, 'make_subject.yaml'), encoding='utf-8'))
    thing = [L for L in doc['layers'] if L['id'] == 'thing']

    def alone(bg):
        return Renderer(Film(data={'film': dict(doc['film'], background=bg), 'layers': thing}))
    Rb, Rw = alone('#000000'), alone('#ffffff')
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', rep['out'], '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], capture_output=True).stdout
    M = np.frombuffer(raw, np.uint8).reshape(-1, 360, 640)
    ious, edge = [], []
    for k in (5, 37, 70):
        true = 1 - (to_u8(Rw.frame(k / 25)).astype(np.float32) - to_u8(Rb.frame(k / 25)).astype(np.float32)).mean(2) / 255   # its alpha: over white less over black
        got = M[k] > 127
        ious.append(float(((true > .5) & got).sum() / ((true > .5) | got).sum()))
        edge.append(float((np.abs((true > .5).astype(int) - got.astype(int)).sum()) / max(1, np.abs(np.diff((true > .5).astype(int), axis=1)).sum() + np.abs(np.diff((true > .5).astype(int), axis=0)).sum())))
    ok('предмет вырезан по своему настоящему контуру', min(ious) > .93 and rep['frames'] == 75 and rep['empty'] == 0 and len(M) == 75,
       f'совпадение площади на кадрах 5, 37, 70: {", ".join(f"{v * 100:.1f} %" for v in ious)}; край ошибается в среднем на {np.mean(edge):.1f} px; кадров 75 из 75')
    rep2 = mt.run(S, 'person', out=os.path.join(D, 'none.mp4'))
    ok('людей в этом клипе нет — и файл не пишется', rep2['out'] is None and rep2['empty'] == 75 and not os.path.exists(os.path.join(D, 'none.mp4')), mt.describe(rep2)[:90])
    P = os.path.join(D, 'still.png')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', S, '-vf', "select='eq(n,37)'", '-frames:v', '1', P], check=True)
    r3 = mt.run(P, 'subject')
    from PIL import Image
    got = np.asarray(Image.open(r3['out']).convert('L')) > 127
    true = 1 - (to_u8(Rw.frame(37 / 25)).astype(np.float32) - to_u8(Rb.frame(37 / 25)).astype(np.float32)).mean(2) / 255 > .5
    ok('и из фотографии — так же', r3['out'].endswith('still.matte.png') and float((true & got).sum() / (true | got).sum()) > .93, f'совпадение площади {(true & got).sum() / (true | got).sum() * 100:.1f} %')
    f = Film(os.path.join(HERE, '47_cutout.yaml'))
    a, b = to_u8(Renderer(f).frame(1.0)).astype(int), GpuRenderer(f).frame_u8(1.0).astype(int)
    ok('демо со словом за предметом одинаково на обоих рендерах', (np.abs(a - b) > 3).mean() < .005, f'расходится {(np.abs(a - b) > 3).mean() * 100:.2f} % пикселей')

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
