"""Known answers for keeping the subject in a narrower frame (kadr/reframe.py): a disc whose every position is known,
its place written beside the clip by hand — the layer must slide so that the disc stands in the middle of a vertical
frame, stop where the footage ends, and jump with a cut; then the Mac's own subject finder on a thing drawn by Kadr,
against where the thing truly is. python tests/ae/test_reframe.py"""
import json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr.render import Renderer, to_u8
from kadr.gpu import GpuRenderer
from kadr import reframe as rf

bad = []
D = tempfile.mkdtemp()
W, H, N, FPS, RAD = 320, 180, 40, 25, 20
FW = 102                                                       # a vertical frame, 9:16 of the clip's height


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


yy, xx = np.mgrid[0:H, 0:W]
X = lambda k: 30 + 6.5 * k if k < 20 else 290 - 4 * (k - 20)    # the disc goes right, then (a cut) is far right and goes left


def frame_(k):
    a = np.zeros((H, W, 3), np.uint8)
    a[..., 2] = 255 if k < 20 else 120                         # the two shots differ, so the cut is seen
    a[..., 1] = 0 if k < 20 else 90
    d = np.clip(RAD + .5 - np.hypot(xx - X(k), yy - 90), 0, 1)[..., None]
    return (a * (1 - d) + np.array([255, 255, 255]) * d).astype(np.uint8)


CLIP = os.path.join(D, 'walk.mp4')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '8',
                '-pix_fmt', 'yuv420p', CLIP], input=b''.join(frame_(k).tobytes() for k in range(N)), check=True)


def side(cuts, step=1, miss=()):
    rows = [[X(k), 90, 0, 0, 0, 0, .1, -1, -1, -1, -1] if k not in miss else [-1] * 11 for k in range(0, N, step)]
    json.dump(dict(clip=dict(w=W, h=H, fps=FPS, frames=N), what='subject', step=step, cuts=cuts, rows=rows), open(os.path.join(D, 'walk.frame.json'), 'w'))
    os.utime(os.path.join(D, 'walk.frame.json'), (0, np.random.randint(1, 10 ** 9)))    # a new file to the cache
    rf._paths.clear()


def film(layer):
    return Film(data={'film': {'size': [FW, H], 'fps': FPS, 'duration': N / FPS, 'background': '#000000'}, 'layers': [layer]})


def disc_x(R, k):
    a = to_u8(R.frame(k / FPS)) if type(R) is Renderer else R.frame_u8(k / FPS)
    m = (a[..., :3].astype(int).min(-1) > 200).astype(float)   # the white disc
    return float((m * np.arange(a.shape[1])).sum() / max(1, m.sum())), a    # in pixel numbers, as X(k) is


V = dict(id='v', type='video', src=CLIP, position=[FW / 2, H / 2])
try:
    film(dict(V, reframe=True))
    ok('без файла рядом с клипом — понятная ошибка', False)
except DocError as e:
    ok('без файла рядом с клипом — понятная ошибка', 'kadr reframe' in str(e), str(e).splitlines()[0][:110])

side([20 / FPS])
f = film(dict(V, reframe={'smooth': 0}))
R, G = Renderer(f), GpuRenderer(f)
# the window is FW wide in a clip W wide: the disc can stand in the middle while its centre is FW/2 … W − FW/2 from the left
mid = [k for k in range(N) if FW / 2 + 2 <= X(k) <= W - FW / 2 - 2]
MID = FW / 2 - .5                                              # the frame's middle in pixel numbers
worst = max(abs(disc_x(R, k)[0] - MID) for k in mid)
ok('предмет стоит посередине вертикального кадра', worst < .6, f'на {len(mid)} кадрах из {N} середина круга мимо середины кадра не больше чем на {worst:.2f} px')
x0, a0 = disc_x(R, 0)
ok('у края съёмки окно останавливается: пустоты нет', abs(x0 - X(0)) < .6 and a0[:, 0, 2].min() > 200 and a0[:, -1, 2].min() > 200,
   f'кадр 0: круг на {x0:.1f} px (в клипе он на {X(0):.1f}), оба края кадра закрыты съёмкой')
x20, a20 = disc_x(R, 20)
ok('на склейке окно прыгает вместе с планом', abs(x20 - (FW - (W - X(20)))) < .6 and abs(disc_x(R, 19)[0] - MID) < .6,
   f'кадр 19: {disc_x(R, 19)[0]:.1f} (середина {MID:.1f}); кадр 20, круг в 30 px от правого края съёмки: {x20:.1f} (ждали {FW - (W - X(20)):.1f})')
d = max(np.abs(to_u8(R.frame(k / FPS)).astype(int) - G.frame_u8(k / FPS).astype(int)).max() for k in (3, 12, 27))
ok('оба рендера кладут кадр одинаково', d <= 2, f'наибольшая разница {d} из 255')

# evened: the window does not shake with every step, and does not travel across the cut
side([20 / FPS])
P = rf.path(CLIP, smooth=.3)
ok('сглаживание не тащит окно через склейку', abs(P[19, 0] - P[20, 0]) > 120 and np.abs(np.diff(P[:20, 0], 2)).max() < 1.5,
   f'между кадрами 19 и 20 окно прыгает на {abs(P[19, 0] - P[20, 0]):.0f} px; внутри плана ускорение не больше {np.abs(np.diff(P[:20, 0], 2)).max():.2f} px')
side([])                                                       # one shot now: the disc runs right at 6.5 px a frame, then 4 px a frame back
free, held = rf.path(CLIP, smooth=.2)[:20, 0], rf.path(CLIP, smooth=.2, lim=((60.0, 100.0), None))[:20, 0]
ok('к краю съёмки окно подходит плавно, а не встаёт за один кадр', held.max() <= 100 + 1e-6 and held.min() >= 60 - 1e-6 and np.abs(np.diff(held, 2)).max() < 2.2 and np.abs(np.diff(np.clip(free, 60, 100), 2)).max() > 5,
   f'окну можно от 60 до 100 px: оно идёт {held.min():.1f} … {held.max():.1f}, скорость меняется не больше чем на {np.abs(np.diff(held, 2)).max():.2f} px за кадр (обрезка готового пути — {np.abs(np.diff(np.clip(free, 60, 100), 2)).max():.1f})')
side([], step=3, miss=(9, 12))
P = rf.path(CLIP, smooth=0)
want = np.array([X(k) for k in range(18)])
ok('каждый третий кадр и пропуски: между ними — по прямой', np.abs(P[:18, 0] - want).max() < .05, f'наибольшая ошибка на кадрах 0–17: {np.abs(P[:18, 0] - want).max():.3f} px')
# a subject that swings faster than the evening: the shot is followed closer until it stays in the middle part
rows = [[160 + 120 * np.sin(k * .5), 90, 0, 0, 0, 0, .1, -1, -1, -1, -1] for k in range(N)]
json.dump(dict(clip=dict(w=W, h=H, fps=FPS, frames=N), what='subject', step=1, cuts=[], rows=rows), open(os.path.join(D, 'walk.frame.json'), 'w'))
rf._paths.clear()
far = lambda keep: float(np.abs(rf.path(CLIP, smooth=1.0, keep=keep)[:, 0] - np.array(rows)[:, 0]).max())
a, b = far(None), far((30, 90))
ok('быстрый предмет: окно идёт ближе, чем просили, и не выпускает его', a > 60 and b <= 30, f'при сглаживании 1 с предмет уходил бы от середины окна на {a:.0f} px; с пределом 30 px — на {b:.1f}')

# ---- with a cut-out: the matte is slid by the same amount as the footage
side([20 / FPS])
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '8', '-pix_fmt', 'gray',
                os.path.join(D, 'walk.matte.mp4')], input=b''.join((np.clip(RAD + .5 - np.hypot(xx - X(k), yy - 90), 0, 1) * 255).astype(np.uint8).tobytes() for k in range(N)), check=True)
Rc = Renderer(film(dict(V, reframe={'smooth': 0}, cutout=True)))
got = []
for k in (5, 12, 25, 33):
    a = to_u8(Rc.frame(k / FPS)).astype(int)
    m = a[..., :3].min(-1) > 200
    got.append((abs(float((m * np.arange(FW)).sum() / max(1, m.sum())) - MID) if FW / 2 + 2 <= X(k) <= W - FW / 2 - 2 else 0.0, int(m.sum()), int((a[..., 2] > 60).sum() - m.sum())))
ok('с вырезкой: мат едет вместе со съёмкой', all(g[0] < .6 and abs(g[1] - np.pi * RAD ** 2) < .08 * np.pi * RAD ** 2 and g[2] < 200 for g in got),
   'вырезанный круг: ' + ', '.join(f'мимо середины на {g[0]:.2f} px, площадь {g[1]}' for g in got[:2]) + f' (круг радиуса {RAD} — {np.pi * RAD ** 2:.0f}); фона вокруг не видно')

# ---- with the camera's shake taken out: the person is followed in the steadied picture
SHOT = os.path.join(D, 'shot.mp4')
import shutil
shutil.copy(os.path.join(HERE, 'shot.mp4'), SHOT)
shutil.copy(os.path.join(HERE, 'shot.stab.json'), os.path.join(D, 'shot.stab.json'))
from kadr.track import plan_of
from kadr.media import probe
ps = probe(SHOT)
VS = dict(id='v', type='video', src=SHOT, position=[ps['h'] * 9 / 32, ps['h'] / 2], stabilize=True)
fs = lambda extra: Film(data={'film': {'size': [int(ps['h'] * 9 / 16), ps['h']], 'fps': ps['fps'], 'duration': 2, 'background': '#000000'}, 'layers': [dict(VS, **extra)]})
A = np.asarray(plan_of(fs({}).layers[0])['A'])
n_ = int(round(ps['duration'] * ps['fps']))
c0 = np.array([ps['w'] * .55, ps['h'] * .5, 1.0])               # a thing that stands still in the steadied picture: in the footage it shakes with the camera
rows = [list((np.linalg.inv(A[min(k, len(A) - 1)]) @ c0)[:2]) + [0, 0, 0, 0, .1, -1, -1, -1, -1] for k in range(n_)]
json.dump(dict(clip=dict(w=ps['w'], h=ps['h'], fps=ps['fps'], frames=n_), what='subject', step=1, cuts=[], rows=rows), open(os.path.join(D, 'shot.frame.json'), 'w'))
rf._paths.clear()
from kadr.render import place
f2 = fs(dict(reframe={'smooth': .4}))
xs_ = np.array([place(f2, f2.layers[0], k / ps['fps'])[0] for k in range(2, int(1.8 * ps['fps']))])
shake = float(np.ptp(np.array(rows)[:int(1.8 * ps['fps']), 0]))
ok('со стабилизацией: окно стоит, когда человек стоит в выровненной картинке', float(np.ptp(xs_)) < .05 and shake > 3,
   f'в съёмке он ходит на {shake:.1f} px (тряска камеры), слой за 1,8 с сдвигается не больше чем на {float(np.ptp(xs_)):.3f} px')

# the real thing: the Mac finds a thing drawn by Kadr; where it truly is — from the same scene drawn alone
SUB = os.path.join(HERE, 'subject.mp4')
if sys.platform != 'darwin':
    print('SKIP поиск предмета: только на macOS')
else:
    import shutil, yaml
    src = os.path.join(D, 'subject.mp4')
    shutil.copy(SUB, src)
    rep = rf.run(src, 'subject')
    doc = yaml.safe_load(open(os.path.join(HERE, 'make_subject.yaml'), encoding='utf-8'))
    doc['layers'] = [L for L in doc['layers'] if L['id'] == 'thing']
    doc['film']['background'] = '#000000'
    Aw, Ab = Renderer(Film(data=dict(doc, film=dict(doc['film'], background='#ffffff')))), Renderer(Film(data=doc))
    P = rf.path(src, smooth=0)
    errs = []
    for k in (0, 12, 30, 47, 66):
        al = 1 - (to_u8(Aw.frame(k / 25)).astype(np.float32) - to_u8(Ab.frame(k / 25)).astype(np.float32)).mean(2) / 255   # its alpha: over white less over black
        al = np.where(al < .02, 0, al)
        ys, xs = np.mgrid[0:al.shape[0], 0:al.shape[1]]
        errs.append(float(np.hypot((al * xs).sum() / al.sum() - P[k, 0], (al * ys).sum() / al.sum() - P[k, 1])))
    ok('найденный предмет — там, где он на самом деле', rep['seen'] == rep['samples'] and max(errs) < 6,
       f"найден на {rep['seen']} кадрах из {rep['samples']}; мимо настоящей середины не больше чем на {max(errs):.1f} px (кадр 640 × 360)")
    f = Film(data={'film': {'size': [202, 360], 'fps': 25, 'duration': 3, 'background': '#000000'},
                   'layers': [dict(id='v', type='video', src=src, position=[101, 180], reframe=True)]})
    Rr = Renderer(f)
    off = []
    for k in (0, 12, 30, 47, 66):
        a = to_u8(Rr.frame(k / 25)).astype(int)
        m = ((a[..., 0] > 150) & (a[..., 1] < 90) & (a[..., 2] < 90)).astype(float)   # the red ball on top of the stack
        off.append(float((m * np.arange(202)).sum() / max(1, m.sum())) - 101)
    ok('в вертикальном кадре он остаётся в средней части', max(abs(o) for o in off) < 202 * .3, 'красный шар от середины кадра: ' + ', '.join(f'{o:+.0f}' for o in off) + ' px (кадр 202 px)')

print('\nВСЁ СХОДИТСЯ' if not bad else f'\nНЕ СХОДИТСЯ: {bad}')
sys.exit(1 if bad else 0)
