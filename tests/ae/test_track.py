"""Known answers for tracking and stabilisation (AE 46, kadr/track.py): a clip is made here with a camera whose every
frame is known (a pan, a shake, a turn, a zoom) and a card swinging in perspective in front of it; what the tracker and
the stabiliser find is compared with what was put in. python tests/ae/test_track.py"""
import math, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from PIL import Image
from kadr.doc import Film, DocError
from kadr.render import Renderer, to_u8, homography
from kadr.gpu import GpuRenderer
from kadr import track as tk

bad = []
D = tempfile.mkdtemp()
W, H, N, FPS = 640, 360, 60, 25
rng = np.random.default_rng(11)


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def texture(w, h, lo=40, hi=250):
    a = np.zeros((h, w), np.float32)
    for k in (3, 7, 16, 38, 90):
        n = rng.standard_normal((h // k + 2, w // k + 2)).astype(np.float32)
        a += np.asarray(Image.fromarray(n).resize((w, h), Image.BICUBIC)) * k ** .6
    return (lo + (a - a.min()) / (a.max() - a.min()) * (hi - lo)).astype(np.float32)


def warp(img, M, size, fill=-1.0):
    """the picture put through M (its px → the output's px, pixel middles at whole numbers); fill where it has nothing"""
    Mi = np.linalg.inv(M)
    S = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1.0]])
    Q = S @ Mi @ np.linalg.inv(S)
    Q /= Q[2, 2]
    return np.asarray(Image.fromarray(img).transform(size, Image.PERSPECTIVE, tuple(Q.ravel()[:8]), Image.BICUBIC, fillcolor=fill), np.float32)


def pts(M, q):
    P = M @ np.c_[np.asarray(q, float), np.ones(len(q))].T
    return (P[:2] / P[2]).T


# the scene: a wall larger than the frame, a card in front of it
BW, BH = 980, 620
WALL, CARD = texture(BW, BH), texture(180, 120, 20, 255)
CQ = np.array([[0, 0], [179, 0], [179, 119], [0, 119]], float)


def camera(k):                                                 # the wall's px → frame k's px: a pan with a hand's shake
    t = k / FPS
    sh = (5 * math.sin(2 * math.pi * 3.1 * t) + 3 * math.sin(2 * math.pi * 7.3 * t + 1), 4 * math.sin(2 * math.pi * 4.3 * t + 2) + 2.5 * math.sin(2 * math.pi * 8.9 * t))
    th = math.radians(1.1 * math.sin(2 * math.pi * 2.3 * t) + .6 * math.sin(2 * math.pi * 6.1 * t + .5) + 1.5 * t)
    s = 1 + .02 * math.sin(2 * math.pi * 1.7 * t) + .03 * t
    cx, cy = BW / 2 + 40 * t, BH / 2 + 12 * t                  # the wall point in the middle of the frame
    a, b = s * math.cos(th), s * math.sin(th)
    return np.array([[a, -b, W / 2 + sh[0] - a * cx + b * cy], [b, a, H / 2 + sh[1] - b * cx - a * cy], [0, 0, 1.0]])


def card(k):                                                   # the card's px → the wall's px: it swings and drifts
    t = k / FPS
    base = np.array([[430, 250], [610, 258], [604, 372], [426, 366]], float) + [18 * t, -6 * t]
    swing = np.array([[10, -6], [-8, 9], [6, 8], [-9, -7]]) * math.sin(2 * math.pi * .45 * t)
    return homography(CQ, base + swing)


FR = []
for k in range(N):
    C, Hk = camera(k), camera(k) @ card(k)
    f = warp(WALL, C, (W, H))
    c = warp(CARD, Hk, (W, H))
    f = np.where(c >= 0, c, f)
    if 30 <= k <= 36:                                          # a hand passes over the card's left half
        q = pts(Hk, [[0, 0], [80, 0], [80, 119], [0, 119]])
        m = warp(np.full((120, 81), 128, np.float32), homography([[0, 0], [80, 0], [80, 119], [0, 119]], q), (W, H))
        f = np.where(m >= 0, m, f)
    if 44 <= k <= 47:                                          # and something covers it whole
        m = warp(texture(180, 120, 20, 255), Hk, (W, H))
        f = np.where(m >= 0, m, f)
    FR.append(np.clip(f, 0, 255).astype(np.uint8))
CLIP = os.path.join(D, 'shot.mp4')
raw = b''.join(np.repeat(f[..., None], 3, 2).tobytes() for f in FR)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '12',
                '-pix_fmt', 'yuv420p', CLIP], input=raw, check=True)

# --- the matcher itself
tpl = tk.Template(FR[0].astype(np.float32), [[30, 40], [240, 40], [240, 320], [30, 320]])      # a piece of the wall, away from the card
Wt = camera(1) @ np.linalg.inv(camera(0))
Wf, q, seen, _ = tk.register(tpl, tk.pyramid(FR[1].astype(np.float32), 5), np.eye(3), 'similar')
e = np.hypot(*(pts(Wf, tpl.quad) - pts(Wt, tpl.quad)).T).max()
ok('два соседних кадра совмещаются до сотых пикселя', e < .08 and q > .98, f'наибольшая ошибка углов {e:.3f} px, сходство {q:.3f}')

# --- a plane through the clip, with a hand over it and a full cover
R0 = 8
rep = tk.track_clip(CLIP, 'card', R0 / FPS, pts(camera(R0) @ card(R0), CQ), 'plane')
tr = tk._load(tk.sidecar(CLIP, 'track'))['tracks']['card']
err, lost = [], np.array(tr['lost'])
for k in range(N):
    Hf = np.append(np.array(tr['H'][k]), 1).reshape(3, 3)
    true = camera(k) @ card(k) @ np.linalg.inv(camera(R0) @ card(R0))
    err.append(float(np.hypot(*(pts(Hf, tr['quad']) - pts(true, tr['quad'])).T).max()))
err = np.array(err)
free = np.array([not (30 <= k <= 36 or 44 <= k <= 47) for k in range(N)])
ok('плоскость (карточка в перспективе на трясущейся камере) — до долей пикселя', err[free].max() < .5 and not lost[free].any(), f'ошибка углов на открытых кадрах: средняя {err[free].mean():.3f}, наибольшая {err[free].max():.3f} px')
ok('рука закрыла половину — трек держится', err[30:37].max() < 1.5 and not lost[30:37].any(), f'наибольшая ошибка под рукой {err[30:37].max():.3f} px')
ok('закрыта целиком — трек помечен потерянным и найден снова', lost[44:48].all() and not lost[48:].any() and err[48:].max() < .5, f'потерян на кадрах {np.nonzero(lost)[0].tolist()}, после возвращения ошибка {err[48:].max():.3f} px')

# --- a piece of the wall: place, size and turn; and place alone
quad = [[90, 60], [250, 60], [250, 180], [90, 180]]
tk.track_clip(CLIP, 'wall', 0, quad, 'similar', t_to=1.2)
tk.track_clip(CLIP, 'dot', 0, [[300, 250], [360, 250], [360, 310], [300, 310]], 'point', t_to=.6)
T = tk._load(tk.sidecar(CLIP, 'track'))['tracks']
e1 = max(float(np.hypot(*(pts(np.append(np.array(T['wall']['H'][k]), 1).reshape(3, 3), quad) - pts(camera(k) @ np.linalg.inv(camera(0)), quad)).T).max()) for k in range(len(T['wall']['H'])))
e2 = max(float(np.hypot(*(pts(np.append(np.array(T['dot']['H'][k]), 1).reshape(3, 3), [[330, 280]]) - pts(camera(k) @ np.linalg.inv(camera(0)), [[330, 280]])).T).max()) for k in range(len(T['dot']['H'])))
ok('кусок стены: место, размер, поворот', e1 < .4, f'наибольшая ошибка углов за {len(T["wall"]["H"])} кадров {e1:.3f} px')
ok('точка: только место', e2 < .6, f'наибольшая ошибка середины за {len(T["dot"]["H"])} кадров {e2:.3f} px')

# --- the camera's path
rep = tk.stabilize_clip(CLIP)
S = tk._load(tk.sidecar(CLIP, 'stab'))
cor = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
es = []
for k in range(N):
    a, b, tx, ty = S['C'][k]
    es.append(float(np.hypot(*(pts(np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]]), cor) - pts(camera(k) @ np.linalg.inv(camera(0)), cor)).T).max()))
ok('путь камеры найден (карточка и рука перед стеной не мешают)', max(es) < .6, f'ошибка в углах кадра: средняя {np.mean(es):.3f}, наибольшая {max(es):.3f} px за {N} кадров; камера ушла на {rep["shift"]:.0f} px')

rep_r = tk.stabilize_clip(CLIP, region=[[20, 20], [240, 20], [240, 340], [20, 340]])       # by the wall at the left alone
Sr = tk._load(tk.sidecar(CLIP, 'stab'))
er = []
for k in range(N):
    a, b, tx, ty = Sr['C'][k]
    er.append(float(np.hypot(*(pts(np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.0]]), cor) - pts(camera(k) @ np.linalg.inv(camera(0)), cor)).T).max()))
ok('путь камеры по названной части кадра (--region)', max(er) < 1.2, f'ошибка в углах кадра: средняя {np.mean(er):.3f}, наибольшая {max(er):.3f} px (судим по трети кадра — углы дальше от неё)')
rep = tk.stabilize_clip(CLIP)                                  # back to the whole frame for what follows


def film(layers, size=(960, 540)):
    return Film(data={'film': {'size': list(size), 'fps': FPS, 'duration': N / FPS, 'background': '#000000'}, 'layers': layers})


def gray(R, t):
    a = to_u8(R.frame(t)) if type(R) is Renderer else R.frame_u8(t)
    return a.astype(np.float32).mean(2)


def drift(frames_, model='similar'):
    """how far the picture's middle moves from the first of these frames to each next one, px"""
    h, w = frames_[0].shape
    t = tk.Template(frames_[0], [[w * .05, h * .15], [w * .36, h * .15], [w * .36, h * .85], [w * .05, h * .85]])   # the wall beside the card
    out, Wc = [], np.eye(3)
    for f in frames_[1:]:
        Wc, q, _, _ = tk.register(t, tk.pyramid(f, 5), Wc, model)
        out.append(pts(Wc, [[w / 2, h / 2]])[0] - [w / 2, h / 2])
    return np.array(out)


# --- the steadied layer: a tripod, then a calm move
V = dict(id='v', type='video', src=CLIP, position=[480, 270])
ks = [0, 6, 13, 21, 29]
f = film([dict(V, stabilize={'lock': True}, out=30 / FPS)])
R = Renderer(f)
fr = [gray(R, k / FPS) for k in ks]
d = np.hypot(*drift(fr).T)
P = tk.plan_of(f.layers[0])
ok('штатив (lock): картинка стоит на месте', d.max() < .5 and min(float(x.min()) for x in fr) > 5 and P['covered'], f'середина кадра уходит не больше чем на {d.max():.3f} px из 960; краёв не видно, увеличение {P["zoom"]:.3f}')
raw_f = film([dict(V)])
st_f = film([dict(V, stabilize={'smooth': 0.6})])
Rr, Rs = Renderer(raw_f), Renderer(st_f)
a, b = drift([gray(Rr, k / FPS) for k in range(0, 40)]), drift([gray(Rs, k / FPS) for k in range(0, 40)])
ja, jb = float(np.hypot(*np.diff(a, 2, 0).T).std()), float(np.hypot(*np.diff(b, 2, 0).T).std())
P = tk.plan_of(st_f.layers[0])
ok('плавный ход (smooth): дрожь уходит, проезд остаётся', jb < ja / 5 and np.hypot(*b[-1]) > 30 and P['covered'], f'рывки от кадра к кадру {ja:.2f} → {jb:.2f} px; за 1,6 с камера всё так же уезжает на {np.hypot(*b[-1]):.0f} px; увеличение {P["zoom"]:.3f}')
g = GpuRenderer(st_f).frame_u8(20 / FPS).astype(int)
ok('стабилизация одинакова на обоих рендерах', np.abs(g - to_u8(Rs.frame(20 / FPS))).max() <= 3, f'наибольшая разница {np.abs(g - to_u8(Rs.frame(20 / FPS))).max()}')

# --- in a document: a marker on the card, a picture pinned on it, hidden while it is lost
for name, vid in (('на исходном клипе', dict(V)), ('на стабилизированном клипе', dict(V, stabilize={'smooth': 0.6}))):
    f = film([vid,
              dict(id='pin', type='solid', color='#00ff00', size=[200, 100], corner_pin="=track('v', 'card').corners", opacity="=1 - track('v', 'card').lost"),
              dict(id='dot', type='solid', color='#ff0000', size=[14, 14], position="=track('v', 'card').position")])
    R = Renderer(f)
    worst, iou = 0.0, 1.0
    for k in (2, 20, 40, 55):
        img = to_u8(R.frame(k / FPS)).astype(int)
        red = (img[..., 0] > 200) & (img[..., 1] < 80) & (img[..., 2] < 80)
        green = ((img[..., 1] > 200) & (img[..., 0] < 80) & (img[..., 2] < 80)) | red
        ys, xs = np.nonzero(red)
        Hk = camera(k) @ card(k)
        if vid.get('stabilize'):
            Hk = tk.plan_of(f.layers[0])['A'][k] @ Hk
        mid = pts(np.linalg.inv(camera(R0) @ card(R0)), [pts(camera(R0) @ card(R0), CQ).mean(0)])   # the card's point under the region's middle
        true = (pts(Hk, mid)[0] + .5) * 1.5                     # the clip is shown 1.5 times its size
        worst = max(worst, float(np.hypot(xs.mean() + .5 - true[0], ys.mean() + .5 - true[1])))
        yy, xx = np.mgrid[0:540, 0:960]
        inside = tk._in_quad(xx + .5, yy + .5, (pts(Hk, CQ) + .5) * 1.5)
        iou = min(iou, float((inside & green).sum() / (inside | green).sum()))
    ok(f'метка и прибитая картинка сидят на карточке ({name})', worst < 1.0 and iou > .97, f'метка мимо середины не больше чем на {worst:.2f} px из 960; совпадение площади {iou * 100:.1f} %')
img = to_u8(R.frame(45 / FPS)).astype(int)
ok('пока карточка закрыта, прибитая картинка скрыта', not ((img[..., 1] > 200) & (img[..., 0] < 80) & (img[..., 2] < 80)).any())
f = film([dict(V), dict(id='t', type='solid', color='#ffffff', size=[40, 40], position="=track('v', 'card').position", scale="=track('v', 'card').scale", rotation="=track('v', 'card').rotation")])
L = f.layers[1]
s55, r55 = L['scale'].at(55 / FPS), L['rotation'].at(55 / FPS)
q0, q1 = pts(camera(R0) @ card(R0), CQ), pts(camera(55) @ card(55), CQ)
ts = math.sqrt(tk._area(q1) / tk._area(q0))
trn = math.degrees(math.atan2(*(q1[1] - q1[0])[::-1]) - math.atan2(*(q0[1] - q0[0])[::-1]))
ok('размер и поворот трека', abs(s55 - ts) < .01 and abs(r55 - trn) < .3, f'размер {s55:.3f} (на деле {ts:.3f}), поворот {r55:.2f}° (на деле {trn:.2f}°)')
for what, layers in (('опечатка в имени трека названа', [dict(V), dict(id='d', type='solid', color='#fff', size=[4, 4], position="=track('v', 'crad').position")]),
                     ('трек не от видео-слоя назван', [dict(V), dict(id='d', type='solid', color='#fff', size=[4, 4], position="=track('d', 'card').position")]),
                     ('опечатка в свойстве трека названа', [dict(V), dict(id='d', type='solid', color='#fff', size=[4, 4], position="=track('v', 'card').positon")]),
                     ('stabilize без пути камеры назван', [dict(V, src=os.path.join(HERE, 'clip.mp4'), stabilize=True)])):
    try:
        film(layers)
        ok(what, False, 'документ принят')
    except DocError as e:
        ok(what, True, str(e).splitlines()[0][:130])
    except Exception as e:
        ok(what, False, f'падение: {type(e).__name__}: {e}')

# --- where a clip jumps from one shot to another: three shots of three walls, the camera panning in each; a flash; a fade
walls = [texture(BW, BH), texture(BW, BH, 10, 200), texture(BW, BH, 60, 255)]
shots = []
for k in range(75):
    n = 0 if k < 22 else 1 if k < 51 else 2
    f_ = warp(walls[n], camera(k), (W, H))
    if n == 1 and 30 <= k < 34:                                # the shot brightens and dims again — not a cut
        f_ = f_ * (1 + .25 * math.sin((k - 30) / 4 * math.pi))
    shots.append(np.clip(f_, 0, 255).astype(np.uint8))
CUTS = os.path.join(D, 'cuts.mp4')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', CUTS],
               input=b''.join(f_.tobytes() for f_ in shots), check=True)
cuts, fps_, n_ = tk.find_cuts(CUTS)
ok('склейки в клипе: два стыка трёх планов, проезд и вспышка — не склейки', n_ == 75 and [round(c * FPS) for c in cuts] == [22, 51], f'найдены на кадрах {[round(c * FPS) for c in cuts]} (ждём 22 и 51)')
one, _, _ = tk.find_cuts(CLIP)
ok('в клипе одним планом склеек нет', one == [], f'найдено {len(one)}')

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
