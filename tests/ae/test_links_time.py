"""Known answers for links between layers (layer(), var()) and for time (Time Remap, frame blending, Echo, Posterize
Time). python tests/ae/test_links_time.py"""
import os, sys, tempfile
os.environ['KADR_TRUST'] = tempfile.gettempdir()             # the films this test writes there are its own: they may read the clip beside the test (kadr trust)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import yaml
from kadr.doc import Film, DocError
from kadr.render import Renderer
from kadr.gpu import GpuRenderer

bad = []
CLIP = os.path.join(HERE, 'clip.mp4')


def film(layers, **f):
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'f.yaml')
    yaml.safe_dump({'film': dict({'size': [320, 180], 'fps': 25, 'duration': 2, 'background': '#000000'}, **f), 'layers': layers}, open(p, 'w'), default_flow_style=None)
    return Film(p)


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def ball_x(img):
    """where the yellow ball of clip.mp4 is in a frame (its column of mass), px"""
    m = (img[..., 0] > 180) & (img[..., 1] > 150) & (img[..., 2] < 120) & (np.arange(img.shape[0])[:, None] < 125)
    return float((m.sum(0) * np.arange(img.shape[1])).sum() / max(1, m.sum()))


def src_x(t):                                                  # the ball's place at source time t (frame floor(t·25))
    return 30 + 260 * (np.floor(t * 25 + 1e-4) / 25) / 2


# ================= links =================
f = film([{'id': 'lead', 'type': 'solid', 'color': '#ffffff', 'size': [10, 10], 'position': {0: [20, 90], 2: [300, 90], 'ease': 'linear'}},
          {'id': 'lag', 'type': 'solid', 'color': '#ffffff', 'size': [10, 10], 'position': "=layer('lead').at(t - 0.5).position"},
          {'id': 'twice', 'type': 'solid', 'color': '#ffffff', 'size': [10, 10], 'position': [0, 0], 'rotation': "=layer('lead').x * 2 + var('extra')",
           'opacity': "=var('fade')"}], vars={'extra': 5, 'fade': {0: 0, 2: 1, 'ease': 'linear'}})
L = {l['id']: l for l in f.layers}
ok('layer().at(t − 0.5): отстающий слой там, где ведущий был полсекунды назад', np.allclose(L['lag']['position'].at(1.5), L['lead']['position'].at(1.0)))
ok('layer().x и var(): формула из чужого свойства и переменной фильма', abs(L['twice']['rotation'].at(1.0) - (L['lead']['position'].at(1.0)[0] * 2 + 5)) < 1e-3)
ok('var() с ключами анимируется', abs(L['twice']['opacity'].at(0.5) - 0.25) < 1e-3)
for what, layers in (('слой, которого нет', [{'id': 'a', 'type': 'solid', 'opacity': "=layer('zz').opacity"}]),
                     ('свойство, которого нет', [{'id': 'a', 'type': 'solid'}, {'id': 'b', 'type': 'solid', 'opacity': "=layer('a').nope"}]),
                     ('переменная, которой нет', [{'id': 'a', 'type': 'solid', 'opacity': "=var('zz')"}])):
    try:
        film(layers)
        ok(f'ошибка в ссылке ловится при загрузке ({what})', False)
    except DocError as e:
        ok(f'ошибка в ссылке ловится при загрузке ({what})', 'ERROR' in str(e))
f = film([{'id': 'a', 'type': 'solid', 'opacity': "=layer('b').opacity"}, {'id': 'b', 'type': 'solid', 'opacity': "=layer('a').opacity"}])
try:
    f.layers[0]['opacity'].at(0.5)
    ok('ссылки по кругу — ошибка, а не зависание', False)
except ValueError as e:
    ok('ссылки по кругу — ошибка, а не зависание', 'по кругу' in str(e))

# ================= time =================
V = {'id': 'v', 'type': 'video', 'src': CLIP, 'size': [320, 180], 'position': [160, 90]}
for name, R in (('CPU', Renderer), ('GPU', GpuRenderer)):
    r = R(film([dict(V, time={0: 0, 2: 1, 'ease': 'linear'})]))
    x = ball_x(r.frame(1.0))
    ok(f'{name} Time Remap: на 1-й секунде фильма идёт 0,5 с исходника', abs(x - src_x(0.5)) < 1.5, f'шарик в {x:.1f}, нужно {src_x(0.5):.1f}')
    r = R(film([dict(V, time={0: 0.4, 1: 0.4, 2: 1.9, 'ease': 'linear'})]))
    x0, x1 = ball_x(r.frame(0.2)), ball_x(r.frame(0.8))
    ok(f'{name} Time Remap: стоп-кадр держится', abs(x0 - x1) < 0.2 and abs(x0 - src_x(0.4)) < 1.5, f'{x0:.1f} и {x1:.1f}')
    r = R(film([dict(V, time={0: 1.6, 2: 0.0, 'ease': 'linear'})]))
    ok(f'{name} Time Remap: назад — шарик летит влево', ball_x(r.frame(0.25)) > ball_x(r.frame(1.0)) > ball_x(r.frame(1.75)))
    # frame blending: half way between two source frames the ball is at both places, half as bright
    r = R(film([dict(V, time='=0.5', frame_blend=True)]))       # source 0.5 s = half way from frame 12 to frame 13
    img = r.frame(0.5).astype(float)
    xa, xb = src_x(0.48), src_x(0.52)                           # the ball in those two frames, 5.2 px apart
    edge = img[90, int(round(xb + 17.5)), 0]                    # the leading rim: only the later frame has the ball there
    ok(f'{name} frame_blend: между кадрами — их смесь пополам', 100 < edge < 170, f'яркость на переднем крае {edge:.0f} (один кадр дал бы 27 или 255)')
    # a comp's time remapped: its inner film held at 1.0 s
    inner = {'id': 'c', 'type': 'comp', 'size': [320, 180], 'position': [160, 90], 'time': 1.0, 'speed': 0,
             'layers': [{'id': 'b', 'type': 'solid', 'color': '#ffffff', 'size': [20, 20], 'position': {0: [20, 90], 2: [300, 90], 'ease': 'linear'}}]}
    r = R(film([dict(inner, time={0: 1.0, 2: 1.0})]))
    img = r.frame(0.3)
    cx = float((img[..., 0] > 128).sum(0) @ np.arange(320) / max(1, (img[..., 0] > 128).sum()))
    ok(f'{name} время композиции ключами: внутренний фильм стоит на 1,0 с', abs(cx - 159.5) < 1.0, f'квадрат в {cx:.1f}')
    # Posterize Time: at 5 steps a second the layer at t = 0.37 is where it was at 0.2
    mov = {'id': 'm', 'type': 'solid', 'color': '#ffffff', 'size': [20, 20], 'position': {0: [20, 90], 2: [300, 90], 'ease': 'linear'}}
    r = R(film([dict(mov, posterize=5)]))
    cx = lambda im: float((im[..., 0] > 128).sum(0) @ np.arange(320) / max(1, (im[..., 0] > 128).sum()))
    a, b = cx(r.frame(0.37)), cx(r.frame(0.21))
    ok(f'{name} posterize 5: слой стоит между шагами', abs(a - b) < 0.05 and abs(a - (20 + 140 * 0.2 - 0.5)) < 1.0, f'{a:.1f} и {b:.1f}')
    # Echo: two copies 0.2 s and 0.4 s back at half and a quarter of the brightness
    r = R(film([dict(mov, echo={'count': 2, 'step': 0.2, 'decay': 0.5})]))
    img = r.frame(1.0).astype(float)
    now, e1, e2 = (img[90, int(20 + 140 * tt), 0] for tt in (1.0, 0.8, 0.6))
    ok(f'{name} echo: копии 0,2 и 0,4 с назад — в половину и в четверть', abs(now - 255) < 2 and abs(e1 - 127.5) < 3 and abs(e2 - 63.75) < 3, f'{now:.0f}, {e1:.0f}, {e2:.0f}')

print('ВСЁ СХОДИТСЯ' if not bad else f'ЕСТЬ РАСХОЖДЕНИЯ: {bad}')
sys.exit(1 if bad else 0)
