"""Known answers for sound as a source (AE 48, kadr/sound.py): tones of a known loudness and pitch, written here, must
come back as those numbers from amp() and band(), stand as the right bar of a spectrum and the right curve of a wave.
python tests/ae/test_sound.py"""
import math, os, sys, tempfile, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr.render import Renderer, to_u8
from kadr.gpu import GpuRenderer
from kadr import sound

bad = []
D = tempfile.mkdtemp()


def wav(name, x, sr=48000):
    p = os.path.join(D, name)
    with wave.open(p, 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())
    return p


def tone(hz, amp, sec, sr=48000):
    return amp * np.sin(2 * np.pi * hz * np.arange(int(sec * sr)) / sr)


STEPS = wav('steps.wav', np.concatenate([tone(440, .2, 1), tone(440, .8, 1), np.zeros(48000)]))   # quiet, loud, silence
TWO = wav('two.wav', np.concatenate([tone(200, .5, 1), tone(3000, .5, 1)]))                       # a low second, a high second
HUM = wav('hum.wav', tone(100, .5, 1))


def film(layers, audio=None, **f):
    return Film(data={'film': dict({'size': [400, 200], 'fps': 25, 'duration': 3, 'background': '#000000'}, **f, **({'audio': audio} if audio else {})), 'layers': layers})


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def near(a, b, tol):
    return abs(a - b) <= tol


# --- the numbers themselves
F = film([], [dict(src=STEPS, id='tone')])
v = [sound.amp(F, t, fall=0, norm=False) for t in (.5, 1.5, 2.5)]
ok('громкость синуса 0,2 и 0,8 — его среднеквадратичное', near(v[0], .2 / math.sqrt(2), .002) and near(v[1], .8 / math.sqrt(2), .006) and v[2] < 1e-3, f'{v[0]:.4f} (ждём 0.1414), {v[1]:.4f} (0.5657), тишина {v[2]:.5f}')
v = [sound.amp(F, t, fall=0) for t in (.5, 1.5, 2.5)]
ok('amp() — доля от самого громкого места', near(v[0], .25, .004) and near(v[1], 1, .004) and v[2] < 1e-3, f'{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f} (ждём 0.25, 1, 0)')
v = sound.amp(F, 2.25, fall=.5)
ok('после удара значение спадает за fall', near(v, math.exp(-.275 / .5), .03), f'через 0,25 с после конца звука {v:.3f} (ждём {math.exp(-.275 / .5):.3f})')
v = [sound.band(F, 1.5, 400, 480, fall=0), sound.band(F, 1.5, 400, 480, fall=0, norm=False), sound.band(F, 1.5, 1000, 4000, fall=0), sound.band(F, 1.5, 40, 200, fall=0)]
ok('band() слышит тон только в его полосе', near(v[0], 1, .03) and near(v[1], .8 / math.sqrt(2), .02) and v[2] < .02 and v[3] < .02,
   f'440 Гц в полосе 400–480: {v[0]:.3f} (уровень {v[1]:.3f}, ждём 0.566), в 1000–4000: {v[2]:.4f}, в 40–200: {v[3]:.4f}')
G = film([], [dict(src=TWO)])
v = [sound.band(G, t, lo, hi, fall=0) for t, lo, hi in ((.5, 150, 260), (.5, 2500, 3600), (1.5, 150, 260), (1.5, 2500, 3600))]
ok('низкий и высокий тон — каждый в свою секунду', v[0] > .97 and v[1] < .02 and v[2] < .02 and v[3] > .97, ', '.join(f'{x:.3f}' for x in v))

# --- where the entry stands in the film
P = film([], [dict(src=STEPS, id='tone', at=1.0)])
Q = film([], [dict(src=STEPS, id='tone', **{'from': 1.0}), dict(src=HUM, role='music', at=2.0)])
v = [sound.amp(P, .5, 'tone', fall=0), sound.amp(P, 1.5, 'tone', fall=0), sound.amp(Q, .5, 'tone', fall=0), sound.amp(Q, .5, 'voice', fall=0), sound.amp(Q, .5, fall=0), sound.amp(Q, 2.5, fall=0), sound.amp(Q, 2.5, HUM, fall=0)]
ok('место дорожки в фильме (at, from) учтено', v[0] == 0 and near(v[1], .25, .004) and near(v[2], 1, .004), f'at: 1 → на 0,5 с {v[0]:.2f}, на 1,5 с {v[1]:.3f}; from: 1 → на 0,5 с {v[2]:.3f}')
ok('источник по роли, по умолчанию (музыка) и по файлу', near(v[3], 1, .004) and v[4] == 0 and near(v[5], 1, .01) and near(v[6], 0, .01), f'voice {v[3]:.3f}; без имени — музыка: на 0,5 с {v[4]:.2f}, на 2,5 с {v[5]:.3f}; файл с нуля на 2,5 с {v[6]:.3f}')
for what, layers in (('опечатка в имени звука названа', [dict(id='a', type='solid', color='#ffffff', size=[10, 10], scale='=1 + amp("tome")')]),
                     ('опечатка в параметре названа', [dict(id='a', type='solid', color='#ffffff', size=[10, 10], scale='=1 + amp("tone", fal=0)')]),
                     ('слою spectrum нужен существующий звук', [dict(id='a', type='spectrum', audio='nope')])):
    try:
        film(layers, [dict(src=STEPS, id='tone')])
        ok(what, False, 'документ принят')
    except DocError as e:
        ok(what, True, str(e).splitlines()[0][:120])
    except Exception as e:
        ok(what, False, f'падение: {type(e).__name__}: {e}')

# --- a layer driven by the sound, on both renderers, and inside a comp
box = dict(id='b', type='solid', color='#ffffff', size=[100, 40], position=[200, 100], scale='=1 + amp(fall=0)')
f = film([box], [dict(src=STEPS)])
ws = []
for R in (Renderer(f), GpuRenderer(f)):
    for t in (.5, 1.5, 2.5):
        a = to_u8(R.frame(t)) if type(R) is Renderer else R.frame_u8(t)
        ws.append(int((a[100, :, 0] > 127).sum()))
ok('масштаб по громкости: 1,25 → 2 → 1', all(near(w, e, 2) for w, e in zip(ws, (125, 200, 100) * 2)), f'ширина плашки {ws[:3]} и на GPU {ws[3:]} (ждём 125, 200, 100)')
f = film([dict(id='c', type='comp', **{'in': 1.0}, layers=[box])], [dict(src=STEPS)])
a = to_u8(Renderer(f).frame(1.5))
ok('во вставке, начатой с 1 с, звук — по времени фильма', near(int((a[100, :, 0] > 127).sum()), 200, 2), f'ширина на 1,5 с фильма {(a[100, :, 0] > 127).sum()} (ждём 200)')

# --- the spectrum: the tone's bar stands where its pitch is, as tall as the plane; it falls in `fall`
S = dict(id='s', type='spectrum', size=[400, 200], position=[200, 100], bands=32, range=[50, 8000], gap=0, round=False, floor=0)
f = film([dict(S, fall=0)], [dict(src=TWO)])
R, Gp = Renderer(f), GpuRenderer(f)
res = []
for t, hz in ((.5, 200), (1.5, 3000)):
    a = to_u8(R.frame(t))[..., 0] > 127
    col = a.sum(0)
    x = float(np.nonzero(col == col.max())[0].mean())
    want = 400 * math.log(hz / 50) / math.log(8000 / 50)
    res.append((x, want, int(col.max()), int((col > 0).sum())))
    d = np.abs(to_u8(R.frame(t)).astype(int) - Gp.frame_u8(t)).max()
ok('полоса тона стоит на месте его частоты', all(abs(x - w) <= 12.5 and hgt >= 170 and wide <= 5 * 12.5 for x, w, hgt, wide in res),
   '; '.join(f'середина самой высокой x={x:.0f} (ждём {w:.0f}), высота {hgt} из 200, занято {wide} px' for x, w, hgt, wide in res))
ok('спектр одинаков на обоих рендерах', d <= 3, f'наибольшая разница {d}')
f = film([dict(S, fall=.35)], [dict(src=TWO)])
h0, h1 = (int((to_u8(Renderer(f).frame(t))[:, 105:115, 0] > 127).any(1).sum()) for t in (.92, 1.08))
drop = (.08 + .035) / .35                                      # the tone is heard at full until its 85 ms window is mostly past its end
ok('полоса падает за fall', h0 >= 170 and near(h1 / h0, 1 - drop, .06), f'до конца тона {h0} px, через 0,08 с после него {h1} px (ждём около {h0 * (1 - drop):.0f})')
f = film([dict(S, fall=0, ring=30, size=[200, 200], gap=.2)], [dict(src=TWO)])
a = to_u8(Renderer(f).frame(.5))[..., 0] > 127
ys, xs = np.nonzero(a)
far = np.hypot(xs - 200, ys - 100).argmax()
ang = math.degrees(math.atan2(ys[far] - 100, xs[far] - 200))
want = -90 + 360 * (math.floor(32 * math.log(200 / 50) / math.log(8000 / 50)) + .5) / 32
ok('по кругу (ring): полоса тона — под своим углом, до края плоскости', near(ang, want, 6) and near(float(np.hypot(xs[far] - 200, ys[far] - 100)), 100, 8),
   f'самая дальняя точка под углом {ang:.1f}° (ждём {want:.1f}°), в {np.hypot(xs[far] - 200, ys[far] - 100):.0f} px от центра (ждём около 100)')

# --- the wave: an oscilloscope and an audiogram
f = film([dict(id='w', type='wave', size=[400, 200], position=[200, 100], window=.05, width=2, norm=False)], [dict(src=HUM)])
kind, lo, hi = sound.wave_data(f, f.layers[0], .5)
mid = (lo + hi) / 2 if hi is not None else lo
cross = int((np.signbit(mid[1:]) != np.signbit(mid[:-1])).sum())
a = to_u8(Renderer(f).frame(.5))[..., 0] > 127
top, bot = np.nonzero(a.any(1))[0][[0, -1]]
ok('осциллограмма: 100 Гц в окне 0,05 с — пять волн', cross in (9, 10, 11) and near(float(np.abs(mid).max()), .5, .02), f'пересечений середины {cross} (ждём 10), размах {np.abs(mid).max():.3f} (ждём 0.5)')
ok('…и нарисована на половину высоты', near(top, 50, 3) and near(bot, 150, 3), f'от y={top} до y={bot} (ждём 50…150)')
f = film([dict(id='w', type='wave', size=[400, 200], position=[200, 100], span=[0, 3], shape='bars', bars=30, gap=0, round=False, color='#ffffff', played='#ff0000')], [dict(src=STEPS)])
img = to_u8(Renderer(f).frame(1.5))
white, red = (img[..., 1] > 127), (img[..., 0] > 127) & (img[..., 1] < 80)
hq, hl = int(red[:, 60].sum()), int(white[:, 260].sum())
ok('аудиограмма: тихая секунда — четверть высоты, громкая — вся', near(hq, 50, 3) and near(hl, 200, 3) and int((white | red)[:, 340].sum()) <= 2, f'столбик тихой части {hq} px (ждём 50), громкой {hl} (ждём 200), тишины {(white | red)[:, 340].sum()}')
ok('…и прозвучавшее к 1,5 с окрашено', near(int(red.any(0).sum()), 200, 8) and not red[:, 205:].any(), f'красных колонок {red.any(0).sum()} из 400 (ждём 200)')
d = np.abs(img.astype(int) - GpuRenderer(f).frame_u8(1.5)).max()
ok('волна одинакова на обоих рендерах', d <= 3, f'наибольшая разница {d}')

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
