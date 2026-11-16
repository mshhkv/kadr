"""Known answers for the music's pulse (kadr/sound.py: find_beats, beat:N, beat()): click tracks whose every beat is
known, and the made-up beat of the sound demos. python tests/ae/test_beats.py"""
import os, sys, tempfile, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr import sound

bad = []
D = tempfile.mkdtemp()
rng = np.random.default_rng(1)


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def clicks(name, bpm, off, dur=24, sr=24000, miss=()):
    x = np.zeros(sr * dur)
    true = np.arange(off, dur - .2, 60 / bpm)
    for k, t in enumerate(true):
        if k in miss:
            continue
        i, n = int(t * sr), int(.06 * sr)
        x[i:i + n] += rng.standard_normal(n) * np.exp(-np.arange(n) / (.012 * sr)) * .6
    p = os.path.join(D, name)
    with wave.open(p, 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())
    return p, true


for bpm, off in ((96, .31), (128, .12), (74, .5), (150, .05)):
    p, true = clicks(f'c{bpm}.wav', bpm, off)
    tempo, bt = sound.find_beats(sound.track(p))
    err = np.array([b - true[np.argmin(np.abs(true - b))] for b in bt]) if len(bt) else np.array([9.0])
    ok(f'метроном {bpm} ударов в минуту: темп и каждый удар', abs(tempo - bpm) < .5 and len(bt) == len(true) and np.abs(err).max() < .012,
       f'темп {tempo:.2f}, битов {len(bt)} из {len(true)}, наибольшая ошибка {np.abs(err).max() * 1000:.1f} мс')
p, true = clicks('gaps.wav', 100, .2, miss=(7, 8, 15, 22, 23, 24))
tempo, bt = sound.find_beats(sound.track(p))
err = np.array([b - true[np.argmin(np.abs(true - b))] for b in bt])
ok('пропущенные удары пульс не сбивают', abs(tempo - 100) < .5 and abs(len(bt) - len(true)) <= 1 and np.abs(err).max() < .03, f'темп {tempo:.2f}, битов {len(bt)} из {len(true)} (шесть не звучали), наибольшая ошибка {np.abs(err).max() * 1000:.1f} мс')
tempo, bt = sound.find_beats(sound.track(os.path.join(HERE, 'beat.wav')))
err = np.array([b - round(b / .5) * .5 for b in bt])
ok('бит демо-ролика (бочка каждые полсекунды)', abs(tempo - 120) < 1 and len(bt) == 16 and np.abs(err[1:]).max() < .012 and abs(err[0]) < .04,
   f'темп {tempo:.1f}, битов {len(bt)}, наибольшая ошибка {np.abs(err[1:]).max() * 1000:.1f} мс (первый удар, с которого файл начинается, — {err[0] * 1000:.0f} мс)')

# --- in a document
p, true = clicks('c96.wav', 96, .31)
f = Film(data={'film': {'size': [64, 64], 'fps': 25, 'duration': 24, 'audio': [dict(id='m', src=p, role='music', at=1.0)]},
               'layers': [dict(id='a', type='solid', color='#fff', size=[4, 4], **{'in': 'beat:5', 'out': 'beat:9+0.1'}, scale='=beat()', opacity={'beat:5': 0, 'beat:6.5': 1})]})
L = f.layers[0]
ok('время словом beat:N — на свой удар (дорожка стоит с 1-й секунды)', abs(L['in'] - (1 + true[4])) < .012 and abs(L['out'] - (1 + true[8] + .1)) < .012,
   f"beat:5 = {L['in']:.3f} с (удар на {1 + true[4]:.3f}), beat:9+0.1 = {L['out']:.3f}")
k = [x[0] for x in L['opacity'].keys]
ok('beat:6.5 — между ударами', abs(k[1] - (1 + (true[5] + true[6]) / 2)) < .012, f'{k[1]:.3f} с (ждём {1 + (true[5] + true[6]) / 2:.3f})')
v = [L['scale'].at(1 + true[4]), L['scale'].at(1 + (true[4] + true[5]) / 2)]
ok('формула beat() — место в ряду ударов', abs(v[0] - 5) < .03 and abs(v[1] - 5.5) < .03, f'на пятом ударе {v[0]:.3f}, посередине до шестого {v[1]:.3f}')
for what, t in (('удара с таким номером нет — названо', 'beat:500'), ('нет музыки — названо', 'beat:nothing:3')):
    try:
        Film(data={'film': {'size': [64, 64], 'fps': 25, 'duration': 2, 'audio': [dict(id='m', src=p, role='music')]}, 'layers': [dict(id='a', type='solid', color='#fff', size=[4, 4], **{'in': t})]})
        ok(what, False, 'документ принят')
    except DocError as e:
        ok(what, True, str(e).splitlines()[0][:110])

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
