"""Known answers for a puppet directed from the document (film.poses, pose(), a scene's named `nodes`, `rig`): the pose
keys' own arithmetic, and a character of the owner's (examples/hero3d — a private scene; these checks are skipped without it) moved by his strings against the recording of the same film made
by the series' own renderer (clip.bin) — every part, at two hundred moments. python tests/ae/test_rig.py"""
import json, math, os, sys, tempfile, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr import scene3d

bad = []
HERO = os.path.join(HERE, '..', '..', 'examples', 'hero3d')
D = tempfile.mkdtemp()


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


# --- the keys' arithmetic
f = Film(data={'film': {'size': [64, 64], 'fps': 25, 'duration': 10, 'poses': {'p': {'defaults': {'a': 0, 'b': 1}, 'keys': [[1, 2, {'a': 10}], [2, 0, {'b': 5}], [2.5, 1, {'a': 0}]]}}},
               'layers': [dict(id='s', type='solid', color='#fff', size=[4, 4], position="=[pose('p', 'a'), pose('p', 'b')]")]})
at = lambda t: f.layers[0]['position'].at(t)
sm = lambda x: x * x * (3 - 2 * x)
ok('поза: до ключа — что было, после — что велено, между — плавный ход', at(.5)[0] == 0 and abs(at(2)[0] - 10 * sm(.5)) < 1e-9 and at(1.9)[1] == 1 and at(2)[1] == 5,
   f'a на 0,5 с = {at(.5)[0]:g}, на 2 с = {at(2)[0]:.3f} (ждём {10 * sm(.5):.3f}); b скачком на 2 с: {at(1.9)[1]:g} → {at(2)[1]:g}')
v = 10 * sm(.75)
ok('следующий ключ ловит значение на ходу', abs(at(3)[0] - (10 * sm(1.0) * (1 - sm(.5)))) < 1e-9 and at(3.6)[0] == 0 and abs(at(2.5)[0] - v) < 1e-9,
   f'на 2,5 с a = {at(2.5)[0]:.3f} (ещё шло к 10: {v:.3f}), на 3 с = {at(3)[0]:.3f}, на 3,6 с = {at(3.6)[0]:g}')
for what, poses, expr in (('нет такой позы — названо', {}, "=pose('q', 'a')"), ('нет такого значения — названо', {'p': {'defaults': {'a': 0}}}, "=pose('p', 'z')"),
                          ('ключ на значение не из defaults — назван', {'p': {'defaults': {'a': 0}, 'keys': [[0, 0, {'zz': 1}]]}}, "=pose('p', 'a')")):
    try:
        Film(data={'film': {'size': [64, 64], 'fps': 25, 'duration': 1, 'poses': poses}, 'layers': [dict(id='s', type='solid', color='#fff', size=[4, 4], opacity=expr)]})
        ok(what, False, 'документ принят')
    except DocError as e:
        ok(what, True, str(e).splitlines()[0][:110])

# --- the hero by his rig against the series' own recording
FIRST = os.path.join(HERO, 'first')                          # the scene as its first renderer made it, with the record of his moves
if os.path.exists(os.path.join(FIRST, 'clip.bin')):
    quiet = os.path.join(D, 'quiet.wav')
    with wave.open(quiet, 'wb') as w_:
        w_.setnchannels(1); w_.setsampwidth(2); w_.setframerate(8000); w_.writeframes(b'\0\0' * 8000)
    P = json.load(open(os.path.join(HERO, 'gta6_pose.json')))
    # the hero is a block figure of the kit (kadr/figure.py): his strings are the kit's own, on the scene in Kadr's words;
    # the recording is of the first scene, whose parts had plain names and whose root stood under the floor
    sys.path.insert(0, HERO)
    import hero as hero_kit
    film = Film(data={'film': {'size': [270, 480], 'fps': 60, 'duration': 69.9, 'audio': [dict(id='voice', src=quiet)],
                               'poses': hero_kit.poses([[k['t'], k['d'], k['p']] for k in P['keys']])},
                      'layers': [dict(id='hero', type='scene3d', src=os.path.join(HERO, 'scene.yaml'), rig=os.path.join(HERO, 'rig.yaml'))]})
    L = film.layers[0]
    sc = scene3d.Scene(os.path.join(FIRST, 'scene.json'), os.path.join(FIRST, 'clip.bin'))
    RECORDED = {'hero.body': 'root', 'hero.signL': 'dollarL', 'hero.signR': 'dollarR'}
    C = sc.C
    where = {nid: k * 11 for k, nid in enumerate(C['nodes'])}
    ids = {n.get('name'): i for i, n in enumerate(sc.nodes) if n.get('name')}
    worst = {}
    ts = np.linspace(0.05, 69.5, 200)
    for t in ts:
        row = sc.rows([t], t)[0]
        for name, key, pr in L['over']:
            kit_name, name = name, RECORDED.get(name, name[5:] if name.startswith('hero.') else name)
            nid = ids.get(name)
            if nid not in where:
                continue                                       # not in the recording (it did not move at the moments probed; the legs were never there)
            o = where[nid]
            v = pr.at(float(t))
            if kit_name == 'hero.body':                        # the kit hops the body in its own units above the seat; the recording — the root, in centimetres
                v = (0.0, (v[1] + .72) * 11 - 3, -22.0)
            if key == 'position':
                e = float(np.abs(np.array(v) - row[o:o + 3]).max())
            elif key == 'rotation':
                q = np.array(scene3d.euler(v))
                e = math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q, row[o + 3:o + 7] / np.linalg.norm(row[o + 3:o + 7])))))))
            elif key == 'scale':
                e = float(np.abs(np.array(v if isinstance(v, tuple) else (v, v, v)) - row[o + 7:o + 10]).max())
            else:
                e = abs((1.0 if v > .5 else 0.0) - round(float(row[o + 10])))
            worst[name, key] = max(worst.get((name, key), 0.0), e)
    speech = {('head', 'rotation'), ('lipL', 'visible'), ('lipR', 'visible'), ('mouthOpen', 'visible'), ('mouthOpen', 'scale'), ('mouthOpen', 'position'), ('teeth', 'visible'), ('teeth', 'scale')}
    body = {k: v for k, v in worst.items() if k not in speech}
    top = sorted(body.items(), key=lambda kv: -kv[1])[:3]
    # turns in degrees (the recording is 120 rows a second, straight between them: fast typing hands differ by hundredths
    # of a degree), places and sizes in the hero's own units
    ok('герой по ригу — как в записи сериала: тело, руки, глаза, веки, брови, губы',
       len(body) >= 18 and all(v < (.05 if k == 'rotation' else .03 if n.startswith('lid') else 2e-3) for (n, k), v in body.items()),   # a blink takes 0.13 s: 16 rows of the recording
       f'сверено {len(body)} свойств в {len(ts)} моментах; наибольшие расхождения: ' + ', '.join(f'{n}.{k} {v:.5f}' for (n, k), v in top))
    ok('голова: то же, кроме кивков в такт речи (здесь звук — тишина)', worst['head', 'rotation'] < 1.35, f"до {worst['head', 'rotation']:.2f}° (кивок в записи — до 1,26°)")
    ok('без звука рот закрыт', all(L2.at(5.0) < .5 for n, k, L2 in L['over'] if (n, k) in {('hero.mouthOpen', 'visible'), ('hero.teeth', 'visible')}) and all(L2.at(5.0) > .5 for n, k, L2 in L['over'] if (n, k) == ('hero.lipL', 'visible'))
       and sum(1 for n, k, L2 in L['over'] if n in ('hero.mouthOpen', 'hero.teeth', 'hero.lipL') and k == 'visible') == 3)
    # what the recording missed (it looked for movement at twelve moments only): the little hop on «Выходит»
    root = next(pr for n, k, pr in L['over'] if (n, k) == ('hero.body', 'position'))
    ok('прыжок героя, которого нет в записи, в риге есть', abs(root.at(3.2)[1] - (-.72 + .04)) < .002 and abs(root.at(1.0)[1] + .72) < 1e-6, f'тело над сиденьем на 1 с: y = {root.at(1.0)[1]:.3f}, на 3,2 с: {root.at(3.2)[1]:.3f} (ждём −0.72 и −0.68)')
    # day and night: the room's named lamps and materials follow the pose's `day`
    def room(day):
        fl = Film(data={'film': {'size': [270, 480], 'fps': 60, 'duration': 5, 'audio': [dict(id='voice', src=quiet)], 'poses': {'hero': {'from': os.path.join(HERO, 'rig.yaml'), 'keys': [[0, 0, {'day': day}]]}}},
                        'layers': [dict(id='hero', type='scene3d', src=os.path.join(HERO, 'scene.yaml'), rig=os.path.join(HERO, 'rig.yaml'), camera=dict(position=[0, 43, 232], look_at=[0, 51, -22], fov=30))]})
        Lr = fl.layers[0]
        lamps = {n: pr.at(1.0) for n, k, pr in Lr['lit']}
        stuff = {(n, k): pr.at(0.0) for n, k, pr in Lr['mts']}
        return lamps, stuff, fl
    (l0, m0, f0), (l7, m7, f7), (l1, m1, f1) = room(0), room(.7), room(1)
    stored = {l['name']: l['intensity'] for l in json.load(open(os.path.join(FIRST, 'scene.json')))['lights'] if l.get('name')}
    ok('свет комнаты по времени суток: вечер, день 0,7 (как в сохранённой сцене), полдень',
       l0['lamp'] == 62000 and l0['sun'] == 0 and all(abs(l7[n] - stored[n]) < 1e-3 * max(1, abs(stored[n])) for n in l7) and abs(l1['sun'] - 1.5) < 1e-9 and m0['skyDay', 'opacity'] == 0 and m1['skyDay', 'opacity'] == 1
       and abs(m7['lampBulb', 'shade'] - .44) < 1e-6,
       f"лампа {l0['lamp']:.0f} → {l7['lamp']:.0f} → {l1['lamp']:.0f}, солнце {l0['sun']:.2f} → {l7['sun']:.2f} → {l1['sun']:.2f}; при 0,7 все пять ламп — как в сцене")
    from kadr.gpu import GpuRenderer
    night, noon = GpuRenderer(f0).frame_u8(1.0).astype(float), GpuRenderer(f1).frame_u8(1.0).astype(float)
    win = (slice(40, 150), slice(20, 110))                    # the window: stars at night, sky by day
    ok('в кадре: окно ночью тёмное, днём светлое', night[win].mean() + 40 < noon[win].mean(), f'яркость окна {night[win].mean():.0f} → {noon[win].mean():.0f}')
    for what, extra in (('опечатка в имени лампы названа', dict(lights={'lmap': {'intensity': 1}})), ('опечатка в свойстве материала названа', dict(materials={'skyDay': {'opacty': 1}}))):
        try:
            Film(data={'film': {'size': [64, 64], 'fps': 25, 'duration': 1}, 'layers': [dict(id='h', type='scene3d', src=os.path.join(HERO, 'scene.yaml'), camera=dict(position=[0, 40, 200], look_at=[0, 40, 0]), **extra)]})
            ok(what, False)
        except DocError as e:
            ok(what, True, str(e).splitlines()[0][:110])
    try:
        Film(data={'film': {'size': [64, 64], 'fps': 25, 'duration': 1}, 'layers': [dict(id='h', type='scene3d', src=os.path.join(HERO, 'scene.yaml'), nodes={'hero.haed': {'rotation': [0, 0, 0]}}, camera=dict(position=[0, 40, 200], look_at=[0, 40, 0]))]})
        ok('опечатка в имени узла названа', False)
    except DocError as e:
        ok('опечатка в имени узла названа', 'hero.head?' in str(e), str(e).splitlines()[0][:110])
else:
    print('ПРОПУЩЕНО: нет examples/hero3d')

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
