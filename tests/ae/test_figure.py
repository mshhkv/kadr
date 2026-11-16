"""The block figure (kadr/figure.py — `shape: figure` in a scene): a character from a few words stands where it is put
and is as tall as it is told, sits on what it is sat on, keeps its height when its proportions change; its strings come
with it (poses, a rest of its own, a mouth for each voice, steps), a film may still move a part itself; what is hung on
a part moves with the part; mistakes in its words are named; the scene's check finds nothing wrong in it.
python tests/ae/test_figure.py"""
import math, os, struct, sys, tempfile, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr import scene3d, figure, scenecheck as sk

bad = []
D = tempfile.mkdtemp()


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def film(nodes, poses=None, audio=None, layer=None, dur=6):
    return Film(data={'film': {'size': [320, 240], 'fps': 30, 'duration': dur, **({'poses': poses} if poses else {}), **({'audio': audio} if audio else {})},
                      'layers': [dict(id='s', type='scene3d', scene=dict(camera=dict(position=[0, 40, 260], look_at=[0, 30, 0], fov=30), lights=[dict(type='sky', intensity=1)], nodes=nodes), **(layer or {}))]})


def boxes(F, t=0.0, only=None):
    """every body's own box in the world at t → {name: (lo, hi)} (seen ones only)"""
    L = F.layers[0]
    sc = scene3d.layer_scene(L)
    world, seen = sk.pose(sc, L, t, 30)
    out = {}
    for i, n in enumerate(sc.nodes):
        if n['kind'] != 'Mesh' or not seen[i] or (only and not n['name'].startswith(only)):
            continue
        first, cnt, base = sc.geos[n['geo']]
        v = sc.verts[base + np.unique(sc.indices[first:first + cnt]), :3].astype(float)
        p = v @ world[i][:3, :3].T + world[i][:3, 3]
        out[n['name']] = (p.min(0), p.max(0))
    return out


def tone(name, spans, dur=6.0, sr=8000):
    path = os.path.join(D, name)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b''.join(struct.pack('<h', int(math.sin(i / sr * 2 * math.pi * 180) * 20000) if any(a <= i / sr < b for a, b in spans) else 0) for i in range(int(dur * sr))))
    return path


def issues(nodes, **kw):
    try:
        F = film(nodes, **kw)
        return [str(i) for i in F.issues]
    except DocError as e:
        return str(e).splitlines()


# ---- 1. a figure from two words
F = film([dict(id='bob', shape='figure', position=[30, 5, -10], height=60, rest={'lean': 0, 'hp': 0})])
B = boxes(F)
lo, hi = np.min([b[0] for b in B.values()], 0), np.max([b[1] for b in B.values()], 0)
face = B['bob.face']
ok('фигурка из двух слов собрана: туловище, руки, ноги, голова с лицом и волосами', len(B) > 25 and all(k.startswith('bob.') for k in B), f'{len(B)} тел')
ok('стоит подошвами там, где сказано', abs(lo[1] - 5) < .02 and abs((B['bob.shoeL'][0][0] + B['bob.shoeR'][1][0]) / 2 - 30) < .02, f'низ на высоте {lo[1]:.3f}, середина между ботинками x = {(B["bob.shoeL"][0][0] + B["bob.shoeR"][1][0]) / 2:.3f}')
# (he breathes: the torso, and the head on it, is up to 1.2 % taller or shorter at any moment — half a centimetre here)
ok('рост — до макушки, как сказано', abs(face[1][1] - 65) < .6, f'верх головы на {face[1][1]:.2f} (ждём 65: стоит на 5, рост 60)')
ok('ноги стоящего смотрят вниз, руки опущены', B['bob.pantL'][1][1] - B['bob.pantL'][0][1] > B['bob.pantL'][1][2] - B['bob.pantL'][0][2] and B['bob.handL'][1][1] < B['bob.top'][0][1] + 6)
ok('по умолчанию он ростом с героя сериала', abs(boxes(film([dict(id='a', shape='figure', rest={'lean': 0, 'hp': 0})]))['a.face'][1][1] - 54.23) < .6)

# ---- 2. sitting: placed by where it sits
F = film([dict(id='seat', shape='box', size=[40, 20, 30], position=[0, 10, 0]), dict(id='sam', shape='figure', position=[0, 20, 0], sits=True, height=60)])
B = boxes(F, only='sam.')
ok('сидящий: низ бёдер — на сиденье, ноги вытянуты вперёд', abs(B['sam.pantL'][0][1] - 20) < .05 and B['sam.pantL'][1][2] - B['sam.pantL'][0][2] > 1.8 * (B['sam.pantL'][1][1] - B['sam.pantL'][0][1]) and B['sam.shoeL'][1][2] > B['sam.pantL'][1][2],
   f'низ бедра на {B["sam.pantL"][0][1]:.3f} (сиденье на 20)')
sc = scene3d.layer_scene(F.layers[0])
w, s = sk.pose(sc, F.layers[0], 0.0)
ok('проверка сцены: сидящему на скамейке сказать нечего', not sk.audit(sc, w, s), '; '.join(r[2] for r in sk.audit(sc, w, s)))
F2 = film([dict(id='seat', shape='box', size=[40, 20, 30], position=[0, 10, 0]), dict(id='sam', shape='figure', position=[0, 14, 0], sits=True, height=60)])
sc2 = scene3d.layer_scene(F2.layers[0])
w, s = sk.pose(sc2, F2.layers[0], 0.0)
r = sk.audit(sc2, w, s)
ok('посаженный на 6 ниже — назван: ноги в скамейке', any(k == 'through' for k, _, _ in r), '; '.join(x[2] for x in r)[:150])

# ---- 3. proportions keep the height; the parts follow
tall = boxes(film([dict(id='k', shape='figure', height=40, build=dict(head=1.3, legs=.6, arms=.8, wide=1.2), rest={'lean': 0, 'hp': 0})]))
norm = boxes(film([dict(id='k', shape='figure', height=40, rest={'lean': 0, 'hp': 0})]))
size = lambda b, ax: b[1][ax] - b[0][ax]
ok('пропорции: рост прежний, голова крупнее, ноги короче, плечи шире',
   abs(tall['k.face'][1][1] - 40) < .5 and abs(min(b[0][1] for b in tall.values())) < .02 and size(tall['k.face'], 0) > size(norm['k.face'], 0) * 1.25 and size(tall['k.pantL'], 1) < size(norm['k.pantL'], 1) * .7
   and size(tall['k.top'], 0) > size(norm['k.top'], 0) * 1.15, f"голова {size(norm['k.face'], 0):.1f} → {size(tall['k.face'], 0):.1f}, нога {size(norm['k.pantL'], 1):.1f} → {size(tall['k.pantL'], 1):.1f}")
ok('руки растут из плеч, какой бы ширины они ни были: рукав касается туловища', abs(tall['k.sleeveL'][1][0] - tall['k.top'][0][0]) < size(tall['k.top'], 0) * .12 and tall['k.pantL'][0][0] >= tall['k.top'][0][0] - .05)

# ---- 4. what it wears
B = boxes(film([dict(id='m', shape='figure', sleeves='short', hair=dict(color='#7a3b1d', style='long'), glasses='#222222', pocket=False, stripes='#ffffff', rest={'hat': 1}, hat='cap')]))
ok('футболка, очки, полоски, длинные волосы, кепка — из слов', all(f'm.{k}' in B for k in ('forearmL', 'lensL', 'templeR', 'stripeL', 'hairBack', 'hatVisor', 'hatBand')) and 'm.pocket' not in B, ', '.join(sorted(k[2:] for k in B))[:140])
ok('под надетой шапкой волосы на макушке спрятаны, а спадающие остались', 'm.hairCap' not in B and 'm.hairBack' in B)
ok('кепка сидит на голове: ободок охватывает её верх, зазора нет', B['m.hatBand'][0][1] < B['m.face'][1][1] - 1 and B['m.hatBand'][0][0] < B['m.face'][0][0] and B['m.hatBand'][1][0] > B['m.face'][1][0])
B0 = boxes(film([dict(id='m', shape='figure', hair='none', hat='beanie')]))
ok('лысый; шапка, пока её не надели позой, не видна', not any('hair' in k or 'tuft' in k or 'hat' in k for k in B0), ', '.join(k for k in B0 if 'ha' in k))
a, b = (boxes(film([dict(id=i, shape='figure')]))[f'{i}.tuft2'][0] for i in ('ann', 'joe'))
F = film([dict(id='ann', shape='figure'), dict(id='joe', shape='figure', position=[40, 0, 0])])
bl = [g['geo']['blink'] for g in F.layers[0]['made']['figures']]
ok('случайное у каждого своё (вихры, когда моргает), а у одного и того же — всегда одно', abs(a[2] - b[2]) > 1e-3 and bl[0] != bl[1] and
   np.allclose(boxes(film([dict(id='ann', shape='figure')]))['ann.tuft2'][0], a), f'моргают: {bl[0]} и {bl[1]}')

# ---- 5. the strings come with it
F = film([dict(id='bob', shape='figure')])
L = F.layers[0]
ok('без поз в фильме фигурка стоит как в покое и дышит', 'bob' in F.poses and abs(next(p for n, k, p in L['over'] if (n, k) == ('bob.torso', 'scale')).at(.83)[1] - 1) > .005)
lid = next(p for n, k, p in L['over'] if (n, k) == ('bob.lidL', 'scale'))
ys = [lid.at(t)[1] for t in np.arange(0, 9, .02)]
ok('и моргает', max(ys) > .5 and np.median(ys) < .05, f'веко от {min(ys):.3f} до {max(ys):.3f}')
F = film([dict(id='bob', shape='figure', rest={'armRf': 1.9, 'smile': .9})], poses={'bob': {'from': 'figure', 'keys': [[1, .5, {'hy': .6, 'y': .2}], [3, .5, {'armRf': .1}]]}})
B0, B1, B2 = boxes(F, 0), boxes(F, 2), boxes(F, 4)
ok('поза покоя — из слов фигурки: рука поднята с первого кадра; ключ фильма её опускает', B0['bob.handR'][1][1] > B0['bob.top'][1][1] and B2['bob.handR'][1][1] < B2['bob.top'][0][1] + 8,
   f"кисть на {B0['bob.handR'][1][1]:.1f} → {B2['bob.handR'][1][1]:.1f}, верх туловища {B0['bob.top'][1][1]:.1f}")
ok('ключи фильма ведут его: подпрыгнул и повернул голову', B1['bob.shoeL'][0][1] > B0['bob.shoeL'][0][1] + 1.5 and abs(B1['bob.browL'][0][0] - B0['bob.browL'][0][0]) > 1)
F = film([dict(id='bob', shape='figure')], poses={'bob': {'from': 'figure', 'defaults': {'smile': 1}}})
ok('покой, заданный в самом фильме, главнее слов сцены', F.poses['bob']['defaults']['smile'] == 1 and film([dict(id='bob', shape='figure', rest={'smile': .9})], poses={'bob': {'from': 'figure', 'defaults': {'smile': 1}}}).poses['bob']['defaults']['smile'] == 1)
F = film([dict(id='bob', shape='figure', pose='boss'), dict(id='tom', shape='figure', pose='boss', position=[40, 0, 0])], poses={'boss': {'from': 'figure', 'keys': [[0, 0, {'armLf': 1.9}]]}})
B = boxes(F, .1)
ok('одной позой можно вести двоих', B['bob.handL'][1][1] > B['bob.top'][1][1] and B['tom.handL'][1][1] > B['tom.top'][1][1])
F = film([dict(id='bob', shape='figure')], poses={'bob': {'from': 'figure', 'keys': [[0, 0, {'walk': 1}]]}})
st = [(boxes(F, t)['bob.shoeL'][0][2], boxes(F, t)['bob.shoeR'][0][2], boxes(F, t)['bob.handL'][0][2]) for t in np.arange(0, 1.8, .15)]
ok('шаги: ноги ходят в противоход, рука — навстречу своей ноге', max(a - b for a, b, c in st) > 4 and min(a - b for a, b, c in st) < -4 and np.corrcoef([a for a, b, c in st], [c for a, b, c in st])[0, 1] < -.8,
   f'левый ботинок впереди правого от {min(a - b for a, b, c in st):.1f} до {max(a - b for a, b, c in st):.1f}')
F = film([dict(id='bob', shape='figure')], layer=dict(nodes={'bob.head': {'rotation': [0, 90, 0]}}))
B, Bn = boxes(F), boxes(film([dict(id='bob', shape='figure')]))
ok('часть можно повернуть и самому — слово фильма главнее нитки', B['bob.browL'][1][0] - B['bob.browL'][0][0] < .5 * (Bn['bob.browL'][1][0] - Bn['bob.browL'][0][0]))

# ---- 6. a mouth for each voice
va, vb = tone('a.wav', [(.5, 2)]), tone('b.wav', [(2.5, 4)])
two = [dict(id='ann', shape='figure'), dict(id='joe', shape='figure', position=[40, 0, 0], voice='second')]
F = film(two, audio=[dict(id='ann', src=va, role='voice', mute=True), dict(id='second', src=vb, role='voice', mute=True)])
mouth = lambda F, who, t: next(p for n, k, p in F.layers[0]['over'] if (n, k) == (f'{who}.mouthOpen', 'visible')).at(t) > .5
ok('двое говорят по очереди: рот открывает тот, чей голос звучит', mouth(F, 'ann', 1.2) and not mouth(F, 'joe', 1.2) and mouth(F, 'joe', 3.2) and not mouth(F, 'ann', 3.2) and not mouth(F, 'ann', 4.8),
   f"1,2 с: {mouth(F, 'ann', 1.2)}/{mouth(F, 'joe', 1.2)}, 3,2 с: {mouth(F, 'ann', 3.2)}/{mouth(F, 'joe', 3.2)}")
B = boxes(F, 1.2)
ok('открытый рот виден, губы убраны', 'ann.mouthOpen' in B and 'ann.lipL' not in B and 'joe.lipL' in B)
F = film([dict(id='bob', shape='figure')], audio=[dict(id='music', src=va, role='music')])
ok('у кого голоса нет — молчит, и это не ошибка', not mouth(F, 'bob', 1.2) and not F.issues)
r = issues([dict(id='bob', shape='figure', voice='bobb')], audio=[dict(id='bob', src=va, role='voice')])
ok('а названный, но не найденный голос — предупреждение', any('bobb' in x and 'рот' in x for x in r), '; '.join(r)[:120])

# ---- 7. things hung on its parts
mug = dict(id='mug', shape='cylinder', radius=.3, height=.6, position=[0, -.4, 0])
F = film([dict(id='bob', shape='figure', attach={'handR': [mug]})], poses={'bob': {'from': 'figure', 'keys': [[1, .5, {'armRf': 1.9}]]}})
B0, B1 = boxes(F, 0), boxes(F, 2)
ok('кружка в руке: поднимается вместе с рукой', B1['mug'][0][1] > B0['mug'][0][1] + 10 and abs((B1['mug'][0] - B1['bob.handR'][0])[1] - (B0['mug'][0] - B0['bob.handR'][0])[1]) < 6,
   f"кружка на высоте {B0['mug'][0][1]:.1f} → {B1['mug'][0][1]:.1f}")
sc = scene3d.layer_scene(F.layers[0])
w, s = sk.pose(sc, F.layers[0], 0.0)
ok('и для проверки сцены она — часть фигурки, а не посторонняя вещь в его руке', not sk.audit(sc, w, s), '; '.join(x[2] for x in sk.audit(sc, w, s))[:120])

# ---- 8. mistakes are named
for what, node, word in (('опечатка в поле', dict(id='b', shape='figure', hieght=50), 'height'), ('неизвестная причёска', dict(id='b', shape='figure', hair=dict(style='tuft')), 'tufts'),
                         ('неизвестная шапка', dict(id='b', shape='figure', hat='helmit'), 'helmet'), ('неизвестное значение покоя', dict(id='b', shape='figure', rest={'smlie': 1}), 'smile'),
                         ('неизвестная часть для подвеса', dict(id='b', shape='figure', attach={'hand': [mug]}), 'hand'), ('цвет не цвет', dict(id='b', shape='figure', top='red-ish'), 'цвет'),
                         ('пропорция вне разумного', dict(id='b', shape='figure', build={'head': 9}), 'build.head')):
    r = issues([node])
    ok(f'{what}: сказано', any(word in x for x in r) and any(x.startswith('ERROR') for x in r), (r[0] if r else 'принято молча')[:110])
r = issues([dict(id='b', shape='figure'), dict(id='b', shape='figure')])
ok('две фигурки с одним именем: сказано', any('две фигурки' in x for x in r), (r[0] if r else '')[:100])
try:
    Film(data={'film': {'size': [64, 64], 'fps': 25, 'duration': 1, 'poses': {'b': {'from': 'figure', 'keys': [[0, 0, {'smlie': 1}]]}}}, 'layers': []})
    ok('ключ на значение, которого у фигурки нет: сказано', False)
except DocError as e:
    ok('ключ на значение, которого у фигурки нет: сказано', 'smile' in str(e), str(e).splitlines()[0][:110])

# ---- 9. the example: four figures, no body written by hand
EX = os.path.join(HERE, '..', '..', 'examples', 'figures')
src = open(os.path.join(EX, 'scene.yaml'), encoding='utf-8').read()
_, L = sk.open_scene(os.path.join(EX, 'scene.yaml'), None)
sc = scene3d.layer_scene(L)
w, s = sk.pose(sc, L, 0.0)
r = sk.audit(sc, w, s)
figs = [f['id'] for f in L['made']['figures']]
ok('пример: четыре персонажа, у каждого — только слова набора; проверке сцены сказать нечего', figs == ['bob', 'mia', 'tim', 'sam'] and not r and sum(1 for n in sc.nodes if n['kind'] == 'Mesh') > 100,
   f"тел {sum(1 for n in sc.nodes if n['kind'] == 'Mesh')}, замечаний {len(r)}")
Fx = Film(os.path.join(EX, 'figures.yaml'))
from kadr.gpu import GpuRenderer
R = GpuRenderer(Fx)
a, b = R.frame_u8(.5)[..., :3].astype(int), R.frame_u8(5.9)[..., :3].astype(int)
ok('пример-фильм рисуется, и в нём есть движение', np.abs(a - b).mean() > 1 and a.std() > 20 and not Fx.issues, f'средняя разница кадров 0,5 и 5,9 с: {np.abs(a - b).mean():.2f} из 255')

print()
print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ СХОДИТСЯ: {len(bad)} — ' + '; '.join(bad))
sys.exit(1 if bad else 0)
