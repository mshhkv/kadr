"""A film is a document, not a program (kadr/safe.py and the gates in kadr/doc.py): documents written to run code, to
write or read files they have no business with, or to reach the network are refused — each attack here was a working
one before 16.11.2026 — and honest films are not. python tests/ae/test_safe.py"""
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
D = tempfile.mkdtemp()
os.environ['HOME'] = os.path.join(D, 'home')                   # a machine where no folder is trusted
os.environ.pop('KADR_TRUST', None)
import yaml
from kadr.doc import Film, DocError
from kadr import safe, plugins
plugins.TRUST_FILE = os.path.join(D, 'home', '.kadr', 'trusted')

bad = []
MARK = os.path.join(D, 'ran')


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def said(data=None, path=None):
    """what the film says when read: (was it refused, its words)"""
    try:
        F = Film(path) if path else Film(data=data)
        return False, '; '.join(str(i) for i in F.issues)
    except DocError as e:
        return True, str(e)


BASE = {'size': [64, 64], 'fps': 25, 'duration': 1}
solid = lambda **kw: dict(id='s', type='solid', color='#fff', size=[4, 4], **kw)
run = f"system('touch {MARK}')"

# ---- 1. formulas only count
for what, expr in (('скрытые поля функции', f"=wiggle.__globals__['os'].{run}"), ('подклассы object', "=().__class__.__base__.__subclasses__()"), ('слой как дверь к фильму', "=layer('s')._film"),
                   ('имя с подчёркивания', f"=__import__('os').{run}"), ('перебор в скобках', "=[x for x in (1,)][0]"), ('кадр генератора', "=(x for x in ()).gi_frame"),
                   ('поля через format', "='{0._film}'.format(layer('s'))"), ('format не у своей строки', "=str.format('{0.__class__}', 1)"), (':=', "=(a := 1)"),
                   ('метод, пишущий файл', "=t.tofile('/tmp/x')"), ('f-строка с полем', "=f'{t.__class__}'")):
    refused, words = said({'film': BASE, 'layers': [solid(opacity=expr)]})
    ok(f'формула: {what} — отказ', refused and not os.path.exists(MARK), words.splitlines()[0][:110])
F = Film(data={'film': {**BASE, 'vars': {'label': "='{:.1f} %'.format(t * 30).zfill(7)"}},
               'layers': [solid(opacity="=(lambda a: a * a * (3 - 2 * a))(clamp(sin(t * 3) * .5 + .5, 0, 1))", position="=[layer('s').x + 0, layer('s').at(t - .1).position[1]]")]})
want = (lambda a: a * a * (3 - 2 * a))(__import__('math').sin(1.5) * .5 + .5)
ok('честные формулы считаются: лямбда, слой, .format, строковые методы', abs(F.layers[0]['opacity'].at(.5) - want) < 1e-9 and F.var('label', .5) == '015.0 %',
   f"{F.layers[0]['opacity'].at(.5):.4f}, «{F.var('label', .5)}»")
refused, words = said({'film': BASE, 'layers': [dict(id='s', type='scene3d', scene=dict(nodes=[dict(id='p', shape='plane', size=[4, 4], segments=[4, 4], relief="x.__class__.__init__.__globals__")]))]})
ok('рельеф листа — тоже только счёт', refused, words.splitlines()[0][:100])

# ---- 2. a figure's names are names
for key in ('pose', 'voice', 'id'):
    node = dict(id='bob', shape='figure')
    node[key] = f"x', 'lid') + clamp.__globals__['__builtins__']['__import__']('os').{run} * 0 + pose('x"
    refused, words = said({'film': BASE, 'layers': [dict(id='s', type='scene3d', scene=dict(nodes=[node]))]})
    ok(f'фигурка: код в поле {key} — отказ, ничего не запущено', refused and not os.path.exists(MARK), words.splitlines()[0][:100])

# ---- 3. sound filters by name
wav = os.path.join(HERE, 'beat.wav')
for what, af in (('запись файла', f"ametadata=mode=add:key=x:value=1,ametadata=mode=print:file={MARK}"), ('чтение файла как звука', "amovie=filename=/etc/hosts"), ('адрес в сети', "amovie=http\\://127.0.0.1\\:9/x"),
                 ('имя файла в значении', "volume=volume=/etc/hosts"), ('вложенный граф', "volume=1[a];[a]amovie=x")):
    refused, words = said({'film': {**BASE, 'audio': [dict(src=wav, audio_filter=af)]}, 'layers': []})
    ok(f'audio_filter: {what} — отказ', refused, words.splitlines()[0][:100])
refused, words = said({'film': {**BASE, 'audio': [dict(src=wav, audio_filter="highpass=f=320,lowpass=f=3400,acompressor=threshold=-18dB:ratio=4,volume=2dB,aecho=0.8:0.9:60|90:0.4|0.3")]}, 'layers': []})
ok('честная цепочка фильтров принята', not refused, words[:100])
from kadr import audio
try:
    audio.decode(wav, 0, .1, 1.0, f"ametadata=mode=print:file={MARK}")
    ok('и сам декодер не возьмёт чужой фильтр', False)
except safe.Refused:
    ok('и сам декодер не возьмёт чужой фильтр', not os.path.exists(MARK))

# ---- 4. a film from a folder nobody trusted: its own folder, and no code
theirs, secret = os.path.join(D, 'downloaded'), os.path.join(D, 'secret')
os.makedirs(os.path.join(theirs, 'plugins')); os.makedirs(secret)
open(os.path.join(secret, 'key.csv'), 'w').write('token\nSECRET-0123\n')
open(os.path.join(secret, 'words.json'), 'w').write('[{"word": "SECRET", "start": 0, "end": 1}]')
__import__('shutil').copy(os.path.join(HERE, 'card.png'), os.path.join(secret, 'photo.png'))
__import__('shutil').copy(os.path.join(HERE, 'card.png'), os.path.join(theirs, 'own.png'))
os.symlink(os.path.join(secret, 'photo.png'), os.path.join(theirs, 'link.png'))
open(os.path.join(theirs, 'plugins', 'x.py'), 'w').write(f"open({MARK!r}, 'w').write('ran')\n")


def film(name, body):
    p = os.path.join(theirs, name)
    yaml.safe_dump(body, open(p, 'w', encoding='utf-8'), allow_unicode=True)
    return p


img = lambda src: {'film': BASE, 'layers': [dict(id='i', type='image', src=src, size=[32, 32], position=[32, 32])]}
ok('свой файл рядом с фильмом читается', not said(path=film('a.yaml', img('own.png')))[0])
for what, src in (('полный путь', os.path.join(secret, 'photo.png')), ('../', '../secret/photo.png'), ('ссылка из своей папки наружу', 'link.png')):
    refused, words = said(path=film('b.yaml', img(src)))
    ok(f'чужой фильм: файл вне его папки ({what}) — не прочитан, сказано почему', refused and 'kadr trust' in words, words.splitlines()[0][:90])
refused, words = said(path=film('c.yaml', {'film': {**BASE, 'root': secret}, 'layers': [dict(id='i', type='image', src='photo.png', size=[32, 32], position=[32, 32])]}))
ok('и через root: чужую папку своей не назвать', refused and 'kadr trust' in words)
refused, words = said(path=film('d.yaml', {'film': BASE, 'layers': [dict(id='c', type='chart', src=os.path.join(secret, 'key.csv'), columns=['zz'], size=[60, 60], position=[32, 32])]}))
ok('таблица вне папки: её строки не попали в отчёт', refused and 'SECRET' not in words and 'token' not in words, words.splitlines()[0][:80])
refused, words = said(path=film('e.yaml', {'film': {**BASE, 'words': os.path.join(secret, 'words.json')}, 'layers': []}))
ok('тайминги слов вне папки не читаются', refused)
refused, words = said(path=film('f.yaml', {'film': {**BASE, 'plugins': ['plugins']}, 'layers': []}))
ok('код из папки чужого фильма не запущен, сказано как доверить', refused and not os.path.exists(MARK) and 'kadr trust' in words, words.splitlines()[0][:90])
plugins.trust(theirs)                                          # the user reads it and says: mine
plugins.REFUSED.clear()
ok('после kadr trust тот же фильм читает всё и подключает свой код', not said(path=film('g.yaml', img(os.path.join(secret, 'photo.png'))))[0] and not said(path=film('f.yaml', {'film': {**BASE, 'plugins': ['plugins']}, 'layers': []}))[0]
   and os.path.exists(MARK))
os.remove(MARK)
ok('фильм, собранный в коде, — своего автора: читает где угодно', not said(img(os.path.join(secret, 'photo.png')))[0])

# ---- 5. a model's files stay beside the model
from kadr import gltf
import json
g = os.path.join(D, 'm.gltf')
json.dump({'asset': {'version': '2.0'}, 'buffers': [{'uri': '../secret/key.csv', 'byteLength': 8}], 'scenes': [{'nodes': []}], 'nodes': []}, open(g, 'w'))
os.makedirs(os.path.join(D, 'mdl')); os.replace(g, os.path.join(D, 'mdl', 'm.gltf'))
try:
    gltf.load(os.path.join(D, 'mdl', 'm.gltf'))
    ok('модель не читает файл вне своей папки', False)
except ValueError as e:
    ok('модель не читает файл вне своей папки', 'вне своей папки' in str(e))

# ---- 6. sizes that would eat the machine
for what, f in (('кадр 200000×200000', {**BASE, 'size': [200000, 200000]}), ('100000 кадров в секунду', {**BASE, 'fps': 100000})):
    refused, words = said({'film': f, 'layers': []})
    ok(f'{what} — отказ словами', refused and 'допустимо' in words)
loop = os.path.join(theirs, 'loop.yaml')
yaml.safe_dump({'film': BASE, 'layers': [dict(id='c', type='comp', src='loop.yaml', size=[64, 64], position=[32, 32])]}, open(loop, 'w'))
refused, words = said(path=loop)
ok('вставка, включающая саму себя, — ошибка, а не зависание', refused and 'саму себя' in words)

print()
print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ СХОДИТСЯ: {len(bad)} — ' + '; '.join(bad))
sys.exit(1 if bad else 0)
