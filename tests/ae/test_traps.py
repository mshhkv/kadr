"""Silent traps a film's author walked into on 16.11.2026 (five reels made with Kadr by an agent) — each either cannot
happen any more or is said in words by `kadr check`:
  a comp's layers without `out` vanished after 5 s · `time: =t` on a comp gave an empty frame · `filter` before a
  colour key · a radial gradient's centre written in px · a rect placed by its corner where a picture is placed by
  its middle · the MCP server silent for half an hour (a renderer asked for a frame from another thread) ·
  characters a font lacks drawn as boxes.   python tests/ae/test_traps.py"""
import os, sys, tempfile, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np
import yaml
from kadr.doc import Film, DocError
from kadr.check import run as check
from kadr.gpu import GpuRenderer

bad = []


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


BASE = {'size': [320, 320], 'fps': 25, 'duration': 12, 'background': '#000000'}


def film(layers, **f):
    return Film(data={'film': {**BASE, **f}, 'layers': layers})


def words(layers, **f):
    try:
        return [str(i) for i in check(film(layers, **f))]
    except DocError as e:
        return ['REFUSED ' + str(e)]


# ---- 1, 2. a comp has no end of its own
box = [dict(id='box', type='shape', shape='rect', size=[200, 200], position=[60, 60], fill='#ff0000')]
for what, extra in (('без time', {}), ('time: =t', {'time': '=t'}), ('time: 3 (точка входа)', {'time': 3}), ('speed: 2', {'speed': 2})):
    F = film([dict(id='c', type='comp', size=[320, 320], position=[160, 160], layers=box, **extra)])
    r = GpuRenderer(F)
    red = [int(r.frame_u8(t)[160, 160, 0]) for t in (1, 4.9, 5.1, 9, 11.9)]
    ok(f'вставка {what}: слой без out виден всю её жизнь', red == [255] * 5 and not F.issues, str(red))
F = film([dict(id='c', type='comp', size=[320, 320], position=[160, 160], layers=[{**box[0], 'out': 2}])])
r = GpuRenderer(F)
ok('слой вставки со своим out кончается на нём', [int(r.frame_u8(t)[160, 160, 0]) for t in (1, 1.9, 2.1, 9)] == [255, 255, 0, 0])
F = film([dict(id='c', type='comp', size=[320, 320], position=[160, 160], out=6, layers=box)])
r = GpuRenderer(F)
ok('а сама вставка кончается на своём out', [int(r.frame_u8(t)[160, 160, 0]) for t in (5.9, 6.1)] == [255, 0])
F = film([dict(id='c', type='comp', size=[320, 320], position=[160, 160], layers=[{**box[0], 'fade_out': 0.5}])])
ok('fade_out у слоя вставки без out не гасит его', int(GpuRenderer(F).frame_u8(11.9)[160, 160, 0]) == 255)
d = tempfile.mkdtemp()
yaml.safe_dump({'film': {'size': [320, 320], 'duration': 3}, 'layers': box}, open(os.path.join(d, 'in.yaml'), 'w'))
yaml.safe_dump({'film': BASE, 'layers': [dict(id='c', type='comp', src='in.yaml', size=[320, 320], position=[160, 160])]}, open(os.path.join(d, 'f.yaml'), 'w'))
r = GpuRenderer(Film(os.path.join(d, 'f.yaml')))
ok('вставка из файла со своим duration кончается на нём', [int(r.frame_u8(t)[160, 160, 0]) for t in (2.9, 3.1)] == [255, 0])

# ---- 3. filter comes before effects
green = dict(id='s', type='solid', color='#00ff00', size=[100, 100], effects=[{'type': 'chroma_key', 'key': '#00ff00'}])
w = words([{**green, 'filter': {'saturation': 0}}])
ok('saturation в filter перед chroma_key — предупреждение с выходом', len(w) == 1 and 'раньше эффектов' in w[0] and 'tint' in w[0], '; '.join(w))
ok('sepia — тоже', any('раньше эффектов' in x for x in words([{**green, 'filter': {'sepia': 1}}])))
ok('contrast перед ключом — без замечаний', words([{**green, 'filter': {'contrast': 1.1}}]) == [])
ok('saturation без ключа — без замечаний', words([{k: v for k, v in green.items() if k != 'effects'} | {'filter': {'saturation': 0}}]) == [])

# ---- 4. a gradient's centre and radius are shares of its box
grad = dict(id='g', type='gradient', kind='radial', stops=[[0, '#ffffff'], [1, '#000000']])
w = words([{**grad, 'center': [160, 160], 'radius': [100, 100]}])
ok('center в пикселях — предупреждение', len(w) == 1 and 'долями' in w[0], '; '.join(w))
ok('radius в пикселях — предупреждение', any('radius' in x and 'долями' in x for x in words([{**grad, 'radius': [120, 120]}])))
ok('center ключами в пикселях — предупреждение', any('долями' in x for x in words([{**grad, 'center': {0: [0.5, 0.5], 1: [200, 160]}}])))
ok('доли (и слегка за рамкой) — без замечаний', words([{**grad, 'center': [1.4, -0.3], 'radius': [0.8, 2.5]}]) == [])
ok('линейному градиенту center не важен', words([{**grad, 'kind': 'linear', 'center': [160, 160]}]) == [])

# ---- 5. a rect by its middle
rect = dict(id='r', type='shape', shape='rect', size=[100, 60], position=[160, 160], fill='#ff0000')
def span(L):
    a = GpuRenderer(film([L])).frame_u8(0)
    ys, xs = np.nonzero(a[..., 0] > 128)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
ok('rect без anchor стоит углом в position', span(rect) == (160, 259, 160, 219), str(span(rect)))
ok('anchor: center ставит его серединой', span({**rect, 'anchor': 'center'}) == (110, 209, 130, 189), str(span({**rect, 'anchor': 'center'})))
ok('и это то же, что anchor числами', span({**rect, 'anchor': 'center'}) == span({**rect, 'anchor': [50, 30]}))
ok('ellipse — так же', span({**rect, 'shape': 'ellipse', 'anchor': 'center'}) == span({**rect, 'shape': 'ellipse', 'anchor': [50, 30]}))
ok('повторитель крутит копии вокруг той же середины', span({**rect, 'anchor': 'center', 'repeat': {'count': 2, 'rotation': 90}}) == span({**rect, 'anchor': [50, 30], 'repeat': {'count': 2, 'rotation': 90}}))
w = words([dict(id='p', type='shape', shape='polygon', points=[[0, 0], [9, 0], [9, 9]], position=[160, 160], anchor='center', fill='#ff0000')])
ok('center у многоугольника — ошибка словами', len(w) == 1 and w[0].startswith('REFUSED') and 'rect и ellipse' in w[0], '; '.join(w))
w = words([{**rect, 'anchor': 'middle'}])
ok('неизвестное слово в anchor — ошибка словами', w[0].startswith('REFUSED') and 'center' in w[0])

# ---- a fly-in is not «the layer is outside the frame»
fly = dict(id='f', type='solid', color='#ffffff', size=[60, 60])
w = words([{**fly, 'position': {0: [-200, 160], 0.4: [160, 160], 'ease': 'out'}}])
ok('влёт из-за кадра (0,2 с снаружи в начале) — без замечаний', not any('за кадром' in x for x in w), '; '.join(w))
w = words([{**fly, 'position': {0: [160, 160], 11.6: [160, 160], 12: [600, 160]}}])
ok('вылет за кадр в конце — без замечаний', not any('за кадром' in x for x in w), '; '.join(w))
w = words([dict(id='g', type='group', out=6, position={0: [160, 160], 5.6: [160, 160], 6: [160, 900]}), {**fly, 'position': [160, 160], 'parent': 'g'}])
ok('ребёнок группы, улетающей за кадр перед своим концом, — без замечаний', not any('за кадром' in x for x in w), '; '.join(w))
w = words([{**fly, 'position': {0: [160, 160], 4: [160, 160], 4.2: [900, 160], 6: [900, 160], 6.2: [160, 160]}}])
ok('ушёл за кадр посреди жизни — замечание', any('за кадром' in x for x in w))
w = words([{**fly, 'position': [900, 160]}])
ok('всю жизнь за кадром — замечание', any('за кадром' in x for x in w))
w = words([{**fly, 'position': {0: [-200, 160], 3: [-200, 160], 3.4: [160, 160]}}])
ok('три секунды снаружи перед влётом — замечание', any('за кадром' in x for x in w))

# ---- 7. characters no font has
text = dict(id='t', type='text', font='Inter-Bold', size=30, position=[160, 160])
w = words([{**text, 'text': 'Цена → 100 ₽ ☃ ☃'}])
ok('знак, которого нет в шрифте, назван один раз, с кодом', len(w) == 1 and '«☃» (U+2603)' in w[0] and w[0].count('☃') == 1 and '→' not in w[0].split('знаков')[1].split('—')[0], '; '.join(w))
ok('выход назван: запасной шрифт списком', 'font: [Inter-Bold, <шрифт, где эти знаки есть>]' in w[0])
sysf = next((p for p in ('/System/Library/Fonts/Supplemental/Courier New.ttf', '/System/Library/Fonts/Supplemental/Impact.ttf', 'C:/Windows/Fonts/impact.ttf') if os.path.exists(p)), None)
if sysf:                                                       # a font without «→»: Inter, which ships with Kadr, has it — named as the way out
    w2 = words([{**text, 'font': sysf, 'text': 'A → B'}])
    ok('если знак есть в Inter из поставки — он и предложен', (not w2) or 'Inter-Regular] — в Inter-Regular (идёт с Kadr) они есть' in w2[0], '; '.join(w2)[:200])
ok('обычный текст (кириллица, перенос строки, пробелы) — без замечаний', words([{**text, 'text': 'Цена:\n100 ₽ → «да»'}]) == [])
w = words([{**text, 'text': '="☃ " + str(int(t))'}])
ok('текст-формула: знаки спрошены по ходу фильма', any('U+2603' in x for x in w), '; '.join(w))
ok('счётчик (текст-формула) не считается рывком оттого, что цифры разной ширины', not any('рывок' in x for x in w))
w = words([dict(id='o', type='shape', shape='text', text='A☃', font='Inter-Bold', font_size=40, position=[100, 160], fill='#ffffff')])
ok('контуры букв (shape: text) — тоже', any('U+2603' in x for x in w), '; '.join(w))
sym = next((p for p in ('/System/Library/Fonts/Apple Symbols.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/seguisym.ttf') if os.path.exists(p)), None)
if sym:
    ok('запасной шрифт, где знак есть, снимает замечание', words([{**text, 'text': 'Цена ☃', 'font': ['Inter-Bold', sym]}]) == [])

# ---- 6. a renderer belongs to the thread that made it
F = film(box)
made, said = {}, {}
def make():
    made['r'] = GpuRenderer(F)
    made['a'] = made['r'].frame_u8(0)
def ask():
    try:
        made['r'].frame_u8(0)
        said['x'] = 'кадр'
    except RuntimeError as e:
        said['x'] = str(e)
    except BaseException as e:                                 # the core's panic: what it was before
        said['x'] = 'PANIC ' + type(e).__name__
for fn in (make, ask):
    th = threading.Thread(target=fn)
    th.start()
    th.join()
ok('кадр у рендерера из другого потока — отказ словами, а не падение ядра', 'в другом потоке' in said.get('x', ''), said.get('x', '')[:90])

try:
    import mcp                                                 # noqa: F401 — the server is an optional part (pip install "kadr[serve]")
except ImportError:
    mcp = None
if mcp is not None:
    from kadr import serve
    from mcp.server.mcpserver.exceptions import ToolError

    seen = []
    @serve.on_gpu
    def where():
        seen.append(threading.current_thread().name)
        return 'ok'
    ths = [threading.Thread(target=where) for _ in range(4)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    ok('сервер: вызовы из четырёх потоков исполнены в одном своём', len(seen) == 4 and len(set(seen)) == 1 and seen[0].startswith('kadr-gpu'), str(set(seen)))

    class Panic(BaseException):
        pass
    @serve.on_gpu
    def dies():
        raise Panic('Cannot get non-existent resource BufferId(25,2)')
    serve._films['x'] = 1
    try:
        dies()
        got = 'ничего'
    except ToolError as e:
        got = str(e)
    ok('сервер: сбой ядра возвращается словами', 'ядро Kadr остановилось' in got and 'BufferId' in got, got[:80])
    ok('…и сервер начинает с чистого листа, следующий вызов работает', not serve._films and where() == 'ok')
    @serve.on_gpu
    def plain():
        raise ToolError('нет файла')
    try:
        plain()
        got = ''
    except ToolError as e:
        got = str(e)
    ok('сервер: обычная ошибка инструмента проходит как была', got == 'нет файла')

    # the real server over stdio: overlapping calls (the library spreads them over its worker threads) all answer
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    doc = os.path.join(d, 'srv.yaml')
    yaml.safe_dump({'film': BASE, 'layers': box + [dict(id='t', type='text', text='=str(int(t * 10))', font='Inter-Bold', size=60, position=[160, 160])]}, open(doc, 'w'))
    got = {}

    async def main():
        params = StdioServerParameters(command=sys.executable, args=['-m', 'kadr', 'serve'], env={**os.environ, 'PYTHONPATH': ROOT})
        async with stdio_client(params) as (rd, wr):
            async with ClientSession(rd, wr) as s:
                await s.initialize()

                async def one(key, t):
                    res = await s.call_tool('frame', {'film': doc, 'at': t, 'width': 160})
                    got[key] = [c.type for c in res.content]
                with anyio.move_on_after(120):
                    for rnd in range(3):
                        async with anyio.create_task_group() as tg:
                            for i in range(6):
                                tg.start_soon(one, (rnd, i), str(0.5 + rnd + i))
    t0 = time.time()
    try:
        anyio.run(main)
        err = ''
    except BaseException as e:                                 # the server died: the old behaviour
        err = type(e).__name__
    ok('настоящий сервер: 18 перекрывающихся вызовов frame — 18 ответов с картинкой', len(got) == 18 and all(v == ['text', 'image'] for v in got.values()) and not err,
       f'{len(got)} ответов за {time.time() - t0:.0f} с {err}')
else:
    print('     (пакет mcp не установлен — проверки сервера пропущены)')

print()
print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ СХОДИТСЯ: {len(bad)} — ' + '; '.join(bad))
sys.exit(1 if bad else 0)
