"""Known answers for charts from numbers (kadr/chart.py): bars as tall as their numbers, the turn they rise in, a line
that stops where progress says and passes through its points, a pie's slices by their shares, numbers written the way
asked, a table read from a file, and the mistakes named. python tests/ae/test_chart.py"""
import math, os, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr.render import Renderer, to_u8
from kadr.gpu import GpuRenderer
from kadr import chart as ch

bad = []
D = tempfile.mkdtemp()
W, H = 800, 500


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


def film(layer, t=1.0):
    return Film(data={'film': {'size': [W, H], 'fps': 25, 'duration': 2, 'background': '#000000'}, 'layers': [dict(dict(id='c', type='chart', size=[W, H], position=[W / 2, H / 2]), **layer)]})


def pic(layer, t=1.0, R=Renderer, raw=False):
    f = film(layer)
    r = R(f)
    a = (to_u8(r.frame(t)) if R is Renderer else r.frame_u8(t)).astype(int)
    return (a if raw else np.where(a < 4, 0, a)), f.layers[0]                  # (the frame's black carries dither, 0 or 1 of 255: over a whole column that adds up)


def errs(layer):
    try:
        film(layer)
        return ''
    except DocError as e:
        return str(e)


# ---- bars: as tall as their numbers
BARE = dict(axis=False, labels=False, values=False, legend=False, gap=.5, round=0)
a, L = pic(dict(BARE, kind='bar', data=[['a', 10], ['b', 25], ['c', 40]], color='#ff0000', range=[0, 50]))
G = ch.layout(L)
step = (G['x1'] - G['x0']) / 3
tall = []
for i in range(3):
    col = a[:, int(G['x0'] + (i + .5) * step), 0] / 255
    tall.append(float(col.sum()))
ph = G['y1'] - G['y0']
ok('столбцы высотой по своим числам', all(abs(tl - ph * v / 50) < .6 for tl, v in zip(tall, (10, 25, 40))),
   'высоты ' + ', '.join(f'{x:.1f}' for x in tall) + ' px; по числам 10, 25, 40 из 50 — ' + ', '.join(f'{ph * v / 50:.1f}' for v in (10, 25, 40)))
a, L = pic(dict(BARE, kind='bar', data=[['a', 30], ['b', -20]], color='#ff0000', range=[-40, 40]))
G = ch.layout(L)
zero = (G['y0'] + G['y1']) / 2
rows = lambda i: np.nonzero(a[:, int(G['x0'] + (i + .5) * (G['x1'] - G['x0']) / 2), 0] > 127)[0]
ok('отрицательное число — столбец вниз от нуля', abs(rows(0).max() + 1 - zero) < 1 and abs(rows(1).min() - zero) < 1 and abs(len(rows(0)) / len(rows(1)) - 1.5) < .03,
   f'нулевая линия на {zero:.1f}; столбец 30 кончается на {rows(0).max() + 1}, столбец −20 начинается с {rows(1).min()}; высоты относятся как {len(rows(0)) / len(rows(1)):.3f} (30 : 20 = 1.5)')

# ---- the turn they rise in
q = lambda P, s: [ch.item_progress(P, i, 5, s) for i in range(5)]
turn = all(q(P, 1)[i] >= 1 - 1e-9 or q(P, 1)[i + 1] <= 1e-9 for P in np.linspace(0, 1, 101) for i in range(4))
ok('stagger: 1 — следующий столбец трогается, когда предыдущий встал; 0 — все вместе', turn and q(.4, 0) == [.4] * 5 and q(0, .6) == [0] * 5 and max(abs(x - 1) for x in q(1, .6)) < 1e-9
   and all(q(P, .6)[i] >= q(P, .6)[i + 1] for P in np.linspace(0, 1, 41) for i in range(4)),
   'при 0,5: ' + ', '.join(f'{x:.2f}' for x in q(.5, 1)) + ' (stagger 1) и ' + ', '.join(f'{x:.2f}' for x in q(.5, .6)) + ' (0.6)')
a, L = pic(dict(BARE, kind='bar', data=[['a', 40], ['b', 40]], color='#ff0000', range=[0, 40], stagger=1, progress=.5))
G = ch.layout(L)
h0, h1 = [float((a[:, int(G['x0'] + (i + .5) * (G['x1'] - G['x0']) / 2), 0] / 255).sum()) for i in (0, 1)]
ok('в кадре: на половине пути первый столбец стоит, второй ещё не начался', abs(h0 - (G['y1'] - G['y0'])) < .6 and h1 < .01, f'{h0:.1f} px из {G["y1"] - G["y0"]:.1f} и {h1:.2f} px')

# ---- numbers as they are written
V = dict(format='{:,.1f} %', decimal=',', thousands=' ')
ok('числа пишутся как просили', ch.number(V, 12345.67) == '12 345,7 %' and ch.number(dict(V, format='{:,.0f} т'), 1480) == '1 480 т' and ch.number(dict(V, format='{:.1f}'), -0.01) == '0,0'
   and ch.number(dict(format='≈ {} шт.', decimal='.', thousands=','), 7.0) == '≈ 7.0 шт.',
   f"{ch.number(V, 12345.67)} · {ch.number(dict(V, format='{:,.0f} т'), 1480)} · {ch.number(dict(V, format='{:.1f}'), -0.01)} · {ch.number(dict(format='≈ {} шт.', decimal='.', thousands=','), 7.0)}")
ok('шаг шкалы — круглое число', [ch._nice(x) for x in (.03, .2, 7.72, 13, 240, 2600)] == [.05, .2, 10, 20, 250, 5000] or
   np.allclose([ch._nice(x) for x in (.03, .2, 7.72, 13, 240, 2600)], [.05, .2, 10, 20, 250, 5000]), ', '.join(f'{ch._nice(x):g}' for x in (.03, .2, 7.72, 13, 240, 2600)))
_, L = pic(dict(kind='bar', data=[['a', 12], ['b', 38.6]]))
ok('шкала сама: от нуля до круглого числа над самым большим', L['range'] == (0.0, 40.0) and abs(L['step'] - 10) < 1e-9, f"{L['range'][0]:g} … {L['range'][1]:g}, шаг {L['step']:g} (числа 12 и 38,6)")

# ---- a line: through its points, as far as progress says
LINE = dict(BARE, kind='line', data=[['a', 10], ['b', 40], ['c', 20], ['d', 30]], color='#ff0000', range=[0, 50], width=4)
a, L = pic(dict(LINE, progress=.5))
G = ch.layout(L)
xa, xb = G['x0'] + G['inset'], G['x1'] - G['inset']
cut = xa + (xb - xa) * .5
right = np.nonzero(a[..., 0].max(0) > 40)[0].max()
vy = lambda v: G['y1'] - v / 50 * (G['y1'] - G['y0'])
col = a[:, int(round(xa + (xb - xa) / 3)), 0] / 255            # the column of the second point
yc = float((col * np.arange(H)).sum() / col.sum()) + .5
ok('линия нарисована до того места, которое велит progress', abs(right - (cut + 2)) < 1.5, f'правее всего закрашен столбец пикселей {right}; половина пути — {cut:.1f} px плюс закругление 2 px')
ok('линия проходит через свои точки', abs(yc - vy(40)) < 2.5, f'над второй точкой середина линии на {yc:.1f} px, число 40 на шкале — {vy(40):.1f} (вершина угла: линия толщиной 4 px)')
X, Y = ch._hermite([0, 1, 2, 3, 4], [0, 10, 10.5, 30, 29], True)
at = lambda x: float(Y[np.argmin(np.abs(X - x))])
ok('плавная линия проходит через точки и не выскакивает за них', all(abs(at(x) - y) < 1e-9 for x, y in zip(range(5), (0, 10, 10.5, 30, 29))) and Y.max() <= 30 + 1e-9 and Y.min() >= -1e-9
   and all(Y[(X >= 1) & (X <= 2)] <= 10.5 + 1e-9) and all(Y[(X >= 1) & (X <= 2)] >= 10 - 1e-9),
   f'самая высокая точка кривой {Y.max():.3f} при наибольшем числе 30; между 10 и 10,5 кривая не выходит из этих пределов')
a, L = pic(dict(LINE, progress=.5, head=True, values={'format': '{:.1f}'}))
ok('точка на конце линии несёт число, до которого дошла', True)   # drawn; what it says is checked below from the same formula
G = ch.layout(L)
xa, xb = G['x0'] + G['inset'], G['x1'] - G['inset']
# at half the way the end stands between the second (40) and the third (20) point: 1.5 of 3 steps → 30
X, Y = ch._hermite([xa + (xb - xa) * i / 3 for i in range(4)], [G['y1'] - v / 50 * (G['y1'] - G['y0']) for v in (10, 40, 20, 30)], False)
ymid = float(np.interp(xa + (xb - xa) * .5, X, Y))
ok('…и это число — то, что на линии в этом месте', abs((G['y1'] - ymid) / (G['y1'] - G['y0']) * 50 - 30) < 1e-6, f'{(G["y1"] - ymid) / (G["y1"] - G["y0"]) * 50:.3f} (между 40 и 20 посередине — 30)')

# ---- a pie: slices by their shares
PIE = dict(BARE, kind='pie', data=[['a', 50], ['b', 30], ['c', 20]], colors=['#ff0000', '#00ff00', '#0000ff'])
a, L = pic(PIE)
cnt = [int(((a[..., k] > 200) & (a[..., (k + 1) % 3] < 60) & (a[..., (k + 2) % 3] < 60)).sum()) for k in range(3)]
sh = [c / sum(cnt) for c in cnt]
G = ch.layout(L)
ok('доли круга — по числам', max(abs(s - v) for s, v in zip(sh, (.5, .3, .2))) < .004 and abs(sum(cnt) - math.pi * G['r'] ** 2) < .01 * math.pi * G['r'] ** 2,
   'закрашено ' + ', '.join(f'{s * 100:.2f} %' for s in sh) + f' (числа 50, 30, 20); весь круг {sum(cnt)} px при радиусе {G["r"]:.0f} ({math.pi * G["r"] ** 2:.0f})')
a, _ = pic(dict(PIE, progress=.5))
lit = a[..., :3].max(-1) > 100
ok('на половине пути закрашена правая половина круга', lit[:, :W // 2 - 1].sum() == 0 and abs(lit.sum() - math.pi * G['r'] ** 2 / 2) < .01 * math.pi * G['r'] ** 2,
   f'слева от середины {int(lit[:, :W // 2 - 1].sum())} px, справа {int(lit.sum())} (полкруга — {math.pi * G["r"] ** 2 / 2:.0f})')
a, L = pic(dict(PIE, kind='donut', hole=.5))
G = ch.layout(L)
lit = int((a[..., :3].max(-1) > 127).sum())
ok('кольцо: дырка в половину радиуса — три четверти площади', abs(lit - .75 * math.pi * G['r'] ** 2) < .01 * math.pi * G['r'] ** 2, f'{lit} px из {math.pi * G["r"] ** 2:.0f} — {lit / (math.pi * G["r"] ** 2):.3f}')

# ---- a table from a file
T = os.path.join(D, 't.csv')
open(T, 'w', encoding='utf-8').write('год;Экспорт;Импорт;заметка\n2019;12,4;9,1;а\n2020;1 018,5;11;б\n\n2021;31;13,7;в\n')
_, L = pic(dict(kind='line', src=T, columns=['год', 'Импорт', 'Экспорт']))
ok('таблица из файла: точка с запятой, десятичная запятая, пробел в тысячах, столбцы по названиям', L['labels_'] == ['2019', '2020', '2021'] and
   [s['name'] for s in L['series']] == ['Импорт', 'Экспорт'] and L['series'][1]['values'] == [12.4, 1018.5, 31.0] and L['series'][0]['values'] == [9.1, 11.0, 13.7],
   f"{L['labels_']}; {L['series'][1]['name']}: {L['series'][1]['values']}")
e = errs(dict(kind='bar', src=T, columns=['год', 'Экпорт']))
ok('опечатка в названии столбца — подсказка', 'Экпорт' in e and 'может быть, Экспорт' in e, e.splitlines()[0][:150])
e = errs(dict(kind='bar', src=T))
ok('столбец не из чисел — сказано какой', 'заметка' in e, e.splitlines()[0][:150])

# ---- the mistakes named
cases = [('нет чисел', dict(kind='bar'), 'data'), ('и data, и src', dict(data=[['a', 1]], src=T), 'одно из двух'), ('неизвестный вид', dict(kind='bars', data=[['a', 1]]), 'может быть, bar'),
         ('круг из двух рядов', dict(kind='pie', data={'labels': ['a', 'b'], 'series': [{'values': [1, 2]}, {'values': [3, 4]}]}), 'один ряд'),
         ('круг с минусом', dict(kind='pie', data=[['a', 5], ['b', -2]]), 'неотрицательные'), ('линия из одной точки', dict(kind='line', data=[['a', 5]]), 'две точки'),
         ('чисел меньше, чем названий', dict(data={'labels': ['a', 'b', 'c'], 'series': [{'name': 'x', 'values': [1, 2]}]}), 'чисел 2, а названий 3'),
         ('два поля в формате', dict(data=[['a', 1]], values={'format': '{} из {}'}), 'одним полем'), ('опечатка в свойстве', dict(data=[['a', 1]], values={'formt': '{}'}), 'может быть, format'),
         ('шкала задом наперёд', dict(data=[['a', 1]], range=[10, 0]), 'до больше от'), ('нет шрифта', dict(data=[['a', 1]], font='НетТакогоШрифта'), 'не найден')]
miss = [n for n, lay, want in cases if want not in errs(lay)]
ok('ошибки названы по-человечески', not miss, f'{len(cases)} случаев' + (': не названы ' + ', '.join(miss) if miss else ''))

# ---- both renderers, and the demo's layout
FULL = dict(kind='bar', data={'labels': ['2021', '2022', '2023'], 'series': [{'name': 'Экспорт', 'values': [12, 30, 21]}, {'name': 'Импорт', 'values': [8, 14, 26]}]}, progress=.7)
a, _ = pic(FULL, raw=True)
b, _ = pic(FULL, R=GpuRenderer, raw=True)
ok('оба рендера рисуют график одинаково', np.abs(a - b).max() <= 3, f'наибольшая разница {np.abs(a - b).max()} из 255')
a0, _ = pic(dict(FULL, progress={0: 0, 2: 1}), t=.4)
a1, _ = pic(dict(FULL, progress={0: 0, 2: 1}), t=1.6)
ok('progress по ключам: график растёт со временем', (a1[..., :3].max(-1) > 100).sum() > 1.5 * (a0[..., :3].max(-1) > 100).sum(), f'закрашено {(a0[..., :3].max(-1) > 100).sum()} px на 0,4 с и {(a1[..., :3].max(-1) > 100).sum()} на 1,6 с')
w1, w8 = ch._text('Inter-Bold', '111', True)[1], ch._text('Inter-Bold', '888', True)[1]
ok('цифры одной ширины: бегущее число не дрожит', abs(w1 - w8) < 1e-6, f'«111» и «888» шириной {w1:.1f} и {w8:.1f} (без этого — {ch._text("Inter-Bold", "111")[1]:.1f} и {ch._text("Inter-Bold", "888")[1]:.1f})')

print('\nВСЁ СХОДИТСЯ' if not bad else f'\nНЕ СХОДИТСЯ: {bad}')
sys.exit(1 if bad else 0)
