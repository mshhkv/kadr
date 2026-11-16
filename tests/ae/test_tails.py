"""Known answers for the rough edges found on the first real film (16.11.2026): keys inside a list are named,
a text «=» is a text, a gradient stop keeps its #rrggbbaa alpha, the same words stacked are one text for the check,
a layer's measured size is there for formulas, and what an opaque full-frame layer covers is not drawn.
python tests/ae/test_tails.py"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr.render import Renderer, to_u8, content_size
from kadr.gpu import GpuRenderer
from kadr import check

bad = []


def film(layers, comps=None, **f):
    d = {'film': dict({'size': [320, 180], 'fps': 25, 'duration': 2, 'background': '#000000'}, **f), 'layers': layers}
    if comps:
        d['comps'] = comps
    return Film(data=d)


def ok(what, cond, note=''):
    print(f'{"OK  " if cond else "FAIL"} {what}{(": " + note) if note else ""}')
    if not cond:
        bad.append(what)


# 1. keys inside a list: an error in words
try:
    film([dict(id='a', type='solid', color='#ffffff', size=[10, 10], position=[{0: 1, 1: 5}, 0])])
    ok('ключи внутри списка названы ошибкой', False, 'документ принят')
except DocError as e:
    ok('ключи внутри списка названы ошибкой', 'a.position' in str(e) and 'ключи' in str(e), str(e).splitlines()[0][:110])
except Exception as e:
    ok('ключи внутри списка названы ошибкой', False, f'падение: {type(e).__name__}')

try:                                                           # …and so is any value of the wrong kind
    film([dict(id='a', type='text', text='а', font='Arial', size=40, stroke='#000000')])
    ok('цвет на месте числа назван ошибкой', False, 'документ принят')
except DocError as e:
    ok('цвет на месте числа назван ошибкой', str(e).startswith('ERROR a'), str(e).splitlines()[0][:110])
except Exception as e:
    ok('цвет на месте числа назван ошибкой', False, f'падение: {type(e).__name__}')

# 2. «=» is a text; «\=5» is «=5»; «=round(t * 10)» is still a counter
f = film([dict(id='a', type='text', text='=', font='Arial', size=40), dict(id='b', type='text', text='\\=5', font='Arial', size=40),
          dict(id='c', type='text', text='=round(t * 10)', font='Arial', size=40)])
A, B, C = f.layers
ok('текст «=» остаётся знаком', A['text'] == '=' and not A.get('text_fn'), repr(A['text']))
ok('«\\=5» пишется как «=5»', B['text'] == '=5', repr(B['text']))
img = to_u8(Renderer(film([dict(id='c', type='text', text='=round(t * 10)', font='Arial', size=60, position=[160, 90], color='#ffffff')])).frame(1.0))
ok('«=формула» по-прежнему счётчик', img.max() > 200, f'кадр с цифрой, самый яркий пиксель {img.max()}')

# 3. a gradient stop written #rrggbbaa keeps its alpha: red fading out to the right over white
lay = lambda stops: [dict(id='bg', type='solid', color='#ffffff', size=[320, 180], position=[160, 90]),
                     dict(id='g', type='shape', shape='rect', size=[320, 180], position=[0, 0], fill={'gradient': 'linear', 'from': [0, 0], 'to': [320, 0], 'stops': stops})]
a = to_u8(Renderer(film(lay([[0, '#ff0000ff'], [1, '#ff000000']]))).frame(0))
b = to_u8(Renderer(film(lay([[0, '#ff0000', 1], [1, '#ff0000', 0]]))).frame(0))
g = GpuRenderer(film(lay([[0, '#ff0000ff'], [1, '#ff000000']]))).frame_u8(0)
ok('прозрачность точки градиента из #rrggbbaa', a[90, 5, 1] < 12 and a[90, 314, 1] > 243 and 110 < a[90, 160, 1] < 146 and np.abs(a.astype(int) - b).max() <= 1 and np.abs(a.astype(int) - g).max() <= 3,
   f'зелёный канал слева {a[90, 5, 1]}, в середине {a[90, 160, 1]}, справа {a[90, 314, 1]}; с записью третьим числом расходится на {np.abs(a.astype(int) - b).max()}')

# 4. the check: an outline under the same words is one text; other words over them are still a remark
T = dict(type='text', text='ПОДПИСЬ', font='Arial', size=40, position=[160, 90])
twin = [i for i in check.run(film([dict(T, id='o', color='#000000', stroke=6), dict(T, id='f', color='#ffffff')])) if 'перекрываются' in str(i)]
diff = [i for i in check.run(film([dict(T, id='o', color='#ffffff'), dict(T, id='f', color='#ffffff', text='ДРУГОЕ')])) if 'перекрываются' in str(i)]
moved = [i for i in check.run(film([dict(T, id='o', color='#ffffff'), dict(T, id='f', color='#ffffff', position=[190, 100])])) if 'перекрываются' in str(i)]
ok('обводка под тем же текстом — не наезд', not twin, f'замечаний {len(twin)}')
ok('другой текст поверх — по-прежнему наезд', bool(diff), f'замечаний {len(diff)}')
ok('тот же текст со сдвигом — по-прежнему наезд', bool(moved), f'замечаний {len(moved)}')

# 5. a plate sized by its text: layer('t').w / .h is the text's measured box, with its scale
f = film([dict(id='t', type='text', text='Привет, мир', font='Arial', size=40, position=[160, 90], scale={0: 1, 1: 2}),
          dict(id='p', type='shape', shape='rect', size="=[layer('t').w + 40, layer('t').h + 20]", position="=[layer('t').x - layer('t').w / 2 - 20, layer('t').y - layer('t').h / 2 - 10]", fill='#333333')])
w0, h0 = content_size(f, f.layers[0], 0)
s0, s1 = f.layers[1]['size'].at(0), f.layers[1]['size'].at(1)
ok('плашка по размеру текста', abs(s0[0] - (w0 + 40)) < .01 and abs(s0[1] - (h0 + 20)) < .01 and abs(s1[0] - (2 * w0 + 40)) < .01,
   f'текст {w0:.0f}×{h0:.0f}, плашка {s0[0]:.0f}×{s0[1]:.0f}, при масштабе 2 — {s1[0]:.0f}')
img = to_u8(Renderer(f).frame(0))
xs = np.nonzero((np.abs(img[90].astype(int) - 0x33).max(1) < 8) | (img[90].max(1) > 60))[0]
ok('плашка нарисована этой ширины', abs((xs.max() - xs.min() + 1) - (w0 + 40)) < 3, f'{xs.max() - xs.min() + 1} px')
try:
    film([dict(id='t', type='text', text='а', font='Arial', size=40), dict(id='p', type='solid', color='#fff', size="=[layer('t').ww, 5]")])
    ok('опечатка в свойстве слоя названа', False)
except DocError as e:
    ok('опечатка в свойстве слоя названа', 'ww' in str(e), str(e).splitlines()[0][:110])

# 6. what a full-frame opaque layer covers is not drawn — and the frame is the same as when it is
under = [dict(id='u1', type='solid', color='#ff0000', size=[200, 100], position={0: [0, 90], 2: [320, 90]}),
         dict(id='u2', type='text', text='ПОД', font='Arial', size=60, position=[160, 90], color='#ffffff')]
over = [dict(id='top', type='text', text='НАД', font='Arial', size=40, position=[160, 60], color='#00ff00')]
cover = dict(id='cov', type='solid', color='#101830', size=[320, 180], position=[160, 90])


def drawn(mid, t=1.0):
    f = film(under + [mid] + over)
    out = []
    for R in (Renderer(f), GpuRenderer(f)):
        n = len(R.seen(t))
        a = to_u8(R.frame(t)) if type(R) is Renderer else R.frame_u8(t)
        R2 = type(R)(f)
        R2.seen = lambda tt, R2=R2: R2.order(tt)                # the same frame with everything drawn
        b = to_u8(R2.frame(t)) if type(R) is Renderer else R2.frame_u8(t)
        out.append((n, int(np.abs(a.astype(int) - b).max())))
    return out


r = drawn(cover)
ok('под сплошной заливкой во весь кадр слои не рисуются', all(n == 2 and d == 0 for n, d in r), f'рисуется слоёв {r[0][0]} из 4, кадр отличается на {max(d for _, d in r)}')
for name, mid in (('полупрозрачная', dict(cover, opacity=.5)), ('с проявлением', dict(cover, **{'in': .8, 'fade_in': .5})), ('повёрнутая', dict(cover, rotation=10)),
                  ('меньше кадра', dict(cover, size=[300, 180])), ('сдвинутая', dict(cover, position=[170, 90])), ('с маской', dict(cover, mask={'shape': 'ellipse', 'size': [200, 100], 'position': [160, 90]})),
                  ('в режиме наложения', dict(cover, blend='screen')), ('уменьшенная', dict(cover, scale=.9))):
    try:
        r = drawn(mid)
    except DocError as e:
        ok(f'заливка {name} ничего не прячет', False, str(e).splitlines()[0][:100]); continue
    ok(f'заливка {name} ничего не прячет', all(n == 4 and d == 0 for n, d in r), f'рисуется слоёв {r[0][0]} из 4')
G = dict(id='g', type='solid', color='#203050', size=[320, 180], position=[160, 90])
Wd = dict(id='w', type='text', text='ВСТАВКА', font='Arial', size=30, position=[160, 120], color='#ffffff')
for name, comp, n_ in (('под вставкой с глухим фоном слои не рисуются', dict(layers=[G, Wd]), 2), ('под вставкой с цветом фона слои не рисуются', dict(layers=[Wd], background='#203050'), 2),
                       ('вставка без фона ничего не прячет', dict(layers=[Wd]), 4), ('вставка с фоном, который проявляется, ничего не прячет', dict(layers=[dict(G, fade_in=1.5), Wd]), 4),
                       ('вставка с прорезью (stencil) ничего не прячет', dict(layers=[G, dict(Wd, blend='silhouette-alpha')]), 4),
                       ('вставка, которая сама проявляется, ничего не прячет', dict(layers=[G, Wd], fade_in=1.5), 4)):
    r = drawn(dict(comp, id='c', type='comp', position=[160, 90]))
    ok(name, all(n == n_ and d == 0 for n, d in r), f'рисуется слоёв {r[0][0]} из 4, кадр отличается на {max(d for _, d in r)}')

# what lies under a cover but works for a layer above it — a matte, an effect's map, a parent — still does
def above(layers, t=.5):
    f = film(layers)
    out = []
    for R in (Renderer(f), GpuRenderer(f)):
        a = to_u8(R.frame(t)) if type(R) is Renderer else R.frame_u8(t)
        R2 = type(R)(f)
        R2.seen = lambda tt, R2=R2: R2.order(tt)
        b = to_u8(R2.frame(t)) if type(R) is Renderer else R2.frame_u8(t)
        out.append((int(np.abs(a.astype(int) - b).max()), int(((a[..., 0] > 200) & (a[..., 1] < 60)).sum())))
    return out


for name, layers, red in (('мат', [dict(id='m', type='shape', shape='ellipse', size=[80, 80], anchor=[40, 40], position=[160, 90], fill='#ffffff'), cover,
                                   dict(id='r', type='solid', color='#ff0000', size=[200, 120], position=[160, 90], matte={'src': 'm', 'mode': 'alpha'})], 4000),
                          ('карта эффекта', [dict(id='pic', type='gradient', kind='linear', angle=0, stops=[[0, '#000000'], [1, '#ffffff']], visible=False), cover,
                                             dict(id='r', type='solid', color='#ff0000', size=[200, 120], position=[160, 90], effects=[{'type': 'displace', 'map': 'pic', 'amount': [20, 0]}])], 20000),
                          ('родитель', [dict(id='p', type='solid', color='#00ff00', size=[20, 20], position={0: [40, 40], 1: [280, 140]}), cover,
                                        dict(id='r', type='solid', color='#ff0000', size=[30, 30], position=[0, 0], parent='p')], 800)):
    r = above(layers)
    ok(f'{name} под глухой заливкой работает для слоя над ней', all(d == 0 and n > red for d, n in r), f'кадр тот же, красных пикселей {r[0][1]}')

# found on the night of 16.11 while trying the new layers together
def said(**f):
    try:
        film([dict(id='a', type='solid', color='#ff0000', size=[30, 30], position=[100, 90])], **f)
        return ''
    except DocError as e:
        return str(e)


e1, e2 = said(motion_blur=0.5), said(size=320)
ok('раздел film: значение не того вида — слова, а не разбор по строкам Python', 'motion_blur' in e1 and 'true' in e1 and 'не того вида' in e2 and 'Traceback' not in e1 + e2,
   e1[:70] + ' | ' + e2[:60])
turning = dict(id='g', type='solid', color='#ffffff', size=[120, 40], position=[160, 90])
w = [str(i) for i in check.run(film([dict(turning, rotation='=40 * sin(3 * t)')]))]
ok('слой, плавно качающийся через ноль градусов, — не рывок', not any('рывок' in x for x in w), (w or ['замечаний нет'])[0][:90])
w = [str(i) for i in check.run(film([dict(turning, rotation={0: 0, 1: 0, 1.04: 60, 2: 60})]))]
ok('…а поворот рывком — рывок', any('рывок' in x for x in w), (w or ['—'])[0][:90])
w = [str(i) for i in check.run(film([dict(id='cam', type='camera', position=[160, 90, -400], zoom=400), dict(id='p', type='solid', color='#00ff00', size=[20, 20], position=[100, 90]),
                                     dict(id='c', type='solid', color='#ff0000', size=[30, 30], position=[20, 0], parent='p', z=50),
                                     dict(id='q', type='solid', color='#0000ff', size=[20, 20], position=[200, 90], z=80),
                                     dict(id='d', type='solid', color='#ff00ff', size=[10, 10], position=[10, 0], parent='q')]))]
ok('parent вместе с z под камерой не работает — сказано для обоих случаев', any('c:' in x and 'родитель не действует' in x for x in w) and any('d:' in x and 'стоит в глубине' in x for x in w),
   ' | '.join(x[:58] for x in w if 'родител' in x))

print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
