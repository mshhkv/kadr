"""Charts from numbers (After Effects: expressions over a data file, chart plugins) — the `chart` layer draws bars, a
line, an area or a pie from the numbers given, and grows it with one number, `progress`.

    - id: growth
      type: chart
      size: [1400, 700]              the layer's plane: the axis, the names and the numbers are fitted inside it
      kind: bar                      bar | hbar (lying bars) | line | area | pie | donut
      data: [[2019, 12], [2020, 18.5], [2021, 31]]                  a name and its value…
      data: {labels: [2019, 2020], series: [{name: Экспорт, values: [12, 18], color: "#e33"}, {name: Импорт, values: [9, 14]}]}
      src: sales.csv                 …or a table: the first column is the names, the others — the series, the first
      columns: [year, sales]         row their names; columns picks them by name or number (the names' one first)
      progress: {0: 0, 2: 1, ease: out}      0 … 1: bars rise in turn, a line is drawn left to right, a pie sweeps round
      stagger: 0.6                   how strictly in turn the bars go (0 — all together, 1 — one ends before the next starts)
      color: "#3aa0ff"               one colour for all (or a gradient);  colors: ["#…", …] — one a series; with one
                                     series — one a bar or a slice (so one bar can stand out)
      range: [0, 40]                 the value axis (by default from 0, or the lowest value below it, to a round number above the highest)
      axis: {ticks: 5, grid: true, color: "#ffffff55", width: 2, format: "{:.0f}"}    the scale and its lines;  axis: false — none
      labels: {size: 28, color: "#cccccc"}                          the names;  labels: false — none
      values: {format: "{:,.1f} %", size: 30, decimal: ",", thousands: " "}     the numbers, counting up as their bar
                                     grows (format: Python's, one field);  values: false — none
      font: Inter   weight: 600      for all the text of the chart (digits are set at one width, so a counting number does not shake)
      width: 6   dots: 7   smooth: true   head: true     line, area: the line's width, its points, a curve instead of
                                     corners, a dot that rides the line's end with the value it has reached
      fill: 0.3                      area: how opaque it is under the line
      gap: 0.3   round: 8            bars: the gap's share of a step; the corner's radius
      hole: 0.6                      donut: the hole's share of the radius
      legend: true                   the names of the series with their colours (it is on by itself when there are several)

A chart is a picture drawn in its own plane and then placed like any layer: position, scale, opacity, effects,
parents and the 3D camera work on it as on a photograph."""
import csv, math, os, re

import numpy as np

KINDS = ('bar', 'hbar', 'line', 'area', 'pie', 'donut')
PALETTE = ('#4c9aff', '#ff6b5e', '#ffc447', '#45d09e', '#b085ff', '#ff8ad0', '#7fd6ff', '#c8d04b')
KEYS = {'size', 'kind', 'data', 'src', 'columns', 'progress', 'stagger', 'color', 'colors', 'range', 'axis', 'labels', 'values', 'font', 'weight',
        'width', 'dots', 'smooth', 'head', 'fill', 'gap', 'round', 'hole', 'legend'}
FIELD = re.compile(r'\{(:[0-9,.+\- <>^=#_a-zA-Z%]*)?\}')


def _num(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).strip().replace(' ', '').replace(' ', '').replace('%', '')
    if re.fullmatch(r'-?\d+,\d+', s):                          # 12,5 — a decimal comma
        s = s.replace(',', '.')
    return float(s.replace(',', ''))


def _table(F, L, lid):
    """the file's rows → labels and series"""
    path = F.resolve(str(L['src']))
    if not os.path.exists(path):
        F.issue('error', f'нет файла {L["src"]}', lid)
        return [], []
    text = open(path, encoding='utf-8-sig').read()
    try:
        dialect = csv.Sniffer().sniff(text[:4000], delimiters=',;\t')
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(text.splitlines(), dialect) if any(c.strip() for c in r)]
    if len(rows) < 2:
        F.issue('error', f'{L["src"]}: нужна строка названий и хотя бы одна строка чисел', lid)
        return [], []
    head, body = [c.strip() for c in rows[0]], rows[1:]
    cols = L.get('columns') or list(range(1, len(head) + 1))
    idx = []
    for c in cols if isinstance(cols, (list, tuple)) else [cols]:
        if isinstance(c, int) and 1 <= c <= len(head):
            idx.append(c - 1)
        elif str(c) in head:
            idx.append(head.index(str(c)))
        else:
            F.issue('error', f'columns: в {L["src"]} нет столбца «{c}»; есть: ' + ', '.join(head), lid, str(c), set(head))
    if len(idx) < 2:
        if not F.issues or 'columns' not in F.issues[-1].msg:
            F.issue('error', 'columns: столбец названий и хотя бы один столбец чисел', lid)
        return [], []
    labels = [r[idx[0]].strip() if idx[0] < len(r) else '' for r in body]
    series = []
    for j in idx[1:]:
        try:
            series.append(dict(name=head[j], values=[_num(r[j]) for r in body]))
        except (ValueError, IndexError):
            F.issue('error', f'{L["src"]}: в столбце «{head[j]}» не всё числа', lid)
    return labels, series


def _text_spec(F, L, lid, key, default_on, size, col):
    v = L.get(key, default_on)
    if v is False or v is None:
        return None
    v = {} if v is True else v
    if not isinstance(v, dict):
        F.issue('error', f'{key}: true, false или {{size, color, …}}', lid)
        v = {}
    allowed = {'size', 'color'} | ({'format', 'decimal', 'thousands'} if key in ('values', 'axis') else set()) | ({'ticks', 'grid', 'width'} if key == 'axis' else set())
    for k in v:
        if k not in allowed:
            F.issue('error', f'{key}: неизвестное «{k}»; есть: ' + ', '.join(sorted(allowed)), lid, k, allowed)
    return dict(size=float(v.get('size', size)), color=F.paint(v.get('color', col), f'{lid}.{key}.color'), **{k: v[k] for k in v if k not in ('size', 'color')})


def parse(F, L, lid, out):
    s = L.get('size', [F.W, F.H])
    if not (isinstance(s, (list, tuple)) and len(s) == 2):
        F.issue('error', 'size: нужно [w, h] — плоскость слоя', lid)
        s = [F.W, F.H]
    out['psize'] = (float(s[0]), float(s[1]))
    out['cell'] = 16.0
    kind = L.get('kind', 'bar')
    if kind not in KINDS:
        F.issue('error', f'kind «{kind}»: нужно ' + ' | '.join(KINDS), lid, kind, set(KINDS))
        kind = 'bar'
    out['kind'] = kind
    # ---- the numbers
    labels, series = [], []
    if ('data' in L) == ('src' in L):
        F.issue('error', 'нужно одно из двух: data (числа в документе) или src (таблица .csv)', lid)
    elif 'src' in L:
        labels, series = _table(F, L, lid)
    else:
        d = L['data']
        try:
            if isinstance(d, dict):
                labels = [str(x) for x in d.get('labels', [])]
                for i, sr in enumerate(d.get('series', [])):
                    series.append(dict(name=str(sr.get('name', f'{i + 1}')), values=[_num(x) for x in sr['values']], color=sr.get('color')))
            else:
                labels = [str(r[0]) for r in d]
                series = [dict(name='', values=[_num(r[1]) for r in d])]
        except (TypeError, ValueError, KeyError, IndexError):
            F.issue('error', 'data: [[название, число], …] или {labels: […], series: [{name, values: […]}, …]}', lid)
            labels, series = [], []
    n = len(labels)
    if n < 1 or not series:
        if not any(i.where == lid for i in F.issues):
            F.issue('error', 'в графике нет ни одного числа', lid)
        labels, series, n = ['—'], [dict(name='', values=[0.0])], 1
    for sr in series:
        if len(sr['values']) != n:
            F.issue('error', f'у ряда «{sr["name"]}» чисел {len(sr["values"])}, а названий {n}', lid)
            sr['values'] = (sr['values'] + [0.0] * n)[:n]
    if kind in ('pie', 'donut'):
        if len(series) > 1:
            F.issue('error', 'круг показывает один ряд чисел', lid)
        if any(v < 0 for v in series[0]['values']) or sum(series[0]['values']) <= 0:
            F.issue('error', 'кругу нужны неотрицательные числа, не все нули', lid)
            series[0]['values'] = [max(v, 0.0) for v in series[0]['values']] if sum(max(v, 0) for v in series[0]['values']) > 0 else [1.0] * n
    if kind in ('line', 'area') and n < 2:
        F.issue('error', 'линии нужны хотя бы две точки', lid)
        kind = out['kind'] = 'bar'
    out['labels_'], out['series'] = labels, series
    # ---- colours: one for all, one a series, or (one series) one an item
    per = len(series) if len(series) > 1 else (n if kind in ('pie', 'donut') or L.get('colors') else 1)
    cols = L.get('colors')
    if cols is not None and not (isinstance(cols, (list, tuple)) and cols):
        F.issue('error', 'colors: список цветов', lid)
        cols = None
    base = [cols[i % len(cols)] if cols else (L['color'] if 'color' in L and not (kind in ('pie', 'donut')) else PALETTE[i % len(PALETTE)]) for i in range(per)]
    if len(series) > 1:
        base = [sr.get('color') or b for sr, b in zip(series, base)]
    out['paints'] = [F.paint(c, f'{lid}.color') for c in base]
    # ---- the scale
    vals = [v for sr in series for v in sr['values']]
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    ax = _text_spec(F, L, lid, 'axis', kind not in ('pie', 'donut'), 24, '#ffffff66')
    ticks = int(ax.get('ticks', 5)) if ax else 5
    if not 1 <= ticks <= 20:
        F.issue('error', 'axis.ticks: от 1 до 20', lid)
        ticks = 5
    if L.get('range') is not None:
        r = L['range']
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and float(r[1]) > float(r[0])):
            F.issue('error', 'range: [от, до], до больше от', lid)
        else:
            lo, hi = float(r[0]), float(r[1])
            step = (hi - lo) / ticks
    if L.get('range') is None or 'step' not in locals():
        step = _nice((hi - lo) / ticks if hi > lo else 1.0)
        lo, hi = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
        if hi <= lo:
            hi = lo + step
    out['range'], out['step'] = (lo, hi), step
    if ax:
        ax['grid'], ax['width'] = bool(ax.get('grid', True)), float(ax.get('width', 2))
    out['axis'] = ax
    out['labels'] = _text_spec(F, L, lid, 'labels', True, 26, '#ffffffcc')
    vs = _text_spec(F, L, lid, 'values', kind in ('bar', 'hbar', 'pie', 'donut'), 28, '#ffffff')
    whole = all(abs(v - round(v)) < 1e-9 for v in vals)
    for spec in (vs, ax):
        if spec is None:
            continue
        auto = '{:,.0f}' if (whole if spec is vs else True) and abs(step - round(step)) < 1e-9 else '{:,.1f}' if abs(step * 10 - round(step * 10)) < 1e-9 or spec is vs else '{:,.2f}'
        if spec is vs and not whole:
            auto = '{:,.1f}' if all(abs(v * 10 - round(v * 10)) < 1e-9 for v in vals) else '{:,.2f}'
        fm = str(spec.get('format', auto))
        if len(FIELD.findall(fm)) != 1 or '{' in FIELD.sub('', fm) or '}' in FIELD.sub('', fm):
            F.issue('error', f'format «{fm}»: текст с одним полем для числа, как в Python — "{{:.1f}} %", "{{:,.0f}} т"', lid)
            fm = '{:,.1f}'
        try:
            fm.format(1234.5)
        except (ValueError, KeyError, IndexError):
            F.issue('error', f'format «{fm}» не подходит числу', lid)
            fm = '{:,.1f}'
        spec['format'] = fm
        spec['decimal'], spec['thousands'] = str(spec.get('decimal', (vs or {}).get('decimal', '.'))), str(spec.get('thousands', (vs or {}).get('thousands', ' ')))
    out['values'] = vs
    fonts = L.get('font', 'Arial')
    fonts = [str(x) for x in (fonts if isinstance(fonts, list) else [fonts])]
    if 'weight' in L:
        fonts = [f"{x.partition('#')[0]}#wght={float(L['weight']):g}" for x in fonts]
    for x in fonts:
        try:
            from .render import font_file
            font_file(x.partition('#')[0])
        except FileNotFoundError as e:
            F.issue('error', str(e), lid)
            fonts = ['Arial']
            break
    out['font'] = '|'.join(fonts)
    out['progress'] = F.prop(L.get('progress', 1), f'{lid}.progress')
    out['stagger'] = min(1.0, max(0.0, float(L.get('stagger', .6))))
    out['width'] = float(L.get('width', 5))
    out['dots'] = float(L.get('dots', 0))
    out['smooth'] = bool(L.get('smooth', False))
    out['head'] = bool(L.get('head', False))
    out['fill'] = min(1.0, max(0.0, float(L.get('fill', .3))))
    out['gap'] = min(.95, max(0.0, float(L.get('gap', .3))))
    out['round'] = max(0.0, float(L.get('round', 6)))
    out['hole'] = min(.95, max(0.0, float(L.get('hole', .6)))) if kind == 'donut' else 0.0
    out['legend'] = bool(L.get('legend', len(series) > 1))


def _nice(x):
    """a round step not smaller than x: 1, 2, 2.5 or 5 times a power of ten"""
    p = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if m * p >= x * (1 - 1e-9):
            return m * p
    return 10 * p


def number(spec, v):
    """the value as the chart writes it"""
    def one(m):
        s = ('{' + (m.group(1) or '') + '}').format(v + 0.0)
        if s.lstrip('-+').strip('0.,') == '' and s.startswith('-'):
            s = s[1:]                                          # no «-0»
        return s.replace(',', '\0').replace('.', spec['decimal']).replace('\0', spec['thousands'])
    return FIELD.sub(one, spec['format'])


_texts = {}
EM = 200.0


def _text(font, s, tnum=False):
    """a line of text as an outline at 200 px an em, its left edge at 0 and its box's middle at y = 0 → (path, width)"""
    key = (font, s, tnum)
    hit = _texts.get(key)
    if hit is None:
        import skia
        from .render import text_layout
        from .paths import _glyph
        glyphs, (w, h) = text_layout(font, EM, s, 'left', 1.2, 0.0, 'tnum' if tnum else '')
        p = skia.Path()
        for g, x, y, _ in glyphs:
            gp, _, upem = _glyph(font, g)
            m = skia.Matrix()
            m.setScale(EM / upem, EM / upem)
            m.postTranslate(x, y - h / 2)
            p.addPath(gp, m)
        if len(_texts) > 4000:
            _texts.clear()
        hit = _texts[key] = (p, w)
    return hit


def _ease(q):
    return 1 - (1 - q) ** 3


def item_progress(P, i, n, stagger):
    """how far item i of n has come when the whole is at P: each has its own stretch of the way, one after another"""
    P = min(1.0, max(0.0, P))
    if n <= 1 or stagger <= 0:
        return P
    w = 1 - stagger * (n - 1) / n
    return _ease(min(1.0, max(0.0, (P - stagger * i / n) / w)))


def _hermite(xs, ys, smooth, per=20):
    """the line through the points as many short pieces: straight, or a curve that never overshoots a point (monotone)"""
    if not smooth or len(xs) < 3:
        return np.asarray(xs, float), np.asarray(ys, float)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    d = np.diff(ys) / np.diff(xs)
    m = np.zeros(len(xs))
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, len(xs) - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 2 * d[i - 1] * d[i] / (d[i - 1] + d[i])   # harmonic mean: no overshoot
    X, Y = [], []
    for i in range(len(xs) - 1):
        h = xs[i + 1] - xs[i]
        u = np.linspace(0, 1, per, endpoint=False)
        X.append(xs[i] + u * h)
        Y.append((2 * u ** 3 - 3 * u ** 2 + 1) * ys[i] + (u ** 3 - 2 * u ** 2 + u) * h * m[i] + (-2 * u ** 3 + 3 * u ** 2) * ys[i + 1] + (u ** 3 - u ** 2) * h * m[i + 1])
    return np.concatenate(X + [xs[-1:]]), np.concatenate(Y + [ys[-1:]])


def layout(L):
    """where the plot stands inside the plane, once the room the text takes is known → dict"""
    w, h = L['psize']
    font, ax, lb, vs, kind = L['font'], L['axis'], L['labels'], L['values'], L['kind']
    lo, hi = L['range']
    tw = lambda s, size, tnum=False: _text(font, s, tnum)[1] * size / EM
    ticks = [lo + i * L['step'] for i in range(int(round((hi - lo) / L['step'])) + 1)]
    tick_w = max(tw(number(ax, v), ax['size'], True) for v in ticks) if ax else 0.0
    lab_w = max(tw(s, lb['size']) for s in L['labels_']) if lb else 0.0
    val_w = max(tw(number(vs, v), vs['size'], True) for sr in L['series'] for v in sr['values']) if vs else 0.0
    pad = .012 * max(w, h)
    top = pad + (L['axis']['size'] * 1.9 if L['legend'] and L['axis'] else (lb or vs or {'size': 24})['size'] * 1.9 if L['legend'] else 0)
    if kind in ('pie', 'donut'):
        room_x = (lab_w + pad * 2.6) if lb else pad
        room_y = (lb['size'] * 1.4) if lb else pad
        r = max(4.0, min(w / 2 - room_x, (h - top) / 2 - room_y))
        return dict(kind=kind, cx=w / 2, cy=top + (h - top) / 2, r=r, top=top)
    if kind == 'hbar':
        x0 = (lab_w + pad * 1.5) if lb else pad
        x1 = w - ((val_w + pad * 1.5) if vs else pad)
        y0 = top + pad * .5
        y1 = h - ((ax['size'] * 1.7) if ax else pad)
    else:
        x0 = (tick_w + pad * 1.5) if ax else pad
        x1 = w - pad
        y0 = top + ((vs['size'] * 1.6) if vs or L['head'] else pad)
        y1 = h - ((lb['size'] * 1.7) if lb else pad)
    x1, y1 = max(x0 + 4, x1), max(y0 + 4, y1)
    inset = 0.0
    if kind in ('line', 'area'):
        ends = max([tw(L['labels_'][0], lb['size']), tw(L['labels_'][-1], lb['size'])]) / 2 if lb else 0.0
        inset = min((x1 - x0) * .12, max(ends, (val_w / 2) if (vs or L['head']) else 0.0, L['dots'], L['width']) + pad * .5)
    return dict(kind=kind, x0=x0, y0=y0, x1=x1, y1=y1, ticks=ticks, top=top, inset=inset)


def render(F, L, t, sc):
    """the chart at t drawn in its own plane at `sc` px per content px → (x0, y0, w, h in content px, premultiplied
    PIL RGBa picture) or None when there is nothing to draw"""
    import skia
    from PIL import Image
    from .render import paint_of
    w, h = L['psize']
    W, H = int(math.ceil(w * sc)), int(math.ceil(h * sc))
    if W <= 0 or H <= 0 or W * H > 40e6:
        return None
    surf = skia.Surface.MakeRaster(skia.ImageInfo.Make(W, H, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType))
    c = surf.getCanvas()
    c.clear(skia.Color4f(0, 0, 0, 0))
    c.scale(sc, sc)
    G = layout(L)
    P = min(1.0, max(0.0, float(L['progress'].at(t))))
    font, ax, lb, vs, kind = L['font'], L['axis'], L['labels'], L['values'], L['kind']
    series, labels, n = L['series'], L['labels_'], len(L['labels_'])
    box = skia.Rect.MakeWH(w, h)

    def say(s, x, y, spec, ax_=.5, alpha=1.0, tnum=False, keep=True):
        """a line of text: (x, y) is its box's left / middle / right by ax_ and its middle in height"""
        if alpha <= .004 or not s:
            return
        p, tw_ = _text(font, s, tnum)
        k = spec['size'] / EM
        x = x - ax_ * tw_ * k
        if keep:
            x = min(w - tw_ * k - 1, max(1, x))                # never out of the plane
        pt = paint_of(spec['color'], box)
        pt.setAlphaf(pt.getAlphaf() * alpha)
        c.save()
        c.translate(x, y)
        c.scale(k, k)
        c.drawPath(p, pt)
        c.restore()

    def paint(i, alpha=1.0, r=None):
        pt = paint_of(L['paints'][i % len(L['paints'])], r or box)
        if alpha < 1:
            pt.setAlphaf(pt.getAlphaf() * alpha)
        return pt

    def legend(names, top):
        if not L['legend']:
            return
        spec = lb or vs or ax or dict(size=24, color=(255, 255, 255))
        sz = spec['size']
        ws = [_text(font, s)[1] * sz / EM + sz * 2.0 for s in names]
        x = max(0.0, (w - sum(ws)) / 2)
        for i, s in enumerate(names):
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, top - sz * .35, sz * .7, sz * .7), sz * .16, sz * .16), paint(i))
            say(s, x + sz * 1.0, top, spec, 0)
            x += ws[i]

    if kind in ('pie', 'donut'):
        v = series[0]['values']
        total = sum(v)
        cx, cy, r = G['cx'], G['cy'], G['r']
        sweep, a0 = 360 * P, -90.0
        oval = skia.Rect.MakeLTRB(cx - r, cy - r, cx + r, cy + r)
        hole = r * L['hole']
        for i, x in enumerate(v):
            ang = 360 * x / total
            part = min(ang, sweep - (a0 + 90))
            if part > 1e-4:
                p = skia.Path()
                if hole > 0:
                    p.arcTo(oval, a0, part, True)
                    p.arcTo(skia.Rect.MakeLTRB(cx - hole, cy - hole, cx + hole, cy + hole), a0 + part, -part, False)
                else:
                    p.moveTo(cx, cy)
                    p.arcTo(oval, a0, part, False)
                p.close()
                c.drawPath(p, paint(i))
            done = min(1.0, max(0.0, (sweep - (a0 + 90) - ang * .5) / max(ang * .5, 1e-6)))    # the slice's second half fades its text in
            mid = math.radians(a0 + ang / 2)
            if vs and ang > 1e-6:
                rr = (r + hole) / 2 if hole > 0 else r * .64
                say(number(vs, x * (part / ang if part > 0 else 0)), cx + rr * math.cos(mid), cy + rr * math.sin(mid), vs, .5, done, True)
            if lb:
                rr = r + .012 * max(w, h) * 2.0
                say(labels[i], cx + rr * math.cos(mid), cy + (rr + lb['size'] * .35) * math.sin(mid), lb, 0 if math.cos(mid) > .15 else 1 if math.cos(mid) < -.15 else .5, done)
            a0 += ang
        legend(labels, G['top'] - (lb or vs or {'size': 24})['size'] * .95)
        return _snap(surf, W, H, sc)

    x0, y0, x1, y1 = G['x0'], G['y0'], G['x1'], G['y1']
    lo, hi = L['range']
    pw, ph = x1 - x0, y1 - y0
    m = len(series)
    if kind == 'hbar':
        vx = lambda val: x0 + (val - lo) / (hi - lo) * pw
        zero = vx(min(max(0.0, lo), hi))
        if ax:
            line = paint_of(ax['color'], box)
            line.setStyle(skia.Paint.kStroke_Style)
            line.setStrokeWidth(ax['width'])
            for tv in G['ticks']:
                if ax['grid']:
                    line.setAlphaf(paint_of(ax['color'], box).getAlphaf() * (1 if abs(vx(tv) - zero) < .5 else .45))
                    c.drawLine(vx(tv), y0, vx(tv), y1, line)
                say(number(ax, tv), vx(tv), y1 + ax['size'] * .95, dict(ax, color=_solid(ax['color'], 1.6)), .5, 1, True)
        step = ph / n
        bh = step * (1 - L['gap']) / m
        for i in range(n):
            q = item_progress(P, i, n, L['stagger'])
            for j, sr in enumerate(series):
                val = sr['values'][i]
                xe = zero + (vx(val) - zero) * q
                top = y0 + i * step + step * L['gap'] / 2 + j * bh
                if abs(xe - zero) > .05:
                    rr = min(L['round'], bh / 2, abs(xe - zero))
                    rect = skia.RRect()
                    right = xe >= zero
                    rect.setRectRadii(skia.Rect.MakeLTRB(min(zero, xe), top, max(zero, xe), top + bh),
                                      [skia.Point(0, 0) if right else skia.Point(rr, rr), skia.Point(rr, rr) if right else skia.Point(0, 0),
                                       skia.Point(rr, rr) if right else skia.Point(0, 0), skia.Point(0, 0) if right else skia.Point(rr, rr)])
                    c.drawRRect(rect, paint(j if m > 1 else i, r=skia.Rect.MakeLTRB(x0, y0, x1, y1)))
                if vs:
                    say(number(vs, val * q), xe + (1 if val >= 0 else -1) * vs['size'] * .35, top + bh / 2, vs, 0 if val >= 0 else 1, min(1.0, q * 4), True)
            if lb:
                say(labels[i], x0 - .012 * max(w, h), y0 + (i + .5) * step, lb, 1)
        legend([sr['name'] for sr in series], G['top'] - (lb or vs or ax or {'size': 24})['size'] * .95)
        return _snap(surf, W, H, sc)

    vy = lambda val: y1 - (val - lo) / (hi - lo) * ph
    zero = vy(min(max(0.0, lo), hi))
    if ax:
        line = paint_of(ax['color'], box)
        line.setStyle(skia.Paint.kStroke_Style)
        line.setStrokeWidth(ax['width'])
        full = line.getAlphaf()
        for tv in G['ticks']:
            base_ = abs(vy(tv) - zero) < .5
            if ax['grid'] or base_:
                line.setAlphaf(full * (1 if base_ else .45))
                c.drawLine(x0, vy(tv), x1, vy(tv), line)
            say(number(ax, tv), x0 - .012 * max(w, h), vy(tv), dict(ax, color=_solid(ax['color'], 1.6)), 1, 1, True)
    plot = skia.Rect.MakeLTRB(x0, y0, x1, y1)
    if kind == 'bar':
        step = pw / n
        bw = step * (1 - L['gap']) / m
        for i in range(n):
            q = item_progress(P, i, n, L['stagger'])
            for j, sr in enumerate(series):
                val = sr['values'][i]
                ye = zero + (vy(val) - zero) * q
                left = x0 + i * step + step * L['gap'] / 2 + j * bw
                if abs(ye - zero) > .05:
                    rr = min(L['round'], bw / 2, abs(ye - zero))
                    up = ye <= zero
                    rect = skia.RRect()
                    rect.setRectRadii(skia.Rect.MakeLTRB(left, min(zero, ye), left + bw, max(zero, ye)),
                                      [skia.Point(rr, rr) if up else skia.Point(0, 0)] * 2 + [skia.Point(0, 0) if up else skia.Point(rr, rr)] * 2)
                    c.drawRRect(rect, paint(j if m > 1 else i, r=plot))
                if vs:
                    say(number(vs, val * q), left + bw / 2, ye + (-1 if val >= 0 else 1) * vs['size'] * .8, vs, .5, min(1.0, q * 4), True)
            if lb:
                say(labels[i], x0 + (i + .5) * step, y1 + lb['size'] * .95, lb, .5)
    else:                                                      # line, area: drawn from the left as far as progress has come
        xa, xb = x0 + G['inset'], x1 - G['inset']
        xs = [xa + (xb - xa) * i / (n - 1) for i in range(n)]
        cut = xa + (xb - xa) * P
        if lb:
            for i in range(n):
                say(labels[i], xs[i], y1 + lb['size'] * .95, lb, .5)
        for j, sr in enumerate(series):
            X, Y = _hermite(xs, [vy(v) for v in sr['values']], L['smooth'])
            k = int(np.searchsorted(X, cut, 'right'))
            if k < 1:
                continue
            if k < len(X):                                     # the end stands between two pieces
                u = (cut - X[k - 1]) / (X[k] - X[k - 1])
                X, Y = np.append(X[:k], cut), np.append(Y[:k], Y[k - 1] + u * (Y[k] - Y[k - 1]))
            p = skia.Path()
            p.moveTo(float(X[0]), float(Y[0]))
            for a, b in zip(X[1:], Y[1:]):
                p.lineTo(float(a), float(b))
            if kind == 'area' and len(X) > 1:
                f = skia.Path(p)
                f.lineTo(float(X[-1]), zero)
                f.lineTo(float(X[0]), zero)
                f.close()
                col = L['paints'][j % len(L['paints'])]
                if isinstance(col, dict):
                    fp = paint(j, L['fill'], plot)
                else:                                          # one colour: denser at the line, fading towards the zero line
                    ytop = float(min(Y.min(), zero - 1))
                    a_ = (col[3] if len(col) > 3 else 1.0) * L['fill']
                    rgb = [int(round(v)) for v in col[:3]]
                    fp = skia.Paint(AntiAlias=True, Shader=skia.GradientShader.MakeLinear(
                        [skia.Point(0, ytop), skia.Point(0, zero)], [skia.Color(*rgb, int(round(255 * a_))), skia.Color(*rgb, int(round(255 * a_ * .15)))], [0, 1]))
                c.drawPath(f, fp)
            st = paint(j, r=plot)
            st.setStyle(skia.Paint.kStroke_Style)
            st.setStrokeWidth(L['width'])
            st.setStrokeCap(skia.Paint.kRound_Cap)
            st.setStrokeJoin(skia.Paint.kRound_Join)
            if len(X) > 1:
                c.drawPath(p, st)
            pop = max(1e-6, .035 * (xb - xa))                         # a point shows as the line reaches it
            for i in range(n):
                q = _ease(min(1.0, max(0.0, (cut - xs[i]) / pop + (1 if i == 0 and P > 0 else 0))))
                if q <= 0:
                    continue
                if L['dots'] > 0:
                    c.drawCircle(xs[i], vy(sr['values'][i]), L['dots'] * q, paint(j, r=plot))
                if vs and not L['head']:
                    say(number(vs, sr['values'][i]), xs[i], vy(sr['values'][i]) - vs['size'] * .9 - L['dots'], vs, .5, q, True)
            if L['head'] and P > 0:
                hx, hy = float(X[-1]), float(Y[-1])
                c.drawCircle(hx, hy, max(L['dots'], L['width'] * 1.4), paint(j, r=plot))
                spec = vs or dict(size=28, color=(255, 255, 255), format=(ax or {}).get('format', '{:,.1f}'), decimal=(ax or {}).get('decimal', '.'), thousands=(ax or {}).get('thousands', ' '))
                say(number(spec, lo + (y1 - hy) / ph * (hi - lo)), hx, hy - spec['size'] * .95 - max(L['dots'], L['width'] * 1.4), spec, .5, min(1.0, P * 12), True)
    legend([sr['name'] for sr in series], G['top'] - (ax or lb or vs or {'size': 24})['size'] * .95)
    return _snap(surf, W, H, sc)


def _solid(col, k):
    """the axis colour for its numbers: the same, less transparent"""
    if isinstance(col, dict):
        return col
    return tuple(col[:3]) + (min(1.0, (col[3] if len(col) > 3 else 1.0) * k),)


def _snap(surf, W, H, sc):
    import skia
    from PIL import Image
    a = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
    if not a[..., 3].any():
        return None
    return 0.0, 0.0, W / sc, H / sc, Image.fromarray(np.ascontiguousarray(a), 'RGBa')
