"""kadr check — the agent's eyes, as text. Every finding names the time, the voice-over word there and the layer."""
import math
import numpy as np
from .doc import Issue
from .media import probe
from .render import bbox, place, content_size, shape_path

SAFE = 0.05            # title-safe margin, share of the frame
READ_SPEED = 20.0      # px per frame above which moving text is hard to read
CPS = 20.0             # caption characters per second above which they can't be read
JERK = 1.5             # px per frame: sudden change of velocity


def ranges(frames):
    """consecutive frame numbers → [(first, last)]"""
    out = []
    for i in frames:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def top_edge(film, L, t, p):
    """the layer's top side on screen as a vector (from its left corner to its right one), px — how large the layer
    is and which way it is turned, in two numbers that change smoothly when it turns, grows or flips over in 3D.
    (The width of its upright box, measured before, has a kink each time a turning layer's angle passes zero —
    w·|cos| + h·|sin| — and every steadily turning gear, wheel and clock hand was reported as jerking.)"""
    if L['type'] == 'shape':
        b = shape_path(L, t).computeTightBounds()
        x0, x1, y0 = b.left(), b.right(), b.top()
    else:
        x0, x1, y0 = 0.0, content_size(film, L, t)[0], 0.0
    if p.mat is not None:
        a, b_ = p.mat @ np.array([x0, y0, 1.0]), p.mat @ np.array([x1, y0, 1.0])
        return np.array([b_[0] / b_[2] - a[0] / a[2], b_[1] / b_[2] - a[1] / a[2]])
    th = math.radians(p.rot if p.rot is not None else (L['rotation'].at(t) if L.get('rotation') else 0.0))
    return p[2] * (x1 - x0) * np.array([math.cos(th), math.sin(th)])


def run(film, strict=False):
    """strict=False: layers a channel style drew are its design — their motion curves and edge positions were matched to
    the channel's films once (kadr check --strict on the style's examples), so only what the document decides is judged"""
    fps, N = film.fps, film.frames
    styled = set() if strict else getattr(film, 'by_style', set())
    issues = list(film.issues)
    seen = set()
    for L in film.layers:                                      # a shader that does not compile: the compiler's own words
        for fx in L.get('effects') or []:
            sh = getattr(fx['def'][0], 'shader', None)
            if sh is not None and sh.key not in seen:
                seen.add(sh.key)
                err = sh.compile_error()
                if err:
                    issues.append(Issue('error', f'шейдер {sh.name} не собирается:\n{err}', L['id']))

    for L in film.layers:                                      # a counter's text is a formula: its characters are known only by asking
        if L['type'] == 'text' and L.get('text_expr') is not None:
            from .render import missing_glyphs
            a_, b_ = L['in'], min(L['out'], film.duration)
            seen_txt = ''
            for j in range(9):
                try:
                    seen_txt += str(L['text_expr'].at(a_ + (b_ - a_) * j / 8))
                except Exception:                              # a formula that fails is reported where formulas are
                    break
            try:
                miss = missing_glyphs(L['font'], seen_txt)
            except (OSError, ValueError):
                miss = []
            if miss:
                issues.append(Issue('warn', 'в шрифте ' + L['font'].split('#')[0] + ' нет знаков ' + ', '.join(f'«{c}» (U+{ord(c):04X})' for c in miss[:8])
                                    + ' — на их месте будет пустой прямоугольник. Дайте запасной шрифт списком (font: [свой, Inter-Regular]) или замените знак', L['id']))

    PART = {'label': 'подпись', 'src': 'источник', 'spname': 'имя в плашке', 'spsub': 'подпись в плашке', 'spbar': 'полоса плашки',
            'st1': 'штамп', 'st2': 'штамп, вторая строка', 'stline': 'штамп, линия', 'paper': 'бумага', 'ghost': 'номер главы',
            'kicker': 'надзаголовок', 'title': 'заголовок'}

    OVERLAYS = {'label', 'src', 'spname', 'spsub', 'st1', 'st2'}     # parts any shot may carry; the rest is the element's own layout

    def owner(lid):
        sid = max((s for s in getattr(film, 'shot_kinds', {}) if lid == s or lid.startswith(s + '_')), key=len, default=None)
        return sid, (lid[len(sid) + 1:] if sid else lid)

    def named(lid):
        """a layer made from a shot is reported as that shot and its part: «кадр genda (print) → подпись»"""
        names = []
        for part in lid.split(' × '):
            sid = max((s for s in getattr(film, 'shot_kinds', {}) if part == s or part.startswith(s + '_')), key=len, default=None)
            if sid is None:
                names.append(part)
                continue
            rest = part[len(sid) + 1:]
            names.append(f'кадр {sid} ({film.shot_kinds[sid]})' + (f' → {PART.get(rest, rest)}' if rest else ''))
        return ' × '.join(names)

    def at(i0, i1, lid, msg, level='warn'):
        lid = named(lid)
        t0, t1 = i0 / fps, i1 / fps
        w = film.word_near(t0)
        span = f'{t0:.2f} с' if i0 == i1 else f'{t0:.2f}–{t1:.2f} с'
        issues.append(Issue(level, msg, f'{span}{f" («{w}»)" if w else ""} {lid}'))

    if any(L['z'] is not None for L in film.layers if L['type'] != 'camera') and film.camera is None:
        issues.append(Issue('warn', 'у слоёв есть z, но нет камеры — z игнорируется'))
    if film.camera is not None:                                # depth and parents are two different ways to place a layer: together they do not work
        for L in film.layers:
            if L['type'] == 'camera' or L.get('parent_layer') is None or L.get('mask_of'):
                continue
            if L['z'] is not None:
                issues.append(Issue('warn', f'у слоя и parent («{L["parent_layer"]["id"]}»), и z под камерой: родитель не действует — слой с глубиной ставится по собственному position. '
                                            f'Уберите parent (а движение родителя прибавьте формулой: =layer("{L["parent_layer"]["id"]}").position) или уберите z', L['id']))
            else:
                q = L['parent_layer']
                while q is not None and q['z'] is None:
                    q = q.get('parent_layer')
                if q is not None:
                    issues.append(Issue('warn', f'родитель «{q["id"]}» стоит в глубине (z под камерой), а у слоя z нет: камера двигает родителя, а слой за ним в глубину не идёт. '
                                                f'Дайте слою тот же z и уберите parent', L['id']))

    for L in film.layers:
        span = L['out'] - L['in'] if L['type'] != 'camera' else 0
        if L['type'] != 'camera' and L['fade_in'] + L['fade_out'] > span + 1e-6:
            at(int(L['in'] * fps), int(L['in'] * fps), L['id'], f'fade_in + fade_out = {L["fade_in"] + L["fade_out"]:.2f} с длиннее самого слоя ({span:.2f} с)')
        if L['type'] == 'video' and L.get('remap') is not None:
            p = probe(L['src'])
            late = [f_ for f_ in range(int(L['in'] * fps), int(L['out'] * fps)) if not (-1e-3 <= L['remap'].at(f_ / fps) <= p['duration'] + 1e-3)]
            for a_, b_ in ranges(late):
                at(a_, b_, L['id'], f'time уводит за пределы исходника (длина {p["duration"]:.2f} с) — там стоит крайний кадр')
            if L.get('volume') is not None or L.get('loudness') is not None:
                issues.append(Issue('warn', 'у клипа с time (Time Remap) звука нет: звук не может идти по кривой времени', L['id']))
        elif L['type'] == 'video':
            p = probe(L['src'])
            need = L['from'] + span * L['speed']
            if not L['loop'] and need > p['duration'] + 1e-3:
                t_end = L['in'] + (p['duration'] - L['from']) / L['speed']
                at(int(t_end * fps), int(L['out'] * fps) - 1, L['id'],
                   f'съёмка кончается на {p["duration"]:.2f} с исходника, а нужна до {need:.2f} с — дальше стоит последний кадр')
            if L['from'] < 0 or L['from'] >= p['duration']:
                at(int(L['in'] * fps), int(L['in'] * fps), L['id'], f'from = {L["from"]:.2f} с вне исходника (длина {p["duration"]:.2f} с)', 'error')
            if L.get('volume') is not None and not p['audio']:
                issues.append(Issue('warn', 'volume задан, но в исходнике нет звука', L['id']))
        if L['type'] == 'captions':
            for ph in L['phrases']:
                chars = len(ph['text'].replace('\n', ' '))
                cps = chars / max(0.01, ph['end'] - ph['start'])
                if cps > CPS:
                    at(int(ph['start'] * fps), int(ph['end'] * fps), L['id'], f'подпись «{ph["text"][:40]}…» читать {cps:.0f} знаков/с (порог {CPS:.0f})')

    mx, my = film.W * SAFE, film.H * SAFE
    boxes = {}
    for L in film.layers:
        if L['type'] in ('camera', 'group'):            # a group draws nothing itself
            continue
        i0, i1 = max(0, int(math.floor(L['in'] * fps))), min(N, int(math.ceil(L['out'] * fps)) + 1)
        bs = [None] * N                                        # only the frames the layer is on: long films stay fast
        for i in range(i0, i1):
            bs[i] = bbox(film, L, i / fps)
        boxes[L['id']] = bs
        off, unsafe = [], {}
        for i, b in enumerate(bs):
            if b is None or L['opacity'].at(i / fps) <= 0:
                continue
            x0, y0, x1, y1 = b
            if x1 <= 0 or y1 <= 0 or x0 >= film.W or y0 >= film.H:
                off.append(i)
            elif L['type'] in ('text', 'captions') and L['opacity'].at(i / fps) >= 0.2:   # a faint backdrop number is decor, not text to read
                for side, d in (('слева', mx - x0), ('сверху', my - y0), ('справа', x1 - (film.W - mx)), ('снизу', y1 - (film.H - my))):
                    if d > 0.5:
                        unsafe.setdefault(side, []).append((i, d))
        for a, b in ranges(off if L['id'] not in styled else ()):
            # a layer that flies in from outside the frame, or out of it, is outside for its first or last moments —
            # that is the move itself (five reels of 16.11.2026: 237 of 248 such findings were these, and the real
            # ones were lost among them). Reported: outside in the middle of its life, or for longer than a second
            # (its first and last moments on screen: a child of a group that ends has no end of its own in the document)
            edge = a - 2 < i0 or b + 2 >= i1 or bs[a - 1] is None or bs[a - 2] is None or bs[b + 1] is None or bs[b + 2] is None
            if edge and (b - a + 1) / fps <= 1.0 and (b - a + 1) < 0.5 * sum(x is not None for x in bs):
                continue
            at(a, b, L['id'], 'слой виден по времени, но целиком за кадром')
        if L['type'] == 'image' and L['fit'] == 'cover':
            gaps = {}
            for i, b in enumerate(bs):
                if b is not None:
                    for side, d in (('слева', b[0]), ('сверху', b[1]), ('справа', film.W - b[2]), ('снизу', film.H - b[3])):
                        if d > 0.01:
                            gaps.setdefault(side, []).append((i, d))
            for side, hits in gaps.items():
                depth = dict(hits)
                for a, b in ranges([i for i, _ in hits]):
                    at(a, b, L['id'], f'фон не закрывает кадр {side}: дыра до {max(depth[i] for i in range(a, b + 1)):.0f} px')
        for side, hits in (unsafe.items() if L['id'] not in styled else ()):
            depth = dict(hits)
            for a, b in ranges([i for i, _ in hits]):
                worst = max(depth[i] for i in range(a, b + 1))
                at(a, b, L['id'], f'текст выходит из безопасной зоны {side} на {worst:.0f} px')

        # motion: velocity of the layer centre and of its size on screen
        if L['type'] == 'captions':                          # captions don't move; a new phrase is not a jerk
            continue
        tr, wd = [None] * N, [None] * N
        for i in range(max(0, i0 - 1), min(N, i1 + 1)):
            tr[i] = place(film, L, i / fps)
            # a counter (a text that is a formula) changes width whenever a digit does: that is its text, not a move
            wd[i] = top_edge(film, L, i / fps, tr[i]) if tr[i] is not None and L['type'] != 'group' and L.get('text_expr') is None else np.zeros(2)
        dvs_all, fast = {}, []
        for i in range(max(1, i0), min(N - 1, i1)):
            if tr[i - 1] is None or tr[i] is None or tr[i + 1] is None:
                continue
            v0 = np.array([tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1], *(wd[i] - wd[i - 1])])
            v1 = np.array([tr[i + 1][0] - tr[i][0], tr[i + 1][1] - tr[i][1], *(wd[i + 1] - wd[i])])
            dvs_all[i] = np.linalg.norm(v1 - v0)
            if L['type'] in ('text', 'captions') and np.linalg.norm(v1[:2]) > READ_SPEED:
                fast.append(i)
        # a jerk is a velocity change concentrated in one frame (a kink between keys, a start/stop without ease);
        # a smooth ease changes velocity over several frames, so its neighbours two frames away change almost as much
        # (a reframed clip is slid frame of footage by frame of footage: 24 fps footage in a 25 fps film repeats a frame a second, and the layer with it)
        jerks = [] if L['id'] in styled or L['id'] in getattr(film, 'sound_driven', ()) or L.get('reframe') else [(i, d) for i, d in dvs_all.items()                # both neighbours must exist: a layer just appearing has no «before»
                 if i - 2 in dvs_all and i + 2 in dvs_all and d > JERK and d > 2.5 * max(dvs_all[i - 2], dvs_all[i + 2])
                 and L['opacity'].at(i / fps) >= 0.2]          # a move while the layer is still invisible is not seen
        dvs = dict(jerks)
        for a, b in ranges([i for i, _ in jerks]):
            at(a, b, L['id'], f'рывок: скорость меняется скачком на {max(dvs[i] for i in range(a, b + 1)):.1f} px/кадр — нужен ease или ключ без излома')
        for a, b in ranges(fast if L['id'] not in styled else ()):
            at(a, b, L['id'], f'текст движется быстрее {READ_SPEED:.0f} px/кадр — его не успеют прочитать')

    texts = sorted((L for L in film.layers if L['type'] in ('text', 'captions')), key=lambda L: L['in'])
    for ai, La in enumerate(texts):
        for Lb in texts[ai + 1:]:
            if Lb['in'] >= La['out']:                          # sorted by start: nothing later overlaps La in time
                break
            a, b = La['id'], Lb['id']
            (sa, ra), (sb, rb) = owner(a), owner(b)
            if (sa and sa == sb and ra not in OVERLAYS and rb not in OVERLAYS) or (a in styled and b in styled):
                continue                                       # letters and lines of one element: the style placed them so
            # the same words in the same place twice — an outline under a fill, a dim copy under a bright one — are one text
            twin = La['type'] == 'text' and Lb['type'] == 'text' and La.get('text') == Lb.get('text') and La.get('font') == Lb.get('font') and La.get('size') == Lb.get('size')
            hit = []
            lo, hi = max(La['in'], Lb['in']), min(La['out'], Lb['out'])
            if hi <= lo:
                continue
            for i in range(max(0, int(lo * fps)), min(N, int(math.ceil(hi * fps)) + 1)):
                p, q = boxes[a][i], boxes[b][i]
                if min(La['opacity'].at(i / fps), Lb['opacity'].at(i / fps)) < 0.2:
                    continue
                tails = getattr(film, 'tr_tails', {})                # one of them is dissolving out under a transition
                if any(x in tails and tails[x][0] <= i / fps < tails[x][1] for x in (a, b)):
                    continue
                if La.get('mask'):
                    m = La['mask'].at(i / fps)
                    p = p and (max(p[0], m[0]), max(p[1], m[1]), min(p[2], m[2]), min(p[3], m[3]))
                if Lb.get('mask'):
                    m = Lb['mask'].at(i / fps)
                    q = q and (max(q[0], m[0]), max(q[1], m[1]), min(q[2], m[2]), min(q[3], m[3]))
                if twin and p and q and max(abs(u - v) for u, v in zip(p, q)) < 1.5:
                    continue
                if p and q and p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]:
                    hit.append(i)
            for x, y in ranges(hit):
                at(x, y, f'{a} × {b}', 'текстовые слои перекрываются')
    for L in film.layers:                                      # a picture bent by pins: folds, thinning, jumps (kadr/puppet.py)
        if L['type'] == 'puppet':
            from . import puppet
            for x, y, msg in puppet.audit(film, L):
                at(x, y, L['id'], msg)
    # a 3D scene written in the document or in Kadr's words: bodies of different things through each other, a body never
    # seen, a body in the air, sheets that will flicker (kadr/scenecheck.py) — as it stands when the film begins
    for L in film.layers:
        if L.get('type') == 'scene3d' and L.get('made') is not None:
            from . import scenecheck, scene3d
            sc = scene3d.layer_scene(L)
            world, seen_ = scenecheck.pose(sc, L, 0.0, fps)
            issues += [Issue('warn', 'сцена: ' + words, L['id']) for _, _, words in scenecheck.audit(sc, world, seen_)]
    return issues
