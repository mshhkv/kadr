"""Shots: the film written as a sequence of what is on screen, in the channel's style.

    shots:
      - omni: clips/s0322.mp4          # one key names the kind of shot and its picture
        to: word:лётчик                 # it ends here (or len: 4.2); it starts where the previous one ended (or at: …)
      - print: archive/genda.jpg
        to: word:Футида
        speaker: {name: Минору Гэнда, sub: штабной лётчик}
        source: Imperial Japanese Navy
      - chapter: {n: 4, title: Слух из Токио}
        len: 4.5
        tr: fadeblack

Kinds: photo, print, doc, video, omni, line, chapter, title. Any shot may carry: id, label, source, stamp {line1, line2},
speaker {name, sub}, tr (fade | fadeblack | fadewhite, or [kind, seconds]). Each shot becomes plain layers (see styles/);
plain `layers:` in the document still go on top of all shots."""
from . import plugins
from .styles import STYLES, ShotError
from .styles.base import r
from . import templates as tpl

KINDS = ('photo', 'print', 'doc', 'video', 'omni', 'line', 'chapter', 'title')
COMMON = {'id', 'at', 'to', 'len', 'tr', 'label', 'source', 'stamp', 'speaker', 'over'}
OWN = {
    'photo': {'zoom', 'origin', 'fit'},
    'print': {'zoom', 'tilt'},
    'doc': {'highlight', 'highlight_at', 'zoom_from', 'zoom_to', 'tilt'},
    'video': {'from', 'speed', 'crop', 'flip', 'grade', 'volume', 'loudness', 'fit', 'audio_filter'},
    'omni': {'from', 'speed', 'fit', 'until', 'delogo', 'flip', 'ambience', 'crop', 'audio_filter'},
    'line': {'from', 'sound_len', 'flip', 'crop', 'audio_filter'},
    'chapter': set(),
    'title': {'sub'},
}
TRANS = {'fade', 'dissolve', 'fadeblack', 'fadewhite', 'hblur', 'zoomin', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown'}
OMNI_MARK = [1130, 570, 60, 60]          # Omni's corner sparkle, 1280×720 source px


def expand(film, raw_shots, style_name):
    """→ (raw layer dicts, end of the last shot)"""
    if style_name not in STYLES:
        film.issue('error', f'нет стиля «{style_name}»', 'film.style', style_name, STYLES)
        return [], 0.0
    st = STYLES[style_name](film.fps)
    st.film = film                                       # media paths (film.resolve), words, size — for any element that needs them
    extra_kinds = plugins.kinds(style_name) | set(film.templates)
    groups, t, last_start = [], 0.0, 0.0
    for n, sh in enumerate(raw_shots):
        kind = next((k for k in sh if k in KINDS or k in extra_kinds), None)
        sid = str(sh.get('id', f'{kind or "shot"}{n + 1}'))
        where = f'shots[{n}] {sid}'
        if kind is None:
            film.issue('error', f'у кадра нет вида: нужен один из ключей {", ".join(sorted(set(KINDS) | extra_kinds))}', where)
            continue
        reg = plugins.lookup(style_name, kind) if kind not in KINDS else None
        own = OWN.get(kind) or (reg[1] if reg else set())
        for k in sh:
            if k not in COMMON | own | {kind}:
                film.issue('error', f'у кадра {kind} нет свойства «{k}»', where, k, COMMON | own)
        el_fields = (reg[1] if reg and reg[1] else None) if kind not in KINDS else None
        tp_fields = set((film.templates[kind].get('params') or {})) if kind in film.templates else None
        checks = [(k, set(v)) for k, v in st.fields.items()]  # a style says which fields its stamp, speaker… take
        if el_fields or tp_fields:                            # a card's own fields: the element's declared params / the template's
            checks.append((kind, el_fields or tp_fields))
        for k, allowed in checks:
            v = sh.get(k)
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk not in allowed:
                        hint = (' — похоже, запятая оборвала текст: возьмите значение в кавычки' if vv is None else '')
                        film.issue('error', f'{k}: нет поля «{kk}»{hint}', where, kk, allowed)
        over = bool(sh.get('over'))                         # drawn over what is on screen; the sequence does not move on
        t0 = film.time(sh['at'], where) if 'at' in sh else (last_start if over else t)
        if 'to' in sh:
            t1 = film.time(sh['to'], where)
        elif 'len' in sh:
            t1 = t0 + float(sh['len'])
        else:
            film.issue('error', 'нужно to: (время или слово, где кадр кончается) или len: (секунды)', where)
            continue
        t0, t1 = round(t0 * film.fps) / film.fps, round(t1 * film.fps) / film.fps   # cuts on the nearest frame, as assemble.py
        if t1 <= t0:
            film.issue('error', f'кадр кончается ({t1:.2f} с) не позже, чем начинается ({t0:.2f} с)', where)
            continue
        sh = dict(sh, id=sid)
        film.shot_kinds = getattr(film, 'shot_kinds', {})
        film.shot_kinds[sid] = kind
        for k in ('photo', 'print', 'doc', 'video', 'omni', 'line'):
            if k in sh:
                sh[k] = film.resolve(str(sh[k]))
        # neighbours, for styles whose look depends on them (a paper landing on the previous paper…):
        # st.prev = (kind, shot, t0, t1) of the shot before, st.next = the raw entry of the next one
        st.prev = (groups[-1][0].get('_kind'), groups[-1][0], groups[-1][1], groups[-1][2]) if groups else None
        st.next = raw_shots[n + 1] if n + 1 < len(raw_shots) else None
        sh['_kind'] = kind
        own = []
        try:
            if kind in film.templates:
                own = tpl.expand(film, kind, film.templates[kind], sh, t0, t1)
                layers = own + overlays(st, sh, t0, t1)
            elif reg:
                got = reg[0](st, sh, t0, t1)
                main, top = got if isinstance(got, tuple) else (got, [])
                layers = main + (st.source(sid, sh['source'], t0, t1) if sh.get('source') else []) + top + overlays(st, sh, t0, t1)
            else:
                layers = shot_layers(film, st, kind, sh, t0, t1)
        except ShotError as e:
            film.issue('error', str(e), where)
            layers = []
        except FileNotFoundError as e:
            film.issue('error', f'нет файла: {e.filename or e}', where)
            layers = []
        mine = {id(L) for L in own} if kind in film.raw_templates else set()   # a document's own template is the document's
        film.by_style.update(str(L.get('id')) for L in layers if id(L) not in mine)   # the rest is the channel's look (kadr check --strict)
        groups.append((sh, t0, t1, layers))
        if not over:
            t, last_start = t1, t0
    extra = transitions(film, groups)
    film.by_style.update(str(x.get('id')) for _, x in extra)
    out = []
    for i, (_, _, _, L) in enumerate(groups):
        out += L + [x for j, x in extra if j == i]
    return out, (groups[-1][2] if groups else 0.0)


def shot_layers(film, st, kind, sh, t0, t1):
    sid = sh['id']
    top, L = [], []
    if kind in ('photo', 'print', 'doc', 'chapter', 'title'):
        args = (sh, t0, t1, bool(sh.get('speaker'))) if kind == 'print' else (sh, t0, t1)
        L, top = getattr(st, kind)(*args)
    elif kind == 'video':
        v = {'id': sid, 'type': 'video', 'src': sh['video'], 'from': r(sh.get('from', 0), 4), 'in': r(t0), 'out': r(t1),
             'size': sh.get('fit', 'cover')}                      # height: whole height, narrower footage pillarboxed (assemble.py)
        for k in ('speed', 'crop', 'flip', 'volume', 'loudness', 'audio_filter'):
            if k in sh:
                v[k] = sh[k]
        grade = sh.get('grade', st.footage_grade)
        if grade:
            v['grade'] = grade
        L = [v]
    elif kind in ('omni', 'line'):
        L = [take(film, st, kind, sh, t0, t1)]
    if sh.get('source'):
        L += st.source(sid, sh['source'], t0, t1)
    return L + top + overlays(st, sh, t0, t1)


def overlays(st, sh, t0, t1):
    L = []
    if sh.get('stamp'):
        L += st.stamp(sh['id'], sh['stamp'], t0, t1)
    if sh.get('speaker'):
        L += st.speaker(sh['id'], sh['speaker'], t0, t1)
    return L


def take(film, st, kind, sh, t0, t1):
    """a generated (Omni) or presenter take: grade, in-point, cut → freeze, retime to land on the next shot, corner mark, sound"""
    from .media import probe
    sid, src = sh['id'], sh[kind]
    t_in = float(sh.get('from', 0))
    v = {'id': sid, 'type': 'video', 'src': src, 'from': r(t_in, 4), 'in': r(t0), 'out': r(t1), 'size': 'cover', 'grade': st.omni_grade}
    for k in ('crop', 'flip', 'audio_filter'):
        if k in sh:
            v[k] = sh[k]
    if kind == 'line':
        v['loudness'] = st.line_loudness
        if 'sound_len' in sh:
            v['sound_to'] = r(t_in + float(sh['sound_len']), 4)
        return v
    L0 = (round(t1 * film.fps) - round(t0 * film.fps)) / film.fps
    avail = probe(src)['duration'] - t_in
    if 'until' in sh:
        avail = min(avail, float(sh['until']) - t_in)
        v['until'] = r(float(sh['until']), 4)
    fit = sh.get('fit', 'auto')                       # auto: a take a little short is slowed down rather than frozen
    if 'speed' in sh:
        v['speed'] = float(sh['speed'])
    elif fit is True or (fit == 'auto' and (0.5 if 'until' in sh else 0.75) * L0 <= avail < L0):
        v['speed'] = r(avail / L0, 5)
    if sh.get('delogo'):
        v['delogo'] = OMNI_MARK if sh['delogo'] is True else list(sh['delogo'])
    if sh.get('ambience', True):
        v['loudness'], v['max_gain'] = st.omni_loudness
    return v


def transitions(film, groups):
    """B starts on its own frame; A keeps playing under it for d seconds and is blended out (the shot stays in sync
    with the narration). fade / dissolve: B fades in over A; fadeblack / fadewhite: through a colour; hblur: both
    blur sideways through the cut; zoomin: A pushes in and dissolves; smoothleft / smoothright / smoothup /
    smoothdown: B wipes in. A transition that moves or blurs a whole shot wraps it in a comp first."""
    extra = []
    for i, (sh, a0, a1, layers) in enumerate(groups):
        tr = sh.get('tr')
        if not tr or i == 0 or not layers:
            continue
        kind, d = (tr, 0.8) if isinstance(tr, str) else (tr[0], float(tr[1]))
        if kind not in TRANS:
            film.issue('warn', f'переход «{kind}» неизвестен — показан растворением', f'shots {sh["id"]}', kind, TRANS)
            kind = 'fade'
        d = max(2 / film.fps, min(d, a1 - a0 - 2 / film.fps))
        psh, pa0, pa1, prev = groups[i - 1]
        film.tr_tails = getattr(film, 'tr_tails', {})
        for L in prev:
            if abs(L['out'] - pa1) < 1e-3:
                L['out'] = r(pa1 + d)
                film.tr_tails[L['id']] = (pa1, pa1 + d)       # it only plays on under the next shot's transition
        if kind in ('fadeblack', 'fadewhite'):
            extra.append((i, {'id': f'{sh["id"]}_tr', 'type': 'solid', 'size': [film.W, film.H], 'color': '#000000' if kind == 'fadeblack' else '#ffffff',
                              'in': r(a0), 'out': r(a0 + d), 'opacity': {r(a0): 0, r(a0 + d / 2): 1, r(a0 + d): 0, 'ease': 'linear'}}))
        elif kind in ('hblur', 'zoomin'):
            A = wrap(film, psh['id'], prev, pa0, pa1 + d)
            B = wrap(film, sh['id'], layers, a0, a1)
            m = r(a0 + d / 2)
            if kind == 'hblur':                                # both shots smear through the cut while B takes over
                A['blur'] = {r(a0): [0, 0], m: [40, 0], 'ease': 'in'}
                B['blur'] = {m: [40, 0], r(a0 + d): [0, 0], 'ease': 'out'}
            else:                                              # A pushes in hard as it dissolves
                A['scale'] = {r(a0): 1.0, r(a0 + d): 1.6, 'ease': 'in'}
            B['opacity'] = {r(a0): 0, r(a0 + d): 1, 'ease': 'linear'}
            prev[:] = [A]
            layers[:] = [B]
        elif kind.startswith('smooth'):
            side = {'smoothleft': 'right', 'smoothright': 'left', 'smoothup': 'bottom', 'smoothdown': 'top'}[kind]
            B = wrap(film, sh['id'], layers, a0, a1)
            B['wipe_in'] = {'dur': d, 'from': side}
            layers[:] = [B]
        else:
            for L in layers:
                if abs(L['in'] - a0) < 1e-3:
                    L['fade_in'] = r(d)
    return extra


def wrap(film, sid, layers, t0, t1):
    """a shot's layers as one comp layer over the whole frame (its own clock = the film's), so a transition can move,
    scale or blur the shot as a whole"""
    return {'id': f'{sid}_group', 'type': 'comp', 'size': [film.W, film.H], 'in': r(t0), 'out': r(t1), 'time': r(t0),
            'layers': [dict(L) for L in layers]}

def clamp(layers, dur):
    """shots keep their full times (their moves keep their real pace); the film's own span cuts them: a layer that
    started before 0 enters its footage that much later"""
    out = []
    for L in layers:
        if L['out'] <= 0 or L['in'] >= dur:
            continue
        if L['in'] < 0:
            if L['type'] == 'video':
                L['from'] = r(L.get('from', 0) + (-L['in']) * L.get('speed', 1), 4)
            L['in'] = 0.0
        L['out'] = min(L['out'], dur)                          # not rounded: a style's switch times are its own
        out.append(L)
    return out
