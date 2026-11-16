"""The film document: loading, validation, time expressions (seconds or words of the voice-over) and animated properties."""
import threading
import difflib, json, math, os, re
import yaml

TRANS = {'fade_in', 'fade_out', 'wipe_in', 'wipe_out'}
COMMON = {'id', 'type', 'position', 'z', 'scale', 'opacity', 'in', 'out', 'blend', 'rotation', 'blur', 'shadow', 'mask', 'filter',
          'matte', 'effects', 'parent', 'anchor', 'skew', 'rotation_x', 'rotation_y', 'perspective', 'perspective_origin', 'backface',
          'corner_pin', 'blur_space', 'visible', 'echo', 'posterize'} | TRANS
MATTES = ('alpha', 'alpha_inverted', 'luma', 'luma_inverted')
BLENDS = {'normal', 'multiply', 'screen', 'overlay', 'soft-light', 'add', 'darken', 'lighten', 'color-dodge', 'color-burn', 'hard-light',
          'difference', 'exclusion', 'subtract', 'hue', 'saturation', 'color', 'luminosity',
          # AE stencils: everything under the layer (in its group / comp) shows only where the layer is — or is not
          'stencil-alpha', 'stencil-luma', 'silhouette-alpha', 'silhouette-luma'}
TYPES = {
    'camera': {'id', 'type', 'position', 'zoom', 'in', 'out'},
    'solid': COMMON | {'color', 'size'},
    'image': COMMON | {'src', 'size', 'crop', 'grade', 'flip', 'seq_offset', 'tile', 'tile_scale', 'tile_offset', 'cutout'},
    'video': COMMON | {'src', 'size', 'crop', 'grade', 'flip', 'from', 'speed', 'volume', 'loop', 'until', 'delogo', 'loudness', 'max_gain', 'sound_to', 'audio_filter', 'time', 'frame_blend', 'stabilize', 'stab_of', 'cutout', 'reframe', 'reframe_of'},
    'text': COMMON | {'text', 'font', 'weight', 'features', 'size', 'color', 'align', 'line_height', 'tracking', 'stroke', 'animate'},
    'gradient': COMMON | {'kind', 'stops', 'angle', 'center', 'radius', 'size', 'interpolate'},
    'noise': COMMON | {'frequency', 'octaves', 'seed', 'color', 'levels'},
    'shape': COMMON | {'shape', 'size', 'radius', 'points', 'd', 'fill', 'stroke', 'trim', 'arrow', 'anchor', 'closed', 'fill_rule',
                       'text', 'font', 'weight', 'features', 'font_size', 'tracking', 'align', 'line_height', 'along', 'along_offset',
                       'paths', 'offset', 'repeat'},        # curves as data: kadr/paths.py
    'comp': COMMON | {'layers', 'size', 'time', 'speed', 'background', 'src'},
    'group': COMMON | {'isolate'},                      # a null object: moves, scales, turns its children (parent: id)
    'dust': COMMON | {'count', 'seed', 'color', 'drift', 'sizes', 'alphas'},
    'particles': COMMON | {'src', 'size', 'cell', 'mode', 'start', 'sweep', 'direction', 'wind', 'spread', 'turbulence', 'gravity',
                           'life', 'jitter', 'seed', 'shrink', 'crop', 'grade'},
    # particle systems (kadr/psys.py): what is born, flies and dies; points and the lines between them
    'emitter': COMMON | {'size', 'seed', 'prewarm', 'emit', 'particle', 'forces', 'collide', 'links'},
    'plexus': COMMON | {'size', 'seed', 'points', 'links', 'triangles', 'turn', 'lens', 'fade'},
    'svg': COMMON | {'src', 'size', 'color'},                       # a vector picture from a file (kadr/svg.py)
    'puppet': COMMON | {'src', 'cutout', 'pins', 'mesh'},            # a picture bent by pins (kadr/puppet.py)
    'chart': COMMON | {'size', 'kind', 'data', 'src', 'columns', 'progress', 'stagger', 'color', 'colors', 'range', 'axis', 'labels', 'values', 'font', 'weight',
                       'width', 'dots', 'smooth', 'head', 'fill', 'gap', 'round', 'hole', 'legend'},   # bars, a line, a pie from numbers (kadr/chart.py)
    # sound as a picture (kadr/sound.py): its frequencies as bars, its own curve
    'spectrum': COMMON | {'size', 'audio', 'bands', 'range', 'axis', 'shape', 'mirror', 'symmetric', 'ring', 'gap', 'round', 'color', 'width', 'gain', 'db', 'fall', 'floor'},
    'wave': COMMON | {'size', 'audio', 'window', 'span', 'shape', 'bars', 'gap', 'round', 'color', 'played', 'width', 'gain', 'norm', 'mirror', 'ring'},
    # a 3D scene drawn by the core (kadr/scene3d.py): from a file or written here from simple bodies
    'scene3d': COMMON | {'size', 'src', 'clip', 'scene', 'samples', 'time', 'look', 'nodes', 'camera', 'cuts', 'rig', 'lights', 'materials'},
    'captions': {'id', 'type', 'position', 'opacity', 'in', 'out', 'font', 'weight', 'features', 'size', 'color', 'highlight', 'max_chars', 'lines',
                 'line_height', 'shadow', 'fade_in', 'fade_out'},
}
from . import safe as _safe
_safe.ATTRS |= {k for keys in TYPES.values() for k in keys if k.isidentifier()}      # layer("id").anything a layer can have
FILM_KEYS = {'size', 'fps', 'duration', 'background', 'words', 'audio', 'loudness', 'style', 'root', 'plugins', 'motion_blur', 'vars', 'poses'}
AUDIO_KEYS = {'id', 'src', 'at', 'from', 'to', 'volume', 'loudness', 'role', 'duck', 'fade_in', 'fade_out', 'audio_filter', 'mute'}
SIDES = {'left', 'right', 'top', 'bottom'}
SHADOW = dict(blur=6.0, offset=(0.0, 2.0), opacity=0.7, color=(0, 0, 0))
PATH_ITEM = {'shape', 'size', 'radius', 'points', 'd', 'closed', 'at', 'op', 'text', 'font', 'weight', 'features', 'font_size', 'tracking',
             'align', 'line_height', 'along', 'along_offset'}   # what an item of a shape's `paths` may say
GRADE = ('contrast', 'saturation', 'sepia', 'brightness', 'gamma', 'balance')
EASES = {'linear', 'in', 'out', 'inOut', 'hold', 'inQuad', 'outQuad', 'inOutQuad', 'inCubic', 'outCubic', 'inOutCubic',
         'inQuart', 'outQuart', 'inOutQuart', 'inExpo', 'outExpo', 'inOutExpo'}
EASE_POW = re.compile(r'^(in|out|inOut)Pow(\d+(?:\.\d+)?)$')       # inPow2.4: Remotion Easing.in(Easing.poly(2.4))


def _flat_pts(v):
    """[[x, y] × 4] → [x, y, …] (8 numbers), also inside keys {time: value} and {v: value}"""
    def f(x):
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
            return [float(c) for q in x for c in q]
        return x
    if isinstance(v, dict):
        return {k: ({**x, 'v': f(x.get('v'))} if isinstance(x, dict) else f(x)) if k != 'ease' else x for k, x in v.items()}
    return f(v)


class Issue:
    def __init__(self, level, msg, where=''):
        self.level, self.msg, self.where = level, msg, where

    def __str__(self):
        return f"{self.level.upper():5} {self.where + ': ' if self.where else ''}{self.msg}"


class DocError(Exception):
    def __init__(self, issues):
        self.issues = issues
        super().__init__('\n'.join(map(str, issues)))


def norm_word(w):
    return re.sub(r'[^\w]+', '', w.lower().replace('ё', 'е'))


def color(v):
    if isinstance(v, (list, tuple)):
        return tuple(float(c) for c in v[:3])
    if str(v).strip().lower() == 'transparent':               # CSS: rgba(0, 0, 0, 0) — its alpha is the stop's
        return (0, 0, 0)
    h = str(v).strip().lstrip('#')
    if re.fullmatch(r'[0-9a-fA-F]{3,4}', h):                   # #rgb / #rgba, as CSS
        h = ''.join(c * 2 for c in h)
    if not re.fullmatch(r'[0-9a-fA-F]{6}([0-9a-fA-F]{2})?', h):
        raise ValueError(f'цвет «{v}»: нужен #rrggbb (или #rgb), список [r, g, b] или transparent')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---------------- easing ----------------
BEZ = {'in': (0.42, 0, 1, 1), 'out': (0, 0, 0.58, 1), 'inOut': (0.42, 0, 0.58, 1)}


def bezier(u, p):
    """CSS-style cubic-bezier(x1, y1, x2, y2) evaluated at progress u"""
    x1, y1, x2, y2 = p
    t = u
    for _ in range(8):                                     # solve x(t) = u
        x = 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t ** 2 * x2 + t ** 3 - u
        dx = 3 * (1 - t) ** 2 * x1 + 6 * (1 - t) * t * (x2 - x1) + 3 * t ** 2 * (1 - x2)
        if abs(dx) < 1e-9:
            break
        t = min(1, max(0, t - x / dx))
    return 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t ** 2 * y2 + t ** 3


POW = {'inQuad': ('in', 2), 'outQuad': ('out', 2), 'inOutQuad': ('inOut', 2),
       'inCubic': ('in', 3), 'outCubic': ('out', 3), 'inOutCubic': ('inOut', 3),
       'inQuart': ('in', 4), 'outQuart': ('out', 4), 'inOutQuart': ('inOut', 4)}
_E10 = 1 - 2 ** -10


def expo(u, side):
    """exponential ease normalised to end exactly at 0 and 1 (the raw curve stops at 0.999: 1898 instead of 1900)"""
    if side == 'out':
        return 1.0 if u >= 1 else (1 - 2 ** (-10 * u)) / _E10
    if side == 'in':
        return 0.0 if u <= 0 else (2 ** (10 * (u - 1)) - 2 ** -10) / _E10
    return expo(2 * u, 'in') / 2 if u < 0.5 else 0.5 + expo(2 * u - 1, 'out') / 2


def ease(u, kind):
    """named curves (Remotion/AE: in/out/inOut × Quad/Cubic, CSS in/out/inOut) or [x1, y1, x2, y2] — a bezier like AE's velocity handles"""
    if isinstance(kind, (list, tuple)):
        return bezier(u, kind)
    if kind == 'linear':
        return u
    if kind == 'hold':
        return 0.0
    if kind in ('inExpo', 'outExpo', 'inOutExpo'):
        return expo(u, {'inExpo': 'in', 'outExpo': 'out', 'inOutExpo': 'inOut'}[kind])
    m = EASE_POW.match(kind) if kind not in POW and kind not in BEZ else None
    if m or kind in POW:
        side, p = (m[1], float(m[2])) if m else POW[kind]
        if side == 'in':
            return u ** p
        if side == 'out':
            return 1 - (1 - u) ** p
        return 2 ** (p - 1) * u ** p if u < 0.5 else 1 - (-2 * u + 2) ** p / 2
    return bezier(u, BEZ[kind])


# ---------------- formulas ----------------
def wiggle(freq, amp, seed=0, t=0.0):
    """smooth pseudo-random wobble (sum of incommensurate sines), like AE's wiggle — deterministic"""
    s = seed * 12.9898
    return amp * (0.5 * math.sin(t * freq * 6.2831 + s) + 0.3 * math.sin(t * freq * 2.3 * 6.2831 + 2 * s) +
                  0.2 * math.sin(t * freq * 5.1 * 6.2831 + 3 * s))


def spring(x, damping=10.0, stiffness=100.0, mass=1.0):
    """0 → 1 with overshoot, x seconds after the start (the spring of Remotion / react-spring, from rest)"""
    if x <= 0:
        return 0.0
    z, w0 = damping / (2 * math.sqrt(stiffness * mass)), math.sqrt(stiffness / mass)
    if z < 1:
        w1 = w0 * math.sqrt(1 - z * z)
        return 1 - math.exp(-z * w0 * x) * (z * w0 / w1 * math.sin(w1 * x) + math.cos(w1 * x))
    if z == 1:
        return 1 - math.exp(-w0 * x) * (1 + w0 * x)
    w2 = w0 * math.sqrt(z * z - 1)
    return 1 - math.exp(-z * w0 * x) * (z * w0 / w2 * math.sinh(w2 * x) + math.cosh(w2 * x))


def wob(x, amp, freq=1.8, decay=4.0):
    """a damped sine after a landing (AE's inertial bounce), x seconds after it"""
    return 0.0 if x < 0 else amp * math.sin(2 * math.pi * freq * x) * math.exp(-decay * x)


def iv(t, a, b, v0, v1, kind='outExpo'):
    """value between v0 and v1 while t goes a → b, clamped outside, along a named curve"""
    if b <= a:
        return v1 if t >= b else v0
    return v0 + (v1 - v0) * ease(min(1.0, max(0.0, (t - a) / (b - a))), kind)


def thousands(n):
    """35000 → «35 000» (the way ru-RU writes it, with a plain space)"""
    return f'{int(round(n)):,}'.replace(',', ' ')


EXPR_ENV = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
EXPR_ENV.update(abs=abs, min=min, max=max, round=round, int=int, str=str, clamp=lambda x, a, b: max(a, min(b, x)), wiggle=wiggle,
                spring=spring, wob=wob, iv=iv, ease=lambda kind, u: ease(min(1.0, max(0.0, u)), kind), thousands=thousands)


def _sound():
    from . import sound
    return sound


def np_interp(n, rows):
    """the moment of beat n (1 is the first; 2.5 — halfway from the second to the third)"""
    i = int(n) - 1
    return rows[i] + (rows[min(i + 1, len(rows) - 1)] - rows[i]) * (n - int(n))


def _track():
    from . import track
    return track


# ---------------- the film ----------------
_depth = threading.local()                                     # how deep formulas are asking one another (a loop is an error)


class LayerRef:
    """another layer inside a formula — After Effects' pick-whip: `layer("dial").rotation`, `layer("a").position[0]`,
    `layer("a").x`, `layer("a").world` (where its pivot is in the frame, through its parents), `layer("a").w` / `.h` (how
    wide and tall its content is — a plate that fits a text), any of its properties
    by name; `layer("a").at(t - 0.2).position` — as it was at another time (a follower that lags)"""

    def __init__(self, film, L, t):
        self._film, self._L, self._t = film, L, t

    def at(self, t):
        return LayerRef(self._film, self._L, float(t))

    def __getattr__(self, name):
        L, t = self._L, self._t
        if name in ('x', 'y'):
            return L['position'].at(t)[0 if name == 'x' else 1]
        if name == 'world':
            from .render import place
            p = place(self._film, L, t)
            return (p[0], p[1]) if p is not None else L['position'].at(t)
        if name in ('w', 'h'):                                 # its content's size at t, with its scale: a box that fits a text
            from .render import content_size
            sz = content_size(self._film, L, t)
            sc = L['scale'].at(t) if isinstance(L.get('scale'), Prop) else 1
            k = sc[0 if name == 'w' else 1] if isinstance(sc, tuple) else sc
            return float(sz[0 if name == 'w' else 1]) * abs(k)
        if name in ('in_', 'start'):
            return L['in']
        if name in ('out', 'end'):
            return L['out']
        v = L.get(name)
        if isinstance(v, Prop):
            return v.at(t)
        if isinstance(v, (int, float, tuple, str)) and not isinstance(v, bool):
            return v
        raise ValueError(f'у слоя «{L["id"]}» нет свойства «{name}», на которое ссылается формула')


class Prop:
    """constant, keyframes [(t, value, ease_of_segment_starting_here)], or a formula «=…» evaluated every frame with
    t (film seconds), lt (seconds since the layer began) and math (sin, cos, pi, …) plus wiggle(freq, amp, seed)"""

    def __init__(self, value=None, keys=None, expr=None, t_in=0.0):
        self.value, self.keys, self.expr, self.t_in = value, keys, expr, t_in
        self.film = None                                       # set when the layer is done: formulas may ask other layers

    @property
    def animated(self):
        return self.keys is not None or self.expr is not None

    def at(self, t):
        if self.expr is not None:
            env = {'__builtins__': {}, **EXPR_ENV, 't': t, 'lt': t - self.t_in}   # globals: a lambda sees them too
            if self.film is not None:                          # links: layer("id").position, var("name") — at this t
                film = self.film
                env['layer'] = lambda lid, _t=t: film.layer_ref(str(lid), _t)
                env['var'] = lambda name, _t=t: film.var(str(name), _t)
                env['amp'] = lambda src=None, _t=t, **k: _sound().amp(film, k.pop('t', _t), src, **k)     # sound: kadr/sound.py
                env['band'] = lambda lo, hi, src=None, _t=t, **k: _sound().band(film, k.pop('t', _t), lo, hi, src, **k)
                env['track'] = lambda lid, name=None, _t=t: _track().ref(film, lid, name, _t)             # footage: kadr/track.py
                env['pose'] = lambda name, what, _t=t: film.pose(str(name), str(what), _t)                # a puppet's poses: film.poses
                env['beat'] = lambda src=None, _t=t: _sound().beat_pos(film, _t, src)                     # where in the music's beats
                _depth.n = getattr(_depth, 'n', 0) + 1
                try:
                    if _depth.n > 48:
                        raise ValueError('формулы ссылаются друг на друга по кругу (layer / var)')
                    v = eval(self.expr, env)
                finally:
                    _depth.n -= 1
            else:
                v = eval(self.expr, env)
            if isinstance(v, str):
                return v                                       # a text made by a formula (a counter)
            return tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else float(v)
        if self.keys is None:
            return self.value
        k = self.keys
        t = t + 6e-4                                           # keys are written rounded to 1 ms: a frame never lands a hair early
        if t <= k[0][0]:
            return k[0][1]
        if t >= k[-1][0]:
            return k[-1][1]
        for (t0, v0, e), (t1, v1, _) in zip(k, k[1:]):
            if t0 <= t < t1:
                u = ease((t - t0) / (t1 - t0), e)
                if isinstance(v0, tuple):
                    return tuple(a + (b - a) * u for a, b in zip(v0, v1))
                return v0 + (v1 - v0) * u

    def times(self):
        return [k[0] for k in self.keys] if self.keys else []


class Film:
    """a film document; `data` + `parent` make a nested composition (a comp layer): it shares the parent's words,
    fps, root and style, and has its own size and layers"""

    def __init__(self, path=None, data=None, parent=None):
        try:
            self._load(path, data, parent)
        except DocError:
            raise
        except (TypeError, ValueError, AttributeError, KeyError, IndexError) as e:   # the last net: never a Python trace instead of words
            if os.environ.get('KADR_DEBUG'):
                raise
            if type(e) is ValueError and re.search('[а-яё]', str(e)):          # said in words where it was raised (a missing file, a clip without picture)
                raise DocError([Issue('error', str(e), '')]) from None
            raise DocError([Issue('error', f'в документе значение не того вида ({type(e).__name__}: {e}) — скорее всего в разделе film: число записано словом, '
                                           f'список — одним значением или наоборот. KADR_DEBUG=1 покажет место в разборе', str(path or ''))]) from None

    def _load(self, path=None, data=None, parent=None):
        self.path = os.path.abspath(path) if path else (parent.path if parent else os.path.abspath('film.yaml'))
        self.dir = os.path.dirname(self.path)
        self.issues = []
        self._links, self._by_id, self.vars, self._parent_film = [], {}, {}, parent
        raw = yaml.safe_load(open(path, encoding='utf-8')) if path else data
        self._from_data = path is None and parent is None       # built in code: its author's own (may_read)
        f = dict(raw.get('film') or {})
        if parent is not None:
            for k in ('fps', 'style', 'root', 'plugins'):
                f.setdefault(k, getattr(parent, '_film_' + k, None) if k in ('style', 'plugins') else
                             {'fps': parent.fps, 'root': parent.root}[k])
            f.setdefault('size', [parent.W, parent.H])
            f.setdefault('background', None)
        self._film_style, self._film_plugins = f.get('style', ''), f.get('plugins') or []
        for k in f:
            if k not in FILM_KEYS:
                self.issue('error', f'неизвестное поле film.{k}', 'film', k, FILM_KEYS)
        self.root = os.path.join(self.dir, os.path.expanduser(str(f['root']))) if 'root' in f else self.dir   # relative media paths start here
        self.W, self.H = map(int, f.get('size', [1920, 1080]))
        self.fps = float(f.get('fps', 25))
        if not (1 <= self.W <= 16384 and 1 <= self.H <= 16384 and self.W * self.H <= 8192 * 8192 and 1 <= self.fps <= 240):
            raise DocError([Issue('error', f'film: size {self.W}×{self.H} при {self.fps:g} кадр/с — допустимо до 16384 по стороне, до 67 млн точек, от 1 до 240 кадров в секунду', 'film')])
        depth_, up_ = 0, parent
        while up_ is not None:
            depth_, up_ = depth_ + 1, up_._parent_film
        if depth_ > 12:
            raise DocError([Issue('error', 'вставки (comp) вложены глубже 12 — скорее всего вставка включает саму себя', 'film')])
        bg = f.get('background', '#000000')
        self.background = color(bg) if bg is not None else None          # None (a comp's default): transparent
        self.words = parent.words if parent is not None and not f.get('words') else self.load_words(f.get('words'))
        from . import plugins
        named = [os.path.join(self.dir, os.path.expanduser(str(p))) for p in self._film_plugins]            # next to the document, not under root
        plugins.load(film=named)
        for d in named:                                        # a film's own plugins are programs: run only from places the user trusts
            if os.path.abspath(d) in plugins.REFUSED:
                self.issue('error', f'фильм подключает код из папки {d} (plugins) — это программа, и она не запущена: папка не в списке доверенных. '
                                    f'Если это ваша папка или вы прочли, что в ней: kadr trust "{os.path.abspath(d)}". Чужой фильм с кодом запускать так же осторожно, как чужую программу', 'film.plugins')
        self.templates = {**plugins.TEMPLATES, **(raw.get('templates') or {})}
        self.raw_templates = raw.get('templates') or {}
        self.by_style = set()                                      # ids of layers a style drew (not the document's own)
        shot_layers, shots_end = [], 0.0
        if raw.get('shots'):                                       # the film as shots in a channel style → plain layers
            from .shots import expand
            shot_layers, shots_end = expand(self, raw['shots'], f.get('style', ''))
        self.shots = raw.get('shots') or []
        self.shot_kinds = getattr(self, 'shot_kinds', {})
        if 'duration' in f:
            self.duration = self.time(f['duration'], 'film.duration')
        else:
            self.duration = shots_end or (self.words[-1][2] + 0.5 if self.words else 5.0)
        # a comp that names no duration has no end of its own: a layer in it without `out` lives as long as the comp
        # layer shows it (it used to end at the guess above — 5 s without a voice-over — and the comp went empty there,
        # silently; `time: =t` ran past it at once)
        self._open_end = parent is not None and 'duration' not in f and not shots_end
        if shot_layers:
            from .shots import clamp
            shot_layers = clamp(shot_layers, self.duration)
        self.loudness = f.get('loudness', -16)
        mb = f.get('motion_blur', parent.motion_blur if parent is not None else None)
        if mb is True:                                         # `motion_blur: true` — the usual 180° shutter ({} read as «off»)
            mb = {'shutter': 180}
        if mb and not isinstance(mb, dict):
            raise DocError([Issue('error', f'motion_blur: {mb!r} — нужно true (обычный затвор 180°) или {{shutter: градусы, max: копий, step: px}}', 'film')])
        self.motion_blur = None if not mb else dict(shutter=float(mb.get('shutter', 180)), max=int(mb.get('max', 32)),
                                                    step=float(mb.get('step', 2.5)))   # px between copies at most
        self.audio = [self.parse_audio(a, i) for i, a in enumerate(f.get('audio', []))]
        self._parent_film, self._links, self._by_id = parent, [], {}
        # film.vars — the film's own controls (After Effects' sliders): numbers, lists, keys or formulas any formula
        # reads with var("name"); one place to turn what many layers follow
        self.vars = {str(k): self.param(v, f'vars.{k}') for k, v in (f.get('vars') or {}).items()}
        self.poses = self.parse_poses(f.get('poses') or {})
        for v in self.vars.values():
            if isinstance(v, Prop):
                v.film = self
        self.layers = []
        self._mattes, self._parents, self._maps = [], [], []
        ids = set()
        listed, masks = [], {}
        for i, L in enumerate(shot_layers + (raw.get('layers') or [])):
            lid = str(L.get('id', f'layer{i}'))
            if isinstance(L.get('mask'), dict):                # a mask drawn as a curve: a hidden shape that is the layer's matte
                L, shape = self.curve_mask(lid, L)
                masks[shape['id']] = lid
                listed += [(lid, L), (shape['id'], shape)]
            elif L.get('cutout') and L.get('type') != 'puppet':   # the person cut out (kadr matte): a hidden copy playing the matte (a puppet reads its cut-out itself)
                L, hidden = self.cutout(lid, L)
                listed += [(lid, L)] + ([(hidden['id'], hidden)] if hidden else [])
            else:
                listed.append((lid, L))
        for lid, L in listed:
            if lid in ids:
                self.issue('error', 'id повторяется', lid)
            ids.add(lid)
            try:
                self.layers.append(self.parse_layer(lid, L))
            except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
                # a value of the wrong kind somewhere in the layer (a colour where a number goes, a word where a list
                # goes): said as a document error with the layer's name, not as a traceback out of the parser
                if os.environ.get('KADR_DEBUG'):
                    raise
                self.issue('error', f'значение не того вида: {e} — число записано словом или цветом, список — одним значением? (KADR_DEBUG=1 покажет место в разборе)', lid)
                continue
            if lid in masks:
                self.layers[-1]['mask_of'] = masks[lid]
        for lid, src in self._mattes:
            if src not in ids:
                self.issue('error', f'matte: нет слоя «{src}» в этой композиции', lid, src, ids)
            elif src == lid:
                self.issue('error', 'matte: слой не может быть маской самому себе', lid)
        self._by_id = {L['id']: L for L in self.layers}
        self.check_links()
        for lid, src in self._maps:                            # an effect's second picture: a layer of this composition
            if src not in ids:
                self.issue('error', f'effects map: нет слоя «{src}» в этой композиции', lid, src, ids)
            elif src == lid:
                self.issue('error', 'effects map: слой не может быть картой самому себе', lid)
        by_id = {L['id']: L for L in self.layers}
        for lid, pid in self._parents:                         # parent chains: the parent must exist, no loops
            if pid not in by_id:
                self.issue('error', f'parent: нет слоя «{pid}» в этой композиции', lid, pid, ids)
                continue
            by_id[lid]['parent_layer'] = by_id[pid]
        for L in self.layers:
            seen, q = {L['id']}, L.get('parent_layer')
            while q is not None:
                if q['id'] in seen:
                    self.issue('error', 'parent: слои ссылаются друг на друга по кругу', L['id'])
                    L.pop('parent_layer', None)
                    break
                seen.add(q['id'])
                q = q.get('parent_layer')
        self.camera = next((L for L in self.layers if L['type'] == 'camera'), None)
        errors = [x for x in self.issues if x.level == 'error']
        if errors and parent is None:
            raise DocError(errors)

    @property
    def frames(self):
        return int(round(self.duration * self.fps))

    def audio_filter(self, chain, where):
        """a sound's filter chain, checked: named ffmpeg filters with plain values only (kadr/safe.py)"""
        if not chain:
            return None
        try:
            return _safe.audio_filter(chain)
        except _safe.Refused as e:
            self.issue('error', str(e), where)
            return None

    def may_read(self, p):
        """a film reads the files of its own folder. A film that lies in a folder the user trusts (`kadr trust`; the same
        list a film's plugins are run by) reads anything — it is the user's own work. Otherwise a document from a
        stranger could draw any file of the machine into its frames (a key, a note, a photo) or have it read out in an
        error. A film built in code, with no document, is its author's own"""
        top = self
        while top._parent_film is not None:
            top = top._parent_film
        free = top.__dict__.get('_free')
        if free is None:
            from . import plugins
            free = top._free = top.__dict__.get('_from_data', False) or plugins.trusted(top.dir)
        if free:
            return True
        real = os.path.realpath(p)                             # (a link inside the folder that leads out of it is out of it)
        return any(real == b or real.startswith(b + os.sep) for b in (os.path.realpath(top.dir), os.path.dirname(os.path.realpath(__file__))))

    def resolve(self, p):
        p = os.path.expanduser(str(p))
        p = p if os.path.isabs(p) else os.path.join(self.root, p)
        if not self.may_read(p):
            seen = self.__dict__.setdefault('_refused', set())
            if p not in seen:
                seen.add(p)
                top = self
                while top._parent_film is not None:
                    top = top._parent_film
                self.issue('error', f'файл {p} лежит вне папки фильма, и он не прочитан: фильм из папки, которой не доверено, читает только её. '
                                    f'Если фильм ваш: kadr trust "{top.dir}" (или папку выше, где лежит вся работа) — тогда ему доступно всё, и свой код тоже', 'film')
            return os.path.join(self.dir, '.вне папки фильма', os.path.basename(p))
        return p

    def issue(self, level, msg, where='', key=None, choices=None):
        if key is not None and choices:
            near = difflib.get_close_matches(str(key), sorted(choices), n=1)
            if near:
                msg += f' (может быть, {near[0]}?)'
        self.issues.append(Issue(level, msg, where))

    def parse_audio(self, a, i):
        where = f'film.audio[{i}]'
        for k in a:
            if k not in AUDIO_KEYS:
                self.issue('error', f'неизвестное поле «{k}»', where, k, AUDIO_KEYS)
        src = self.resolve(a['src'])
        if not os.path.exists(src):
            self.issue('error', f'нет файла {a["src"]}', where)
        role = a.get('role', 'voice')
        if role not in ('voice', 'music', 'fx'):
            self.issue('error', f'role «{role}»: нужно voice, music или fx', where)
        dflt = {'voice': -17.0, 'music': -32.0, 'fx': None}.get(role)
        return dict(id=str(a['id']) if a.get('id') is not None else None,   # a name for formulas and sound layers: amp("id")
                    mute=bool(a.get('mute', False)),            # heard by formulas and sound layers, not put into the film's sound

                    src=src, role=role, at=self.time(a.get('at', 0), where), start=self.time(a.get('from', 0), where),
                    end=self.time(a['to'], where) if 'to' in a else None, volume=float(a.get('volume', 0)),
                    loudness=a.get('loudness', dflt), duck=float(a.get('duck', -6 if role == 'music' else 0)),
                    fade_in=float(a.get('fade_in', 0)), fade_out=float(a.get('fade_out', 0)), where=where,
                    audio_filter=self.audio_filter(a.get('audio_filter'), where))       # any ffmpeg audio filter chain: eq, compressor, echo, pitch…

    # ---------- words ----------
    def load_words(self, src):
        if not src:
            return []
        near = os.path.join(self.dir, os.path.expanduser(src))             # timings usually sit next to the document
        p = near if os.path.exists(near) and self.may_read(near) else self.resolve(src)
        if not os.path.exists(p):
            self.issue('error', f'нет файла таймингов слов {src}', 'film.words')
            return []
        d = json.load(open(p, encoding='utf-8'))
        d = d.get('words', d) if isinstance(d, dict) else d
        # Whisper's {word, start, end} or the reels' words.json {w, t, e}
        d = [{'word': w['w'], 'start': w['t'], 'end': w['e']} if 'w' in w else w for w in d]
        return [(norm_word(w['word']), float(w['start']), float(w['end']), w['word'].strip()) for w in d if norm_word(w['word'])]

    def word_near(self, t):
        """the voice-over word spoken at (or last before) t — context for reports"""
        best = None
        for w, s, e, raw in self.words:
            if s <= t + 1e-6:
                best = raw
        return best

    # ---------- time ----------
    TIME = re.compile(r'^word:(?P<w>[^#.+\-]+)(?:#(?P<n>\d+))?(?P<end>\.end)?(?P<off>[+\-]\d+(?:\.\d+)?)?$')

    def time(self, expr, where=''):
        if isinstance(expr, (int, float)):
            return float(expr)
        s = str(expr).strip()
        try:
            return float(s.rstrip('s'))
        except ValueError:
            pass
        mb = re.match(r'^beat:(?:(?P<src>[A-Za-z_][\w.-]*):)?(?P<n>\d+(?:\.\d+)?)(?P<off>[+-]\d*\.?\d+)?$', s)
        if mb:                                                 # on the music's beat: beat:5, beat:8.5, beat:5+0.1, beat:intro:3
            try:
                rows = _sound().beats_of(self, mb['src'])[1]
            except ValueError as e:
                self.issue('error', f'«{s}»: {e}', where)
                return 0.0
            n = float(mb['n'])
            if len(rows) < 2 or n < 1 or n > len(rows):
                self.issue('error', f'«{s}»: в музыке {len(rows)} битов' if len(rows) >= 2 else f'«{s}»: в музыке не слышно ровного бита', where)
                return 0.0
            return float(np_interp(n, rows)) + float(mb['off'] or 0)
        m = self.TIME.match(s)
        if not m:
            self.issue('error', f'не понимаю время «{s}»: нужно число секунд или word:слово[#n][.end][±сек]', where)
            return 0.0
        if not self.words:
            self.issue('error', f'«{s}»: у фильма нет film.words — словам не к чему привязаться', where)
            return 0.0
        w, n = norm_word(m['w']), int(m['n'] or 1)
        hits = [x for x in self.words if x[0] == w]
        if len(hits) < n:
            near = difflib.get_close_matches(w, sorted({x[0] for x in self.words}), n=3)
            have = f'есть {len(hits)} вхождений' if hits else 'такого слова нет в озвучке'
            self.issue('error', f'«{s}»: {have}' + (f'; похожие: {", ".join(near)}' if near and not hits else ''), where)
            return 0.0
        _, st, en, _ = hits[n - 1]
        return (en if m['end'] else st) + float(m['off'] or 0)

    # ---------- properties ----------
    def prop(self, v, where, dim=None):
        """constant or {time: value, ..., ease: kind}; a key's value may be {v: value, ease: kind} for its own segment"""
        def val(x):
            if isinstance(x, (list, tuple)):
                try:
                    x = tuple(float(c) for c in x)
                except (TypeError, ValueError):                # [{0: …, 2: …}, 5]: keys inside a list
                    self.issue('error', 'в списке должны быть числа; ключи пишутся на всё свойство целиком: {0: [x, y], 2: [x, y]}, а не внутри списка', where)
                    return tuple(0.0 for _ in x)
                if dim and len(x) != dim:
                    self.issue('error', f'нужно {dim} числа, а не {len(x)}', where)
                return x
            try:
                return float(x)
            except (TypeError, ValueError):
                self.issue('error', f'нужно число, а не «{x}»', where)
                return 0.0

        if isinstance(v, str) and v.startswith('='):
            src = re.sub(r'wiggle\(([^)]*)\)', r'wiggle(\1, t=t)', v[1:])     # wiggle needs the frame's time
            try:
                p = Prop(expr=_safe.compile_formula(src, where))     # a formula only counts: kadr/safe.py
                heard = re.search(r'(?<![\w.])(amp|band|track)\(', src) is not None
                if 'layer(' in src or 'var(' in src or 'pose(' in src or heard:  # checked when every layer is known (check_links)
                    self._links.append((where, src))
                if heard:                                      # a layer that jumps with a sound or rides footage is meant to (kadr check: not a jerk)
                    self.__dict__.setdefault('sound_driven', set()).add(where.split('.')[0])
                return p
            except SyntaxError as e:
                self.issue('error', f'формула «{v}» не разбирается: {e.msg}', where)
            except _safe.Refused as e:
                self.issue('error', f'формула «{v}»: {e}', where)
                return Prop(value=0.0)
        if not isinstance(v, dict):
            return Prop(value=val(v))
        default = v.get('ease', 'inOut')
        if isinstance(default, list):
            default = tuple(map(float, default))
        elif default not in EASES and not EASE_POW.match(str(default)):
            self.issue('error', f'неизвестный ease «{default}»', where, default, EASES)
        keys = []
        for k, x in v.items():
            if k == 'ease':
                continue
            e = default
            if isinstance(x, dict):
                e, x = x.get('ease', default), x.get('v')
            keys.append((self.time(k, where), val(x), e))
        keys.sort(key=lambda k: k[0])
        for a, b in zip(keys, keys[1:]):
            if abs(a[0] - b[0]) < 1e-9:
                self.issue('error', f'два ключа на одном времени {a[0]:.2f} с', where)
        return Prop(keys=keys) if keys else Prop(value=None)

    def parse_layer(self, lid, L):
        from . import plugins
        t = L.get('type')
        custom = plugins.LAYER_TYPES.get(t)
        if t == 'shader':                                      # a layer its own shader draws: src | code, params, size
            sh = self.shader_of(L, lid, lid, COMMON | {'size', 'id'})
            if sh is None:
                return dict(id=lid, type='solid')
            from . import shader as _sh
            custom = (_sh.layer_fn(sh), sh.defaults(), 'gpu')
        if custom and custom[2] == 'gpu':
            for k in custom[1]:
                if k in COMMON or k in ('size', 'src', 'code'):
                    self.issue('error', f'шейдер слоя: параметр «{k}» называется как свойство слоя — дайте другое имя', lid)
        if t not in TYPES and not custom:
            self.issue('error', f'неизвестный type «{t}»', lid, t, set(TYPES) | set(plugins.LAYER_TYPES) | {'shader'})
            return dict(id=lid, type='solid')
        allowed = TYPES[t] if not custom else COMMON | {'size'} | set(custom[1]) | ({'src', 'code'} if t == 'shader' else set())
        for k in L:
            if k not in allowed:
                self.issue('error', f'у слоя {t} нет свойства «{k}»', lid, k, allowed)
        out = dict(id=lid, type=t, raw=L, parent=self)
        out['in'] = self.time(L['in'], lid) if 'in' in L else 0.0
        out['out'] = self.time(L['out'], lid) if 'out' in L else (math.inf if getattr(self, '_open_end', False) else self.duration)
        ec = L.get('echo')                                     # After Effects' Echo: the layer as it was a moment ago, under itself
        out['echo'] = None
        if ec is not None:
            ec = {'count': ec} if not isinstance(ec, dict) else ec
            for k in ec:
                if k not in ('count', 'step', 'decay'):
                    self.issue('error', f'echo: нет поля «{k}»', lid, k, {'count', 'step', 'decay'})
            out['echo'] = {'count': self.prop(ec.get('count', 4), f'{lid}.echo.count'), 'step': self.prop(ec.get('step', 2 / self.fps), f'{lid}.echo.step'),
                           'decay': self.prop(ec.get('decay', 0.6), f'{lid}.echo.decay')}
        out['posterize'] = float(L['posterize']) if L.get('posterize') else None   # Posterize Time: the layer moves at this fps
        out['visible'] = bool(L.get('visible', True))        # false: not drawn itself (and nothing parented to it) — it
        out['blend'] = L.get('blend', 'normal')              # still works as a map, a matte or a parent
        if out['blend'] not in BLENDS:
            self.issue('error', f'blend «{out["blend"]}»', lid, out['blend'], BLENDS)
        out['fade_in'] = float(L.get('fade_in', 0))
        out['fade_out'] = float(L.get('fade_out', 0))
        for k in ('wipe_in', 'wipe_out'):
            w = L.get(k)
            if w is not None:
                w = w if isinstance(w, dict) else {'dur': w}
                side = w.get('from', 'left')
                if side not in SIDES:
                    self.issue('error', f'{k}.from «{side}»: нужно left, right, top или bottom', lid)
                w = dict(dur=float(w.get('dur', 0.5)), side=side)
            out[k] = w
        if t == 'camera':
            out['position'] = self.prop(L.get('position', [0, 0, 0]), f'{lid}.position', 3)
            out['zoom'] = self.prop(L.get('zoom', 1000), f'{lid}.zoom')
            return out
        out['z'] = self.prop(L['z'], f'{lid}.z') if 'z' in L else None
        default_pos = [0, 0] if ('z' in L or t == 'shape') else ([self.W / 2, self.H * 0.9] if t == 'captions' else [self.W / 2, self.H / 2])
        if t == 'group':                                       # a group's children live in frame px; it turns around `anchor`
            out['pivot'] = self.prop(L.get('anchor', [self.W / 2, self.H / 2]), f'{lid}.anchor', 2)
            if not out['pivot'].animated:
                default_pos = list(out['pivot'].value)
            elif 'position' not in L:
                self.issue('error', 'у группы с анимированным anchor нужна position', lid)
        elif t != 'shape' and 'anchor' in L:                   # pivot of scale/rotation in content px (AE anchor point), animatable
            out['pivot'] = self.prop(L['anchor'], f'{lid}.anchor', 2)
        out['position'] = self.prop(L.get('position', default_pos), f'{lid}.position', 2)
        out['scale'] = self.prop(L.get('scale', 1), f'{lid}.scale')       # a number or [sx, sy] (−1 mirrors)
        out['rotation'] = self.prop(L['rotation'], f'{lid}.rotation') if 'rotation' in L else None
        out['skew'] = self.prop(L['skew'], f'{lid}.skew') if 'skew' in L else None              # degrees, along x
        for k3 in ('rotation_x', 'rotation_y'):               # a 3D turn of the layer's plane (needs no camera)
            out[k3] = self.prop(L[k3], f'{lid}.{k3}') if k3 in L else None
        out['perspective'] = float(L.get('perspective', 2000))  # CSS perspective: eye distance in px
        # CSS perspective-origin: where the eye looks from, content px (default: the anchor the plane turns about)
        out['persp_origin'] = self.prop(L['perspective_origin'], f'{lid}.perspective_origin', 2) if 'perspective_origin' in L else None
        out['backface'] = L.get('backface', 'visible')
        if out['backface'] not in ('visible', 'hidden'):
            self.issue('error', 'backface: visible или hidden', lid)
        if 'parent' in L:
            out['parent_id'] = str(L['parent'])
            self._parents.append((lid, out['parent_id']))
        if 'corner_pin' in L:                                  # the content box's corners pinned to 4 points (parent space)
            out['corner_pin'] = self.prop(_flat_pts(L['corner_pin']), f'{lid}.corner_pin', 8)
        out['mask'] = self.prop(L['mask'], f'{lid}.mask', 4) if 'mask' in L else None   # [x0, y0, x1, y1] frame px, animatable
        out['blur'] = self.prop(L['blur'], f'{lid}.blur') if 'blur' in L else None
        out['blur_space'] = L.get('blur_space', 'frame')      # layer: the blur in the layer's own px, squashed and turned with it (CSS)
        if out['blur_space'] not in ('frame', 'layer'):
            self.issue('error', 'blur_space: frame (размытие в кадре) или layer (сплющивается и поворачивается со слоем, как в CSS)', lid)
        sh = L.get('shadow', t == 'captions')
        out['shadow'] = []                                    # one shadow or a list (CSS: several); blur, offset, opacity animate
        for one in (sh if isinstance(sh, list) else [sh] if sh else []):
            sd = dict(SHADOW, **(one if isinstance(one, dict) else {}))
            if sd.get('space', 'frame') not in ('frame', 'layer'):
                self.issue('error', 'shadow.space: frame (сдвиг в кадре) или layer (сдвиг и размытие поворачиваются и сплющиваются со слоем, как в CSS и AE)', lid)
            out['shadow'].append({'color': color(sd['color']), 'outside': bool(sd.get('outside', False)), 'space': sd.get('space', 'frame'),
                                  'blur': self.prop(sd['blur'], f'{lid}.shadow.blur'), 'opacity': self.prop(sd['opacity'], f'{lid}.shadow.opacity'),
                                  'spread': self.prop(sd.get('spread', 0), f'{lid}.shadow.spread'),   # CSS box-shadow spread: px, < 0 shrinks
                                  'offset': self.prop(list(sd['offset']) if isinstance(sd['offset'], tuple) else sd['offset'], f'{lid}.shadow.offset', 2)})
        out['shadow'] = out['shadow'] or None
        flt = L.get('filter') or {}                            # CSS-like filter on the layer's own picture, animatable
        out['filter'] = [(k, self.prop(v, f'{lid}.filter.{k}')) for k, v in flt.items() if k in GRADE and k != 'balance'] or None
        for k in flt:
            if k not in GRADE or k == 'balance':
                self.issue('error', f'filter.{k}: есть brightness, contrast, saturation, sepia, gamma', lid, k, GRADE)
        out['opacity'] = self.prop(L.get('opacity', 1), f'{lid}.opacity')
        mt = L.get('matte')                                    # track matte: another layer's alpha or brightness
        if mt is not None:
            mt = {'src': str(mt)} if not isinstance(mt, dict) else {'src': str(mt.get('src', mt.get('layer', ''))), 'mode': mt.get('mode', 'alpha')}
            mt.setdefault('mode', 'alpha')
            if mt['mode'] not in MATTES:
                self.issue('error', f'matte.mode «{mt["mode"]}»', lid, mt['mode'], MATTES)
            self._mattes.append((lid, mt['src']))
        out['matte'] = mt
        out['effects'] = []
        for n, fx in enumerate(L.get('effects') or []):
            name = fx.get('type') if isinstance(fx, dict) else fx
            got = fx if isinstance(fx, dict) else {}
            d = self.own_shader(got, f'{lid}.effects[{n}]', lid) if name == 'shader' else plugins.EFFECTS.get(name)
            if d is None:
                if name != 'shader':
                    self.issue('error', f'effects[{n}]: нет эффекта «{name}»', lid, name, set(plugins.EFFECTS) | {'shader'})
                continue
            gpu = getattr(d[0], 'kadr_gpu', False)             # a shader (kadr/shader.py): may take `map: <layer id>`
            for k in got:
                if k != 'type' and k not in d[1] and not (gpu and k in ('map', 'src', 'code')):
                    self.issue('error', f'effects[{n}] {name}: нет параметра «{k}»', lid, k, d[1])
            e = {'def': d, 'name': name, 'p': {k: self.param(got.get(k, v), f'{lid}.effects.{name}.{k}') for k, v in d[1].items()}}
            if gpu and isinstance(e['p'].get('file'), str):    # a table the effect reads (a LUT): a file of the film
                if not e['p']['file'] or not os.path.exists(self.resolve(e['p']['file'])):
                    self.issue('error', f'effects[{n}] {name}: нет файла «{e["p"]["file"]}»', lid)
                else:
                    e['p']['file'] = self.resolve(e['p']['file'])
            if gpu and got.get('map') is not None:
                if not getattr(d[0], 'takes_map', False):
                    self.issue('error', f'effects[{n}] {name}: этот шейдер не берёт вторую картинку (map)', lid)
                else:
                    e['map'] = str(got['map'])
                    self._maps.append((lid, e['map']))
            out['effects'].append(e)
        out['effects'] = out['effects'] or None
        # `filter` is the layer's own picture, graded before any effect sees it: a green screen that saturation has
        # turned grey is no longer green, and the key takes nothing out — the layer just comes out grey, no error
        if out['effects'] and out['filter'] and any(e['name'] == 'chroma_key' for e in out['effects']):
            off = [k for k, _ in out['filter'] if k in ('saturation', 'sepia')]
            if off:
                self.issue('warn', f'filter.{off[0]} действует раньше эффектов: к ключу (chroma_key) цвет экрана придёт уже изменённым, и ключ его не уберёт. '
                                   f'Уберите {off[0]} из filter и сделайте то же после ключа — эффектом в effects за chroma_key '
                                   f'(tint: {{black: "#000000", white: "#ffffff"}} — чёрно-белое)', lid)
        if custom:
            out['plugin'] = (custom[0], custom[2])
            out['size'] = self.prop(L.get('size', [200, 200]), f'{lid}.size', 2)
            out['params'] = {k: self.param(L.get(k, v), f'{lid}.{k}') for k, v in custom[1].items()}
            return self.stamp_in(out)
        if t == 'solid':
            out['color'] = color(L.get('color', '#ffffff'))
            out['size'] = self.prop(L.get('size', [self.W, self.H]), f'{lid}.size', 2)
        elif t in ('image', 'video'):
            out['src'] = self.resolve(L['src'])
            if t == 'image' and '%' in str(L['src']):         # an image sequence: person/%d.png — frame = round(t·fps) + seq_offset
                out['seq'], out['seq_offset'] = out['src'], int(L.get('seq_offset', 0))
                out['src'] = out['seq'] % (int(round(out['in'] * self.fps)) + out['seq_offset'])
            if not os.path.exists(out['src']):
                self.issue('error', f'нет файла {L["src"]}', lid)
            out['fit'] = L.get('size', 'native' if t == 'image' else 'cover')
            out['crop'] = tuple(map(float, L['crop'])) if 'crop' in L else None
            g = L.get('grade') or {}
            for k in g:
                if k not in GRADE:
                    self.issue('error', f'grade.{k}: есть только {", ".join(GRADE)}', lid, k, GRADE)
            out['grade'] = tuple((k, (tuple(tuple(map(float, g[k].get(z, [0, 0, 0]))) for z in ('shadows', 'mids', 'highlights'))
                                      if k == 'balance' else float(g[k]))) for k in GRADE if k in g) or None
            out['flip'] = bool(L.get('flip', False))
            if t == 'image' and L.get('tile') is not None:     # the picture repeated over the whole box (an endless floor)
                if L['tile'] not in ('repeat', 'mirror'):
                    self.issue('error', f'tile «{L["tile"]}»: repeat или mirror', lid)
                if not isinstance(L.get('size'), (list, tuple)):
                    self.issue('error', 'tile: нужен size [w, h] — поле, которое плитка покрывает', lid)
                out['tile'] = L['tile']
                out['tile_scale'] = float(L.get('tile_scale', 1.0))
                out['tile_offset'] = tuple(map(float, L.get('tile_offset', [0, 0])))
            if t == 'video':
                rf = L.get('reframe')                           # the person kept in a narrower frame (kadr reframe)
                out['reframe'] = None
                if rf:
                    rf = {} if rf is True else rf
                    if not isinstance(rf, dict) or any(k not in ('smooth',) for k in rf):
                        self.issue('error', 'reframe: true или {smooth: секунды}', lid)
                        rf = {}
                    out['reframe'] = dict(smooth=float(rf.get('smooth', .6)))
                    out['reframe_src'] = self.resolve(L['reframe_of']) if L.get('reframe_of') else out['src']   # a cut-out's matte follows the footage's own person
                    if not os.path.exists(_track().sidecar(out['reframe_src'], 'frame')):
                        self.issue('error', f'неизвестно, где в клипе человек: сначала kadr reframe {L.get("reframe_of") or L["src"]}', lid)
                        out['reframe'] = None
                st = L.get('stabilize')                         # the camera's shake taken out (kadr stabilize, kadr/track.py)
                out['stab'] = None
                if st:
                    st = {} if st is True else st
                    if not isinstance(st, dict) or any(k not in ('smooth', 'lock', 'zoom') for k in st):
                        self.issue('error', 'stabilize: true или {smooth: секунды, lock: true | false, zoom: auto | число}', lid)
                        st = {}
                    zoom = st.get('zoom', 'auto')
                    out['stab'] = dict(smooth=float(st.get('smooth', 1.0)), lock=bool(st.get('lock', False)), zoom=zoom if zoom == 'auto' else float(zoom))
                    out['stab_src'] = self.resolve(L['stab_of']) if L.get('stab_of') else out['src']
                    if not os.path.exists(_track().sidecar(out['stab_src'], 'stab')):
                        self.issue('error', f'у клипа нет пути камеры: сначала kadr stabilize {L.get("stab_of") or L["src"]}', lid)
                        out['stab'] = None
                out['from'] = self.time(L.get('from', 0), lid)
                out['speed'] = float(L.get('speed', 1))
                # Time Remap: the source second shown at each film time — keys ({0: 0, 1: 0.2, 2: 3}: slow, then fast),
                # a formula, a hold (ease: hold), backwards. Then `from` / `speed` are not used and the clip is silent
                out['remap'] = self.prop(L['time'], f'{lid}.time') if 'time' in L else None
                out['frame_blend'] = bool(L.get('frame_blend', False))   # between two source frames: their mix, not the earlier
                out['volume'] = float(L.get('volume', 0)) if L.get('volume') is not None else None
                out['loudness'] = float(L['loudness']) if 'loudness' in L else None      # LUFS of the footage sound; turns it on
                out['max_gain'] = float(L.get('max_gain', 99))
                out['loop'] = bool(L.get('loop', False))
                out['until'] = self.time(L['until'], lid) if 'until' in L else None
                out['delogo'] = tuple(map(int, L['delogo'])) if 'delogo' in L else None
                out['sound_to'] = self.time(L['sound_to'], lid) if 'sound_to' in L else None   # source time where its sound stops
                out['audio_filter'] = self.audio_filter(L.get('audio_filter'), lid)
        elif t == 'shape':
            out.update(self.outline(L, lid, lid))
            if 'paths' in L:                                   # Merge Paths: outlines joined, cut, crossed one after another
                out['paths'] = []
                for n, it in enumerate(L['paths'] if isinstance(L['paths'], list) else []):
                    if not isinstance(it, dict):
                        self.issue('error', f'paths[{n}]: нужен словарь — d, points или shape + size, at, op', lid)
                        continue
                    for k in it:
                        if k not in PATH_ITEM:
                            self.issue('error', f'paths[{n}]: нет поля «{k}»', lid, k, PATH_ITEM)
                    one = self.outline(it, f'{lid}.paths[{n}]', lid)
                    one['op'] = it.get('op', 'union')
                    if one['op'] not in ('union', 'subtract', 'intersect', 'xor'):
                        self.issue('error', f'paths[{n}].op «{one["op"]}»: union, subtract, intersect или xor', lid)
                    one['at'] = self.prop(it.get('at', [0, 0]), f'{lid}.paths[{n}].at', 2)
                    out['paths'].append(one)
                if not out['paths']:
                    self.issue('error', 'paths: пустой список', lid)
            out['offset'] = self.prop(L['offset'], f'{lid}.offset') if 'offset' in L else None   # Offset Paths, px
            # a shape's `position` is where its own (0, 0) goes — a rect's top left corner — while a picture, a text and
            # a video are placed by their middle. `anchor: center` puts a rect / ellipse by its middle too
            an = L.get('anchor', [0, 0])
            if isinstance(an, str):
                if an != 'center' or out['shape'] not in ('rect', 'ellipse') or out['size'].keys or out['size'].expr is not None:
                    self.issue('error', f'anchor: «{an}» — у фигуры это [x, y] в её собственных пикселях или слово center '
                                        f'(только rect и ellipse с постоянным size: точка опоры в середине, position ставит середину)', lid)
                    an = [0, 0]
                else:
                    an = [out['size'].value[0] / 2, out['size'].value[1] / 2]
            L = {**L, 'anchor': list(an)} if 'anchor' in L else L
            rp = L.get('repeat')
            out['repeat'] = None
            if rp is not None:                                 # Repeater
                rp = {'count': rp} if not isinstance(rp, dict) else rp
                for k in rp:
                    if k not in ('count', 'offset', 'rotation', 'scale', 'anchor', 'opacity'):
                        self.issue('error', f'repeat: нет поля «{k}»', lid, k, {'count', 'offset', 'rotation', 'scale', 'anchor', 'opacity'})
                out['repeat'] = {'count': self.prop(rp.get('count', 3), f'{lid}.repeat.count'),
                                 'offset': self.prop(rp.get('offset', [0, 0]), f'{lid}.repeat.offset', 2),
                                 'rotation': self.prop(rp.get('rotation', 0), f'{lid}.repeat.rotation'),
                                 'scale': self.prop(rp.get('scale', 1), f'{lid}.repeat.scale'),
                                 'anchor': self.prop(rp.get('anchor', L.get('anchor', [0, 0])), f'{lid}.repeat.anchor', 2),
                                 'opacity': self.prop(rp.get('opacity', [1, 1]), f'{lid}.repeat.opacity', 2)}
            out['fill'] = self.paint(L['fill'], f'{lid}.fill') if L.get('fill') else None
            st = L.get('stroke')
            out['stroke'] = None if not st else dict(color=self.paint(st.get('color', '#ffffff'), f'{lid}.stroke.color'), width=float(st.get('width', 2)),
                                                     cap=st.get('cap', 'round'), join=st.get('join', 'round'),
                                                     dash=tuple(map(float, st['dash'])) if st.get('dash') else None,
                                                     dash_offset=self.prop(st.get('dash_offset', 0), f'{lid}.stroke.dash_offset'))   # marching ants
            out['trim'] = self.prop(L.get('trim', [0, 1]), f'{lid}.trim', 2)          # draw-on: [start, end] of the outline, 0..1
            out['arrow'] = L.get('arrow')                                              # 'end' | 'start' | 'both'
            out['anchor'] = tuple(map(float, an))                                       # the shape's point that sits at `position` (and the pivot of scale / rotation), shape px
            out['fill_rule'] = L.get('fill_rule', 'nonzero')                          # evenodd: overlaps cut holes (SVG)
            if out['fill_rule'] not in ('nonzero', 'evenodd'):
                self.issue('error', 'fill_rule: nonzero или evenodd', lid)
        elif t == 'comp':
            sub = {'film': {'size': L.get('size', [self.W, self.H]), 'background': L.get('background')}, 'layers': L.get('layers') or []}
            if L.get('src'):
                sub = yaml.safe_load(open(self.resolve(L['src']), encoding='utf-8'))
            out['film'] = Film(data=sub, parent=self)
            self.issues += [Issue(x.level, x.msg, f'{lid} → {x.where}') for x in out['film'].issues]
            out['time'] = self.prop(L.get('time', 0), f'{lid}.time')   # comp time at its in-point; keys / a formula: Time Remap
            out['speed'] = float(L.get('speed', 1))
            if not out['time'].animated and out['speed']:      # the comp's second → the film's (sound read inside a comp)
                out['film']._to_parent = lambda tc, a=out['in'], t0=out['time'].value, sp=out['speed']: a + (tc - t0) / sp
        elif t == 'gradient':
            out['kind'] = L.get('kind', 'linear')
            if out['kind'] not in ('linear', 'radial'):
                self.issue('error', f'kind «{out["kind"]}»: нужно linear или radial', lid)
            out['stops'] = tuple((float(st[0]), color(st[1]), float(st[2]) if len(st) > 2 else 1.0) for st in L.get('stops', []))
            if len(out['stops']) < 2:
                self.issue('error', 'нужно хотя бы две точки stops: [позиция 0..1, цвет, прозрачность]', lid)
            out['angle'] = float(L.get('angle', 180))
            out['center'] = self.prop(L.get('center', [0.5, 0.5]), f'{lid}.center', 2)
            out['radius'] = tuple(map(float, L.get('radius', [0.5, 0.5])))
            if out['kind'] == 'radial':                        # written in px, it is one flat colour — no error, nothing to see
                vals = [out['center'].value] if not (out['center'].keys or out['center'].expr) else [k[1] for k in out['center'].keys or []]
                far = [v for v in vals if isinstance(v, tuple) and any(abs(c) > 3 for c in v)]
                if far or any(r_ > 3 for r_ in out['radius']):
                    which = 'center' if far else 'radius'
                    got = list(far[0]) if far else list(out['radius'])
                    self.issue('warn', f'{which}: {got} — у градиента center и radius задаются долями рамки (0…1: center [0.5, 0.5] — середина, '
                                       f'radius [0.5, 0.5] — до краёв), а не в пикселях; так выйдет ровная заливка одним цветом. '
                                       f'Рамка — весь кадр или size слоя', lid)
            out['box'] = self.prop(L['size'], f'{lid}.size', 2) if 'size' in L else None   # a gradient in its own box (else the whole frame)
            out['premul'] = self.interp_mode(L, lid)
        elif t == 'particles':                                 # a picture breaking into grains (or gathered from them)
            out['src'] = self.resolve(L['src'])
            if not os.path.exists(out['src']):
                self.issue('error', f'нет файла {L["src"]}', lid)
            if not isinstance(L.get('size'), (list, tuple)):
                self.issue('error', 'particles: нужен size [w, h]', lid)
            out['psize'] = tuple(map(float, L.get('size', [100, 100])))
            out['crop'] = tuple(map(float, L['crop'])) if 'crop' in L else None
            out['grade'] = None
            out['mode'] = L.get('mode', 'blow')
            if out['mode'] not in ('blow', 'gather'):
                self.issue('error', f'particles.mode «{out["mode"]}»: blow или gather', lid)
            out['start'] = self.time(L['start'], lid) if 'start' in L else out['in']
            for k_, d in (('cell', 4.0), ('sweep', 0.6), ('direction', 0.0), ('spread', 250.0), ('turbulence', 300.0), ('gravity', 0.0),
                          ('life', 1.2), ('jitter', 0.15), ('shrink', 0.3)):
                out[k_] = float(L.get(k_, d))
            out['wind'] = tuple(map(float, L.get('wind', [900, -200])))
            out['seed'] = int(L.get('seed', 1))
        elif t == 'scene3d':                                   # a 3D scene: kadr/scene3d.py
            sz = L.get('size', [self.W, self.H])
            out['psize'] = (float(sz[0]), float(sz[1]))
            out['cell'] = 16.0
            out['samples'] = int(L['samples']) if 'samples' in L else None
            out['look'] = dict(L['look']) if isinstance(L.get('look'), dict) else None
            out['time'] = self.prop(L['time'], f'{lid}.time') if 'time' in L else None   # the scene's own time (default: the film's)
            out['made'], out['src'], out['clip'] = None, None, None
            own = str(L.get('src', '')).lower().endswith(('.yaml', '.yml'))      # a scene in Kadr's own words, kept in a file of its own
            if 'scene' in L or own:
                from . import scene3d
                d, base = L.get('scene'), None
                if 'scene' not in L:
                    sp = self.resolve(L['src'])
                    d, base = {}, os.path.dirname(os.path.abspath(sp))
                    self.__dict__.setdefault('sources', set()).add(os.path.abspath(sp))      # the server re-reads the film when it changes
                    if not os.path.exists(sp):
                        self.issue('error', f'нет файла {L["src"]}', lid)
                    else:
                        try:
                            d = yaml.safe_load(open(sp, encoding='utf-8')) or {}
                            d = d.get('scene', d) if isinstance(d, dict) else d
                        except yaml.YAMLError as e:
                            self.issue('error', f'{L["src"]}: не читается как сцена ({str(e).splitlines()[0]})', lid)
                out['made'] = scene3d.parse(self, d, lid, base)
                for group in ('props', 'lprops'):
                    for item in out['made'][group]:
                        item[2].film, item[2].t_in = self, 0.0
                for v in out['made']['cam'].values():
                    if v is not None:
                        v.film, v.t_in = self, 0.0
            elif 'src' in L:
                out['src'] = self.resolve(L['src'])
                if not os.path.exists(out['src']):
                    self.issue('error', f'нет файла {L["src"]}', lid)
                if 'clip' in L:
                    out['clip'] = self.resolve(L['clip'])
                    if not os.path.exists(out['clip']):
                        self.issue('error', f'нет файла {L["clip"]}', lid)
            else:
                self.issue('error', 'scene3d: нужен scene: {nodes, lights, camera} или src: файл сцены', lid)
            if out['made'] is not None or out['src']:
                # a scene from a file directed here: its named nodes moved by keys and formulas (a puppet), its camera
                # set here, the moments the camera jumps. `rig: file.yaml` — a file with `nodes` (and `camera`) to start from
                over, cam, lit, mts = {}, {}, {}, {}
                for fig in (out['made'] or {}).get('figures') or ():      # the scene's block figures bring their own strings
                    from . import figure
                    over.update(figure.wire(self, fig, lid))
                if L.get('rig'):
                    rp = self.resolve(str(L['rig']))
                    self.__dict__.setdefault('sources', set()).add(os.path.abspath(rp))
                    if not os.path.exists(rp):
                        self.issue('error', f'нет файла {L["rig"]}', lid)
                    else:
                        rig = yaml.safe_load(open(rp, encoding='utf-8')) or {}
                        over.update(rig.get('nodes') or {})
                        cam = dict(rig.get('camera') or {})
                        lit, mts = dict(rig.get('lights') or {}), dict(rig.get('materials') or {})
                        for k, v in (rig.get('vars') or {}).items():      # what the rig's formulas share (the film's own win)
                            if str(k) not in self.vars:
                                self.vars[str(k)] = self.param(v, f'vars.{k}')
                                if isinstance(self.vars[str(k)], Prop):
                                    self.vars[str(k)].film = self
                over.update(L.get('nodes') or {})
                cam.update(L.get('camera') or {})
                lit.update(L.get('lights') or {})
                mts.update(L.get('materials') or {})
                names, lnames, mnames = set(), set(), set()
                if out['made'] is not None:
                    Js = out['made']['J']
                    names = {n.get('name') for n in Js['nodes'] if n.get('name')}
                    lnames = {n.get('name') for n in Js['lights'] if n.get('name')}
                    mnames = {n.get('name') for n in Js['mats'] if n.get('name')}
                elif os.path.exists(out['src']):
                    try:
                        Js = json.load(open(out['src'], encoding='utf-8'))
                        for q in [t_.get('path') for t_ in Js.get('texs', [])] + [(Js.get('env') or {}).get('src')]:     # files the scene names itself
                            if q and q not in ('studio', 'room') and not self.may_read(q if os.path.isabs(str(q)) else os.path.join(os.path.dirname(out['src']), str(q))):
                                self.issue('error', f'{L["src"]}: сцена ссылается на файл вне папки фильма ({q})', lid)
                        names = {n.get('name') for n in Js.get('nodes', []) if n.get('name')}
                        lnames = {n.get('name') for n in Js.get('lights', []) if n.get('name')}
                        mnames = {n.get('name') for n in Js.get('mats', []) if n.get('name')}
                    except (ValueError, OSError):
                        self.issue('error', f'{L["src"]}: не читается как сцена', lid)
                # its named lamps and materials turned from the document: how bright a lamp is; how see-through a material,
                # how much it glows, how dark its colour (shade: 1 — as it is)
                out['lit'], out['mts'] = [], []
                for group, known, allowed, dest in ((lit, lnames, ('intensity',), out['lit']), (mts, mnames, ('opacity', 'glow', 'shade'), out['mts'])):
                    what = 'lights' if dest is out['lit'] else 'materials'
                    for name, body in group.items():
                        if name not in known:
                            self.issue('error', f'{what}: в сцене нет «{name}»', lid, name, known)
                            continue
                        for key, val in (body or {}).items():
                            if key not in allowed:
                                self.issue('error', f'{what}.{name}: нет свойства «{key}» ({", ".join(allowed)})', lid, key, set(allowed))
                                continue
                            dest.append((str(name), key, self.prop(val, f'{lid}.{what}.{name}.{key}')))
                out['over'] = []
                for name, body in over.items():
                    if name not in names:
                        self.issue('error', f'nodes: в сцене нет узла «{name}»', lid, name, names)
                        continue
                    for key, val in (body or {}).items():
                        if key not in ('position', 'rotation', 'scale', 'visible'):
                            self.issue('error', f'nodes.{name}: нет свойства «{key}» (position, rotation, scale, visible)', lid, key, {'position', 'rotation', 'scale', 'visible'})
                            continue
                        pr = self.prop(float(val) if isinstance(val, bool) else val, f'{lid}.nodes.{name}.{key}', 3 if key in ('position', 'rotation') else None)
                        out['over'].append((str(name), key, pr))
                out['cam_over'] = None
                if cam:
                    for k in cam:
                        if k not in ('position', 'look_at', 'fov', 'lens', 'focus'):
                            self.issue('error', f'camera: нет поля «{k}»', lid, k, {'position', 'look_at', 'fov', 'lens', 'focus'})
                    if 'position' not in cam or 'look_at' not in cam:
                        self.issue('error', 'camera: нужны position и look_at', lid)
                    else:
                        out['cam_over'] = {k: self.prop(cam[k], f'{lid}.camera.{k}', 3 if k in ('position', 'look_at') else None) if k in cam else None
                                           for k in ('position', 'look_at', 'fov', 'lens', 'focus')}
                out['cuts'] = sorted(self.time(c, f'{lid}.cuts') for c in (L.get('cuts') or []))
                for pr in [x[2] for x in out['over'] + out['lit'] + out['mts']] + [v for v in (out['cam_over'] or {}).values() if v is not None]:
                    pr.film, pr.t_in = self, 0.0
        elif t == 'svg':                                       # a vector picture from a file: kadr/svg.py
            from . import svg
            svg.parse(self, L, lid, out)
        elif t == 'puppet':                                    # a picture bent by pins: kadr/puppet.py
            from . import puppet
            puppet.parse(self, L, lid, out)
        elif t == 'chart':                                     # bars, a line, a pie from numbers: kadr/chart.py
            from . import chart
            chart.parse(self, L, lid, out)
        elif t in ('spectrum', 'wave'):                        # sound as a picture: kadr/sound.py
            (_sound().parse_spectrum if t == 'spectrum' else _sound().parse_wave)(self, L, lid, out)
        elif t in ('emitter', 'plexus'):                       # particle systems: kadr/psys.py
            from . import psys
            (psys.parse_emitter if t == 'emitter' else psys.parse_plexus)(self, L, lid, out)
        elif t == 'dust':
            out['count'] = int(L.get('count', 46))
            out['seed'] = int(L.get('seed', 1))
            out['color'] = color(L.get('color', '#c4e2f2'))
            out['drift'] = tuple(map(float, L.get('drift', [0.15, 0.5, -0.08, -0.33])))   # px/frame: x from..to, y from..to
            out['sizes'] = tuple(map(float, L.get('sizes', [1.2, 3.4, 6, 16])))           # small from..to, big from..to (px)
            out['alphas'] = tuple(map(float, L.get('alphas', [0.14, 0.44, 0.05])))         # small from..to, big
        elif t == 'noise':
            fq = L.get('frequency', 0.85)
            out['frequency'] = tuple(map(float, fq)) if isinstance(fq, (list, tuple)) else (float(fq), float(fq))
            out['octaves'] = int(L.get('octaves', 2))
            out['seed'] = int(L.get('seed', 1))
            out['color'] = color(L['color']) if 'color' in L else None      # none: grey noise; a colour: that colour, alpha from noise
            out['levels'] = tuple(map(float, L.get('levels', [1, 0])))       # alpha = a·n + b (like an SVG colour matrix row)
        elif t in ('text', 'captions'):
            out['text'] = str(L.get('text', ''))
            if t == 'text' and out['text'].startswith('\\='):       # «\=…»: the sign itself, not a formula
                out['text'] = out['text'][1:]
            elif t == 'text' and out['text'].startswith('=') and out['text'].strip('= '):   # a counter: the text is a formula of t («=» alone is a text)
                out['text_expr'] = self.prop(out['text'], f'{lid}.text')
                out['text'] = ''                                    # drawn from text_expr at each frame
            if t == 'text' and L.get('animate'):
                out['animate'] = self.animator(L['animate'], lid)
            fonts = L.get('font', 'Arial')                     # a list: fallback for characters the first font lacks
            fonts = [str(x) for x in (fonts if isinstance(fonts, list) else [fonts])]
            if 'weight' in L:                                  # a variable font's weight axis (static fonts ignore it)
                fonts = [f"{x.partition('#')[0]}#wght={float(L['weight']):g}" for x in fonts]
            out['font'] = '|'.join(fonts)
            ft_ = L.get('features') or []                     # OpenType: [tnum, lnum, smcp] or {liga: false, tnum: true}
            ft_ = [f"{'' if on else '-'}{k}" for k, on in ft_.items()] if isinstance(ft_, dict) else [str(x) for x in (ft_ if isinstance(ft_, list) else [ft_])]
            for x in ft_:
                if not re.fullmatch(r'[-+]?[a-z0-9]{4}', x):
                    self.issue('error', f'features: «{x}» — нужен четырёхбуквенный тег OpenType (tnum, lnum, smcp, liga…)', lid)
            out['features'] = ','.join(ft_)
            self.glyphs(out['font'], out['text'] if t == 'text' else ' '.join(w_[3] for w_ in self.words), lid)
            out['size'] = float(L.get('size', 48 if t == 'text' else 44))
            out['color'] = color(L.get('color', '#ffffff'))
            out['align'] = L.get('align', 'center')
            out['line_height'] = float(L.get('line_height', 1.2))
            out['tracking'] = float(L.get('tracking', 0))
            st = L.get('stroke')
            out['stroke'] = dict(width=float(st if not isinstance(st, dict) else st.get('width', 2))) if st else None
            if t == 'captions':
                if not self.words:
                    self.issue('error', 'подписям нужны слова: укажите film.words', lid)
                out['highlight'] = color(L['highlight']) if 'highlight' in L else None
                out['max_chars'] = int(L.get('max_chars', 42))
                out['lines'] = int(L.get('lines', 2))
                from .captions import phrases
                out['phrases'] = phrases(self.words, out['max_chars'], out['lines'], out['in'], out['out'])
        return self.stamp_in(out)

    def glyphs(self, font, text, lid):
        """a character none of the layer's fonts has is drawn as an empty box, and nothing said so (an arrow and a
        rouble sign in a Cyrillic cut of a font, 16.11.2026): the report names them and the way out"""
        from .render import missing_glyphs
        try:
            miss = missing_glyphs(font, text)
        except (OSError, ValueError):                          # no such font: said where the font is read
            return
        if miss:
            names = ', '.join(f'«{c}» (U+{ord(c):04X})' for c in miss[:8]) + (f' и ещё {len(miss) - 8}' if len(miss) > 8 else '')
            first = font.split('|')[0].split('#')[0]
            spare = 'Inter-Regular' if 'inter' not in first.lower() and not missing_glyphs('Inter-Regular', ''.join(miss)) else None
            self.issue('warn', f'в шрифте {font.split("#")[0]} нет знаков {names} — на их месте будет пустой прямоугольник. Дайте запасной шрифт списком: '
                               + (f'font: [{first}, {spare}] — в {spare} (идёт с Kadr) они есть' if spare else f'font: [{first}, <шрифт, где эти знаки есть>]')
                               + ' (знак берётся из первого шрифта списка, где он есть) или замените знак', lid)

    def stamp_in(self, out):
        extra = [v for s_ in (out.get('shadow') or []) for v in s_.values()] + [v for _, v in (out.get('filter') or [])]
        extra += [v for fx in (out.get('effects') or []) for v in fx['p'].values()] + list((out.get('params') or {}).values())
        extra += [(out.get('stroke') or {}).get('dash_offset')] + list((out.get('repeat') or {}).values())
        extra += [v for it in (out.get('paths') or []) for v in it.values()]
        for v in list(out.values()) + extra:                   # formulas know when their layer began (lt)
            if isinstance(v, Prop):
                v.t_in = out['in']

        def link(v, depth=0):                                  # …and their film: layer("id"), var("name")
            if isinstance(v, Prop):
                v.film = self
            elif isinstance(v, dict) and depth < 6:
                for k, x in v.items():
                    if k not in ('raw', 'parent', 'parent_layer', 'film', 'def'):
                        link(x, depth + 1)
            elif isinstance(v, (list, tuple)) and depth < 6:
                for x in v:
                    link(x, depth + 1)
        link({k: v for k, v in out.items()})
        return out

    ANIM_PROPS = {'x': 0.0, 'y': 0.0, 'scale': 1.0, 'opacity': 1.0, 'blur': 0.0}

    def animator(self, a, lid):
        """a text animator (AE: an animator with a range selector): units (letters or words) come in one after another,
        each from its own `from` state to rest. from: {y: 22, opacity: 0, blur: 9, scale: 1.1} or per property
        {v: 22, dur: 0.33, ease: outExpo}; stagger — seconds between units; origin of a unit's scale: bottom | center"""
        where = f'{lid}.animate'
        for k in a:
            if k not in ('by', 'at', 'stagger', 'dur', 'ease', 'from', 'origin'):
                self.issue('error', f'нет поля «{k}»', where, k, {'by', 'at', 'stagger', 'dur', 'ease', 'from', 'origin'})
        by = a.get('by', 'letters')
        if by not in ('letters', 'words'):
            self.issue('error', 'by: letters или words', where)
        dur, e = float(a.get('dur', 0.33)), a.get('ease', 'outExpo')
        props = {}
        for k, v in (a.get('from') or {}).items():
            if k not in self.ANIM_PROPS:
                self.issue('error', f'from.{k}: есть x, y, scale, opacity, blur', where, k, self.ANIM_PROPS)
                continue
            d = v if isinstance(v, dict) else {'v': v}
            props[k] = (float(d.get('v', 0)), float(d.get('dur', dur)), d.get('ease', e))
        return dict(by=by, at=self.time(a.get('at', 0), where), stagger=float(a.get('stagger', 1 / self.fps)), props=props,
                    origin=a.get('origin', 'bottom'))

    def layer_ref(self, lid, t):
        L = self._by_id.get(lid)
        if L is None:
            raise ValueError(f'формула ссылается на слой «{lid}», которого нет в этой композиции')
        return LayerRef(self, L, t)

    def parse_poses(self, d):
        """film.poses — a puppet directed the way one tells an actor: «by this moment, over this long, be like this».
            poses:
              hero:
                defaults: {smile: .3, lean: .08, look_up: 0}
                keys:
                  - [0, 0, {smile: .5}]                         # [when, over how many seconds, what becomes what]
                  - ["word:золотая", .25, {smile: .9, look_up: 1}]
        A formula reads a value with pose("hero", "smile"). Each key eases every value it names from wherever it is at
        that moment towards the key's own, over the key's seconds (a smooth step); keys are applied in the order they are
        written, so a later one may catch a value still on its way"""
        out = {}
        if not isinstance(d, dict):
            self.issue('error', 'film.poses: {имя: {defaults: {…}, keys: [[когда, за сколько секунд, {…}], …]}}', 'film.poses')
            return out
        for name, body in d.items():
            where = f'film.poses.{name}'
            if not isinstance(body, dict) or any(k not in ('defaults', 'keys', 'from') for k in body):
                self.issue('error', 'нужно {defaults: {…} или from: файл рига, keys: [[когда, за сколько секунд, {…}], …]}', where)
                continue
            base = {}
            if body.get('from') == 'figure':                   # a block figure of a scene (kadr/figure.py): its values are the kit's
                from . import figure
                base = dict(figure.DEFAULTS)
            elif body.get('from'):                             # the values and what they are at rest come with the puppet's rig
                rp = self.resolve(str(body['from']))
                if not os.path.exists(rp):
                    self.issue('error', f"нет файла {body['from']}", where)
                else:
                    base = (yaml.safe_load(open(rp, encoding='utf-8')) or {}).get('defaults') or {}
            try:
                dflt = {str(k): float(v) for k, v in {**base, **(body.get('defaults') or {})}.items()}
            except (TypeError, ValueError):
                self.issue('error', 'defaults: каждому имени — число', where)
                dflt = {}
            tracks = {k: [] for k in dflt}                     # for each value: [(when, over, target)] in the order written
            for n, key in enumerate(body.get('keys') or []):
                if not (isinstance(key, (list, tuple)) and len(key) == 3 and isinstance(key[2], dict)):
                    self.issue('error', f'keys[{n}]: [когда, за сколько секунд, {{что: чему равно}}]', where)
                    continue
                try:
                    t0, over = self.time(key[0], f'{where}.keys[{n}]'), float(key[1])
                except (TypeError, ValueError):
                    self.issue('error', f'keys[{n}]: время и длительность — числа (время можно словом: word:…)', where)
                    continue
                for what, target in key[2].items():
                    if str(what) not in dflt:
                        self.issue('error', f'keys[{n}]: «{what}» нет в defaults', where, what, set(dflt))
                        continue
                    try:
                        tracks[str(what)].append((t0, over, float(target)))
                    except (TypeError, ValueError):
                        self.issue('error', f'keys[{n}].{what}: нужно число', where)
            out[str(name)] = dict(defaults=dflt, tracks=tracks, given=set(body.get('defaults') or {}) if isinstance(body.get('defaults'), dict) else set())
        return out

    def pose(self, name, what, t):
        f = self
        while f is not None:                                   # a comp reads its own poses, then its parent film's
            P = f.poses.get(name)
            if P is not None:
                if what not in P['defaults']:
                    raise ValueError(f'у позы «{name}» нет значения «{what}»')
                v = P['defaults'][what]
                for t0, over, target in P['tracks'][what]:
                    if t < t0:
                        continue
                    a = min(1.0, (t - t0) / over) if over > 0 else 1.0
                    v += (target - v) * (a * a * (3 - 2 * a))
                return v
            f = f._parent_film
        raise ValueError(f'формула читает pose("{name}", …), а в film.poses такой нет')

    def var(self, name, t):
        f = self
        while f is not None:                                   # a comp reads its own vars, then its parent film's
            if name in f.vars:
                v = f.vars[name]
                return v.at(t) if isinstance(v, Prop) else v
            f = f._parent_film
        raise ValueError(f'формула читает var("{name}"), а в film.vars такой нет')

    def check_links(self):
        """formulas that name a layer or a film variable: both must exist (and the layer's property too)"""
        names, f = set(), self
        while f is not None:
            names |= set(f.vars)
            f = f._parent_film
        for where, src in self._links:
            for q, lid, prop in re.findall(r'''layer\(\s*(['"])(.+?)\1\s*\)(?:\.at\([^)]*\))?(?:\.(\w+))?''', src):
                L = self._by_id.get(lid)
                if L is None:
                    self.issue('error', f'формула ссылается на слой «{lid}», которого нет в этой композиции', where, lid, set(self._by_id))
                elif prop and prop not in ('x', 'y', 'w', 'h', 'world', 'at', 'start', 'end', 'out', 'in_') and prop not in L:
                    self.issue('error', f'формула: у слоя «{lid}» нет свойства «{prop}»', where, prop,
                               {k for k, v in L.items() if isinstance(v, Prop)} | {'x', 'y', 'world'})
            for q, name in re.findall(r'''var\(\s*(['"])(.+?)\1\s*\)''', src):
                if name not in names:
                    self.issue('error', f'формула читает var("{name}"), а в film.vars такой нет', where, name, names)
            for q, name, q2, what in re.findall(r'''(?<![\w.])pose\(\s*(['"])(.+?)\1\s*,\s*(['"])(.+?)\3\s*\)''', src):
                f, P = self, None
                while f is not None and P is None:
                    P, f = f.poses.get(name), f._parent_film
                if P is None:
                    self.issue('error', f'формула читает pose("{name}", …), а в film.poses такой нет', where, name, set(self.poses))
                elif what not in P['defaults']:
                    self.issue('error', f'у позы «{name}» нет значения «{what}»', where, what, set(P['defaults']))
            if re.search(r'(?<![\w.])(amp|band)\(', src):   # a sound named in a formula must be there
                _sound().check_formula(self, src, where)
            for m in re.finditer(r'''(?<![\w.])track\(\s*(['"])(.+?)\1\s*(?:,\s*(['"])(.+?)\3\s*)?\)(?:\.at\([^)]*\))?(?:\.(\w+))?''', src):
                try:                                           # a track named in a formula: the video layer, its file, the name
                    _track().find(self, m.group(2), m.group(4))
                except ValueError as e:
                    self.issue('error', str(e), where)
                if m.group(5) and m.group(5) not in _track().PROPS:
                    self.issue('error', f'у трека нет свойства «{m.group(5)}»', where, m.group(5), set(_track().PROPS))

    def cutout(self, lid, L):
        """`cutout: true | {invert, feather}` on a video or image layer → the layer with a matte, and the hidden layer
        that is it: the file's matte (shot.matte.mp4 / photo.matte.png, made by `kadr matte`) in the same place, at the
        same time, steadied the same way"""
        from . import matte as mt
        spec = {} if L['cutout'] is True else L['cutout']
        out = {k: v for k, v in L.items() if k != 'cutout'}
        if L.get('type') not in ('video', 'image') or not isinstance(spec, dict) or any(k not in ('invert', 'feather') for k in spec):
            self.issue('error', 'cutout: true или {invert: true, feather: px} — у слоя video или image', lid)
            return out, None
        if L.get('matte') is not None:
            self.issue('error', 'у слоя и cutout, и matte: оставьте одно', lid)
            return out, None
        src = mt.out_path(str(L.get('src', '')))
        if not os.path.exists(self.resolve(src)):
            self.issue('error', f'у файла нет вырезки: сначала kadr matte {L.get("src")}', lid)
            return out, None
        same = ('type', 'position', 'z', 'scale', 'rotation', 'anchor', 'skew', 'rotation_x', 'rotation_y', 'perspective', 'perspective_origin', 'corner_pin',
                'parent', 'in', 'out', 'size', 'crop', 'flip', 'from', 'speed', 'loop', 'until', 'time', 'frame_blend', 'stabilize', 'seq_offset')
        hidden = {k: L[k] for k in same if k in L}
        hidden.update(id=f'{lid}.cutout', src=src)
        if L.get('stabilize'):
            hidden['stab_of'] = L['src']                       # its frames are moved by the footage's own camera path
        if L.get('reframe') and L.get('type') == 'video':
            hidden['reframe'], hidden['reframe_of'] = L['reframe'], L['src']   # …and slid with the footage's own person
        if spec.get('feather'):
            hidden['blur'] = float(spec['feather']) / 2
        out['matte'] = {'src': hidden['id'], 'mode': 'luma_inverted' if spec.get('invert') else 'luma'}
        return out, hidden

    def curve_mask(self, lid, L):
        """`mask: {d | points | shape… | paths, feather, invert, space}` → the layer with a matte, and the hidden shape
        layer that is it (After Effects' mask path: any outline of kadr/paths.py — morphing keys, merged paths,
        offset; `feather` px of soft edge; `space: layer` — drawn in the layer's own px and moving with it, else in
        the frame's, like the rectangle mask)"""
        m = dict(L['mask'])
        feather, invert, space = m.pop('feather', 0), bool(m.pop('invert', False)), m.pop('space', 'frame')
        if space not in ('frame', 'layer'):
            self.issue('error', 'mask.space: frame (координаты кадра) или layer (координаты слоя, маска едет с ним)', lid)
        if L.get('matte') is not None:
            self.issue('error', 'у слоя и mask-кривая, и matte: оставьте одно (маску можно вычесть в самой mask через paths)', lid)
        shape = {'id': f'{lid}.mask', 'type': 'shape', 'fill': '#ffffff', 'position': [0, 0], **m}
        if feather:                                            # the blur whose soft edge is about `feather` px wide
            half = (lambda v: v / 2 if isinstance(v, (int, float)) else v)
            shape['blur'] = {k: (half(v) if k != 'ease' else v) for k, v in feather.items()} if isinstance(feather, dict) else \
                (f'=({feather[1:]}) / 2' if isinstance(feather, str) and feather.startswith('=') else half(feather))
        for k in ('in', 'out'):
            if k in L:
                shape[k] = L[k]
        if space == 'layer':
            shape['parent'], shape['blur_space'] = lid, 'layer'
        out = {k: v for k, v in L.items() if k != 'mask'}
        out['matte'] = {'src': shape['id'], 'mode': 'alpha_inverted' if invert else 'alpha'}
        return out, shape

    def outline(self, S, where, lid):
        """the fields of one outline — a shape layer's own, or an item of its `paths` (kadr/paths.py): shape, size,
        radius, points / d (plain, or keys {time: value} that morph), a text's font and run along a curve"""
        o = {}
        o['shape'] = S.get('shape', 'text' if 'text' in S else 'path' if 'd' in S else 'polyline' if 'points' in S else 'rect')
        if o['shape'] not in ('rect', 'ellipse', 'line', 'polyline', 'polygon', 'path', 'text'):
            self.issue('error', f'{where}: shape «{o["shape"]}»: rect, ellipse, line, polyline, polygon, path или text', lid)
        o['size'] = self.prop(S.get('size', [100, 100]), f'{where}.size', 2)
        o['radius'] = self.prop(S.get('radius', 0), f'{where}.radius')    # corner radius; animates (a bar → a pill)
        o['closed'] = bool(S.get('closed', o['shape'] == 'polygon'))
        o['keys'], o['key_kind'] = None, None
        d, pts = S.get('d'), S.get('points', [])
        as_pts = lambda v: tuple(tuple(map(float, q)) for q in v)
        if isinstance(d, dict) or isinstance(pts, dict):       # a morph: {time: outline, …, ease}
            kd = d if isinstance(d, dict) else pts
            default = kd.get('ease', 'inOut')
            keys = []
            for k, v in kd.items():
                if k == 'ease':
                    continue
                e = default
                if isinstance(v, dict):
                    e, v = v.get('ease', default), v.get('v')
                keys.append((self.time(k, where), str(v) if isinstance(d, dict) else as_pts(v), e))
            keys.sort(key=lambda k: k[0])
            if not keys:
                self.issue('error', f'{where}: у морфинга нет ни одного ключа', lid)
                keys = [(0.0, '' if isinstance(d, dict) else (), default)]
            o['keys'], o['key_kind'] = keys, 'd' if isinstance(d, dict) else 'points'
            d, pts = (keys[0][1], []) if isinstance(d, dict) else (None, keys[0][1])
        o['points'] = [tuple(map(float, q)) for q in pts]
        o['d'] = d
        if o['shape'] == 'text':                               # a text's outlines: «Create Shapes from Text»
            o['text'] = str(S.get('text', ''))
            fonts = S.get('font', 'Arial')
            fonts = [str(x) for x in (fonts if isinstance(fonts, list) else [fonts])]
            if 'weight' in S:
                fonts = [f"{x.partition('#')[0]}#wght={float(S['weight']):g}" for x in fonts]
            o['font'] = '|'.join(fonts)
            self.glyphs(o['font'], o['text'], lid)
            ft_ = S.get('features') or []
            ft_ = [f"{'' if on else '-'}{k}" for k, on in ft_.items()] if isinstance(ft_, dict) else [str(x) for x in (ft_ if isinstance(ft_, list) else [ft_])]
            o['features'] = ','.join(ft_)
            o['font_size'] = self.prop(S.get('font_size', 48), f'{where}.font_size')
            o['tracking'], o['line_height'] = float(S.get('tracking', 0)), float(S.get('line_height', 1.2))
            o['align'] = S.get('align', 'left')
            o['along'] = str(S['along']) if S.get('along') else None            # Path Text: the curve the glyphs ride
            o['along_offset'] = self.prop(S.get('along_offset', 0), f'{where}.along_offset')
            try:
                from .render import font_file
                for x in fonts:
                    font_file(x.partition('#')[0])
            except Exception as e:
                self.issue('error', f'{where}: шрифт: {e}', lid)
        return o

    def own_shader(self, got, where, lid):
        """`{type: shader, src: file.wgsl | code: "…", …params}` → the effect's (fn, params, grow), None with an issue"""
        from . import shader
        sh = self.shader_of(got, where, lid, shader.RESERVED)
        return shader.effect_def(sh) if sh is not None else None

    def shader_of(self, got, where, lid, not_params):
        """the film's own shader of an effect entry or a shader layer: a file's `//!` header names its params; for
        inline code the entry's own keys (all but `not_params`) are the params. None with an issue"""
        from . import shader
        try:
            if got.get('src'):
                path = self.resolve(got['src'])
                if not os.path.exists(path):
                    self.issue('error', f'{where}: нет файла шейдера {got["src"]}', lid)
                    return None
                return shader.load(path)
            if got.get('code'):
                extra = {}
                for k, v in got.items():
                    if k in not_params or k in shader.RESERVED:
                        continue
                    pv = self.param(v, f'{where}.{k}')
                    extra[k] = v if isinstance(v, str) and v.startswith('#') else (pv.at(0.0) if isinstance(pv, Prop) else pv)
                return shader.parse(str(got['code']), 'inline', extra_params=extra)
            self.issue('error', f'{where}: шейдеру нужен src (файл .wgsl) или code (текст)', lid)
        except shader.ShaderError as e:
            self.issue('error', f'{where}: {e}', lid)
        return None

    def param(self, v, where):
        """a plugin's parameter: numbers, number lists, keyframes and formulas animate; a #colour becomes (r, g, b);
        anything else is passed as it is"""
        if isinstance(v, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', v):
            return color(v)
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.startswith('=')) or \
                (isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)) or \
                (isinstance(v, dict) and v and all(k == 'ease' or not isinstance(k, str) or k.startswith('word:') or k.startswith('=') for k in v)):
            return self.prop(v, where)
        return v

    def interp_mode(self, v, where):
        """how a gradient mixes its stops: premultiplied (default — as CSS: red → transparent fades red, not through
        black) or straight (colour and alpha apart — what Kadr did before 16.11.2026)"""
        m = v.get('interpolate', 'premultiplied')
        if m not in ('premultiplied', 'straight'):
            self.issue('error', f'interpolate «{m}»: premultiplied или straight', where)
        return m != 'straight'

    def paint(self, v, where):
        """a colour or {gradient: linear | radial, stops: [[pos, colour, alpha], …], from, to | center, radius}"""
        if not isinstance(v, dict):
            hx = str(v).lstrip('#')
            return color(v) + ((int(hx[6:8], 16) / 255,) if isinstance(v, str) and len(hx) == 8 else ())   # #rrggbbaa
        kind = v.get('gradient', 'linear')
        if kind not in ('linear', 'radial'):
            self.issue('error', f'gradient «{kind}»: linear или radial', where)
        def alpha(st):                                         # [pos, colour, alpha] or the colour's own #rrggbbaa
            if len(st) > 2:
                return float(st[2])
            hx = str(st[1]).strip().lstrip('#')
            return int(hx[6:8], 16) / 255 if isinstance(st[1], str) and len(hx) == 8 else 1.0
        stops = tuple((float(st[0]), color(st[1]), alpha(st)) for st in v.get('stops', []))
        if len(stops) < 2:
            self.issue('error', 'нужно хотя бы две точки stops: [позиция 0..1, цвет, прозрачность]', where)
        for k in v:
            if k not in ('gradient', 'stops', 'from', 'to', 'center', 'radius', 'interpolate'):
                self.issue('error', f'у градиента нет поля «{k}»', where, k, {'gradient', 'stops', 'from', 'to', 'center', 'radius', 'interpolate'})
        return {'kind': kind, 'stops': stops, 'premul': self.interp_mode(v, where), **{k: (tuple(map(float, v[k])) if k != 'radius' else float(v[k])) for k in ('from', 'to', 'center', 'radius') if k in v}}

    def key_times(self, layer):
        ts = set()
        for k in ('position', 'scale', 'z', 'zoom'):
            p = layer.get(k)
            if isinstance(p, Prop):
                ts.update(p.times())
        return sorted(ts)
