"""Extending Kadr without touching it.

A plugin is a Python file in a plugin folder. It may add a channel style, new kinds of shots, or both:

    from kadr.plugins import style, element
    from kadr.styles.base import Style, r

    @style('mychannel')                   # film.style: mychannel
    class MyChannel(Style):
        ...

    @element('bignumber', style='mychannel', params={'value', 'unit', 'caption'})
    def bignumber(st, sh, t0, t1):        # st: the film's style object (fps, f2t, keys, reveal, stage, fonts, colours…)
        p = sh['bignumber']               # the shot's own entry; t0, t1: when the shot is on screen
        return [...layers...]             # or (layers, layers_on_top) — plain layer dicts, times in film seconds

An element registered without a style works in every style. YAML files in a plugin folder may hold `templates:` too
(see templates.py) — new elements with no code at all.

Lower down, a plugin may add what a layer is or what happens to it — the engine stays as it is:

    @layer_type('waveform', params={'samples': [], 'color': '#ffab45'})
    def waveform(canvas, p, w, h, fx):    # a Skia canvas in the layer's own units: its box is 0..w × 0..h
        ...                               # (draw='pixels': fn(p, w, h, fx) → array (h·fx.scale, w·fx.scale, 4) instead)

    @effect('glow', params={'radius': 12, 'strength': 0.8}, grow=lambda p, s: 3 * p['radius'] * s)
    def glow(rgb, a, p, fx):              # the layer's picture (colour 0..255, alpha 0..1); returns (rgb, a) or fx.grow(…)
        ...

A layer of a plugin type has the usual position, scale, rotation, opacity, blur, shadow, mask, matte, effects and a
`size: [w, h]`; numbers and lists in its params animate like any property. `effects:` on any layer lists effects in order:
`effects: [{type: glow, radius: 20}]`. Built-in effects live in kadr/effects.py and are registered the same way.

Plugin folders: kadr/styles (built in), ~/.kadr/plugins, every folder in $KADR_PLUGINS (separated by «:»), and the
folders a film lists in `film.plugins`."""
import glob, importlib.util, os, sys
import yaml

STYLES = {}                     # name → Style subclass
ELEMENTS = {}                   # (style or '*', kind) → (fn, params)
TEMPLATES = {}                  # name → template dict (from plugin YAML files)
LAYER_TYPES = {}                # name → (fn, params with defaults, 'skia' | 'pixels')
EFFECTS = {}                    # name → (fn, params with defaults, grow(p, scale) → px or None)
_loaded = set()


def style(name):
    def deco(cls):
        cls.name = name
        STYLES[name] = cls
        return cls
    return deco


def element(kind, style='*', params=None):
    def deco(fn):
        ELEMENTS[(style, kind)] = (fn, set(params or ()))
        return fn
    return deco


def layer_type(name, params=None, draw='skia', pure=False):
    """a new layer type. pure=True: the drawing depends only on its parameters and box (not on time) — the GPU
    renderer then draws it once in its own box, keeps it, and places it like a picture (moving, turning, zooming,
    motion blur) until a parameter changes"""
    def deco(fn):
        fn.kadr_pure = bool(pure)
        LAYER_TYPES[name] = (fn, dict(params or {}), draw)
        return fn
    return deco


def effect(name, params=None, grow=None):
    def deco(fn):
        EFFECTS[name] = (fn, dict(params or {}), grow)
        return fn
    return deco


def lookup(style_name, kind):
    return ELEMENTS.get((style_name, kind)) or ELEMENTS.get(('*', kind))


def kinds(style_name):
    return {k for (s, k) in ELEMENTS if s in (style_name, '*')}


def folders(extra=()):
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'styles')
    out = [here, os.path.expanduser('~/.kadr/plugins')]
    out += [p for p in os.environ.get('KADR_PLUGINS', '').split(os.pathsep) if p]
    out += list(extra)
    return [p for p in out if os.path.isdir(p)]


TRUST_FILE = os.path.expanduser('~/.kadr/trusted')
REFUSED = set()                                                # folders a film named whose code was not run


def trust_roots():
    """folders whose Python a film may bring in: one a line in ~/.kadr/trusted (`kadr trust папка`), and KADR_TRUST"""
    out = [p for p in os.environ.get('KADR_TRUST', '').split(os.pathsep) if p]
    if os.path.isfile(TRUST_FILE):
        out += [ln.strip() for ln in open(TRUST_FILE, encoding='utf-8') if ln.strip() and not ln.startswith('#')]
    return [os.path.realpath(os.path.expanduser(p)) for p in out]


def trusted(folder):
    """a style is a program. The ones that ship with Kadr and the ones in the user's own ~/.kadr/plugins (or
    KADR_PLUGINS) are theirs to run; a folder a FILM names is run only if the user has said they trust where it lies —
    otherwise a film from a stranger would be a program from a stranger"""
    real = os.path.realpath(folder)
    return any(real == r or real.startswith(r.rstrip(os.sep) + os.sep) for r in trust_roots())


def trust(folder):
    """remember the folder (and all inside it) as one whose plugins may run"""
    real = os.path.realpath(os.path.expanduser(folder))
    if not os.path.isdir(real):
        raise ValueError(f'нет папки {folder}')
    if not trusted(real):
        os.makedirs(os.path.dirname(TRUST_FILE), exist_ok=True)
        with open(TRUST_FILE, 'a', encoding='utf-8') as f:
            f.write(real + '\n')
    return real


def load(extra=(), film=()):
    """import every plugin file once (built-in styles are a package and import themselves). film: folders a film's
    own `plugins:` names — their Python runs only from trusted places (their templates, which are data, always load)"""
    from . import styles, effects  # noqa: F401  (register the built-in styles, elements and effects)
    from . import shader
    for name, sh in shader.builtin().items():                  # kadr/shaders/*.wgsl: an effect each, run on the GPU
        if name not in EFFECTS or getattr(EFFECTS[name][0], 'shader', None) is not sh:
            EFFECTS[name] = shader.effect_def(sh)
    for name, sh in shader.generators().items():               # kadr/shaders/gen/*.wgsl: a layer type each
        if name not in LAYER_TYPES or getattr(LAYER_TYPES[name][0], 'shader', None) is not sh:
            LAYER_TYPES[name] = (shader.layer_fn(sh), sh.defaults(), 'gpu')
    from . import effects_gpu  # noqa: F401  (effects of several GPU passes: bloom, lut, light_wrap)
    asked = {os.path.abspath(d) for d in film}
    for d in folders(list(extra) + list(film)):
        if os.path.abspath(d) == os.path.join(os.path.dirname(os.path.abspath(__file__)), 'styles'):
            continue
        code = sorted(glob.glob(os.path.join(d, '*.py')))
        if code and os.path.abspath(d) in asked and not trusted(d):
            REFUSED.add(os.path.abspath(d))                    # the film is told (kadr/doc.py); nothing of it is run
            code = []
        for f in code:
            key = os.path.abspath(f)
            if key in _loaded or os.path.basename(f).startswith('_'):
                continue
            _loaded.add(key)
            name = 'kadr_plugin_' + os.path.splitext(os.path.basename(f))[0]
            spec = importlib.util.spec_from_file_location(name, f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        for f in sorted(glob.glob(os.path.join(d, '*.yaml'))):
            key = os.path.abspath(f)
            if key in _loaded:
                continue
            _loaded.add(key)
            data = yaml.safe_load(open(f, encoding='utf-8')) or {}
            TEMPLATES.update(data.get('templates') or {})
