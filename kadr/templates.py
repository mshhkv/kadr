"""Templates: new kinds of shots written in YAML, no code. In the film (top-level `templates:`) or in a plugin folder:

    templates:
      lower_third:
        params: {name: '', sub: '', accent: '#ffab45'}          # defaults; a shot overrides them
        layers:                                                  # times relative to the shot's start; len = its length
          - {id: bar, type: solid, color: '{accent}', size: [4, 80], position: [92, 920], out: =len}
          - {id: name, type: text, text: '{name}', size: 46, align: left, position: [116, 900],
             opacity: {0.1: 0, 0.5: 1}, blur: {0.1: 8, 0.5: 0}}

    shots:
      - omni: clips/a.mp4
        to: word:Гэнда
      - lower_third: {name: Минору Гэнда, sub: штабной лётчик}   # a template is a kind of shot
        len: 3
        over: true                                               # drawn over the previous shot instead of replacing it

In strings, {param} is the shot's value. A string starting with «=» is a formula (math, len, fps, w, h, the params); a
formula that uses t (film time) or lt (time since the layer began) is kept and evaluated on every frame."""
import math, re

SAFE = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
SAFE.update(abs=abs, min=min, max=max, round=round, int=int, float=float, len=None)


def formula(expr, env):
    """evaluate «=…» at expansion time; None if it needs per-frame variables"""
    from . import safe
    try:
        return eval(safe.compile_formula(expr[1:], 'шаблон'), {'__builtins__': {}, **SAFE, **env})
    except NameError:
        return None


def is_keyframes(v):
    return isinstance(v, dict) and v and all(k == 'ease' or isinstance(k, (int, float)) or
                                             (isinstance(k, str) and (k.startswith('=') or _num(k))) for k in v)


def _num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def expand(film, name, tpl, sh, t0, t1):
    """→ plain layer dicts with film times"""
    p = dict(tpl.get('params') or {})
    own = sh.get(name)
    if isinstance(own, dict):
        p.update(own)
    elif own is not None:
        p['value'] = own
    env = {**p, 'len': t1 - t0, 'fps': film.fps, 'w': film.W, 'h': film.H}

    def sub(v, at_time=False):
        if isinstance(v, str):
            if '{' in v and not v.startswith('='):
                try:
                    v = v.format_map(p)
                except (KeyError, IndexError, ValueError):
                    pass
            if v.startswith('='):
                v2 = v.format_map(p) if '{' in v else v
                got = formula(v2, env)
                if got is None:
                    return v2                                   # per-frame formula: kept for the property
                v = got
            if at_time and isinstance(v, (int, float)):
                return round(t0 + float(v), 4)
            if at_time and isinstance(v, str) and _num(v):
                return round(t0 + float(v), 4)
            return v
        if isinstance(v, list):
            return [sub(x) for x in v]
        if isinstance(v, dict):
            if is_keyframes(v):
                return {(k if k == 'ease' else sub(k, True)): sub(x) for k, x in v.items()}
            return {k: sub(x) for k, x in v.items()}
        return v

    out = []
    for i, L in enumerate(tpl.get('layers') or []):
        L = {k: (sub(v, True) if k in ('in', 'out') else ([{kk: sub(vv) for kk, vv in c.items()} for c in v] if k == 'layers' else sub(v)))
             for k, v in L.items()}                              # a nested comp's own layers keep their comp-local times
        L['id'] = f"{sh['id']}_{L.get('id', i)}"
        L.setdefault('in', round(t0, 4))
        L.setdefault('out', round(t1, 4))
        out.append(L)
    return out
