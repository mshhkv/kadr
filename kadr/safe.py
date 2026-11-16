"""What a film's formula may be: arithmetic, and nothing a program could do with it.

A film is a document, and a document from a stranger must not be a program. Its formulas («=sin(t) * 10») are Python
expressions, and Python evaluated as it is can reach anything — through an object's hidden attributes
(`().__class__.__base__.__subclasses__()`), a format string, a generator's frame. So before a formula is compiled it
is read as a tree and only this is let through:

    numbers, strings, lists, tuples, dicts; arithmetic, comparisons, `a if b else c`; calls; indexing and slices;
    lambda (formulas use it for a local name); f-strings; names that do not begin with an underscore;
    attributes from a short list — a layer's properties (`layer("a").position`, `.x`, `.w`, `.at(t)`), a track's, a few
    harmless methods of a string (`"{:.1f}".format(x)` with a plain format, `.upper()`)

Comprehensions, generators, `:=`, awaiting, hidden names and every other attribute are refused, with the reason said.
What a formula can call is what the film gives it (kadr/doc.py: the math functions, layer, var, pose, amp, …); builtins
are not among it. A formula can still be slow or huge (`9 ** 9 ** 9`): that wastes a render, it does not leave it."""
import ast, re, string

# attributes any formula may read: a layer through layer("id"), a track through track("id"); kadr/doc.py adds every
# property a layer can have
ATTRS = {'x', 'y', 'w', 'h', 'world', 'at', 'in_', 'start', 'out', 'end', 'position', 'scale', 'rotation', 'corners', 'lost', 'q', 'local', 'map'}
STR = {'upper', 'lower', 'title', 'capitalize', 'strip', 'lstrip', 'rstrip', 'zfill', 'rjust', 'ljust', 'center', 'replace'}
NODES = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.keyword, ast.Starred,
         ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.Lambda, ast.arguments, ast.arg, ast.Attribute, ast.JoinedStr, ast.FormattedValue,
         ast.operator, ast.unaryop, ast.boolop, ast.cmpop)
WORDS = {'ListComp': 'перебор в скобках (for … in …)', 'SetComp': 'перебор в скобках (for … in …)', 'DictComp': 'перебор в скобках (for … in …)',
         'GeneratorExp': 'перебор в скобках (for … in …)', 'NamedExpr': 'присваивание :=', 'Await': 'await', 'Yield': 'yield', 'YieldFrom': 'yield'}


class Refused(ValueError):
    """a formula that asks for more than arithmetic"""


def _plain_format(s):
    """a format string that only places its arguments: «{:.1f}», «{0} из {1}», «{name}» — no «{0.attr}», no «{0[key]}»"""
    try:
        return all(f is None or ('.' not in f and '[' not in f) for _, f, spec, _ in string.Formatter().parse(s)) and \
            all(spec is None or '{' not in spec for _, _, spec, _ in string.Formatter().parse(s))
    except ValueError:
        return False


def check(src, attrs=None):
    """the formula's text (without the «=») → its tree, or Refused with what is not allowed in it. attrs: more attribute
    names this kind of formula may read"""
    tree = ast.parse(src.strip(), mode='eval')                 # SyntaxError is the caller's to report
    ok = ATTRS | (attrs or set())
    for n in ast.walk(tree):
        if not isinstance(n, NODES):
            raise Refused(f'в формуле нельзя: {WORDS.get(type(n).__name__, type(n).__name__)} — формула только считает')
        if isinstance(n, ast.Name) and n.id.startswith('_'):
            raise Refused(f'в формуле нельзя имя «{n.id}»: имена с подчёркивания закрыты')
        if isinstance(n, ast.arg) and n.arg.startswith('_'):
            raise Refused(f'в формуле нельзя имя «{n.arg}»: имена с подчёркивания закрыты')
        if isinstance(n, ast.keyword) and n.arg is not None and n.arg.startswith('_'):
            raise Refused(f'в формуле нельзя параметр «{n.arg}»')
        if isinstance(n, ast.Attribute):
            a = n.attr
            if a == 'format':                                  # a text made of numbers: only on a string written right here
                if not (isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) and _plain_format(n.value.value)):
                    raise Refused('в формуле .format можно только у строки, написанной тут же, и без обращений внутри скобок: "{:.1f}".format(x)')
            elif a in STR:
                pass
            elif a.startswith('_') or a not in ok:
                import difflib
                near = difflib.get_close_matches(a, sorted(ok), n=1)
                raise Refused(f'в формуле нельзя «.{a}»: через точку читаются только свойства слоя (layer("id").position, .x, .w, .at(t))' + (f' — может быть, .{near[0]}?' if near else ''))
    return tree


def compile_formula(src, where='<формула>', attrs=None):
    """the checked formula as code for eval"""
    return compile(check(src, attrs), where, 'eval')


# ---- sound: `audio_filter` of a film goes to ffmpeg, and ffmpeg's filters can write files (ametadata … file=), read
# them and open addresses (amovie). So a film names filters from this list only, with plain values: numbers, units and
# short words — no file names, no addresses, no nested graphs
AUDIO_FILTERS = {'volume', 'highpass', 'lowpass', 'bandpass', 'bandreject', 'equalizer', 'bass', 'treble', 'acompressor', 'alimiter', 'agate', 'aecho', 'chorus',
                 'flanger', 'aphaser', 'tremolo', 'vibrato', 'afade', 'atempo', 'loudnorm', 'dynaudnorm', 'stereowiden', 'extrastereo', 'crystalizer', 'deesser', 'compand',
                 'apulsator', 'acrusher', 'aexciter', 'asubboost', 'asoftclip', 'adeclick', 'anlmdn', 'afftdn', 'areverse', 'earwax', 'haas', 'surround', 'virtualbass'}
_AF_NUM = r'[+\-]?[0-9]+(?:\.[0-9]+)?[A-Za-z%]{0,3}'            # 320, -18dB, 0.5k
_AF_VALUE = re.compile(rf'(?:{_AF_NUM}(?:[|/]{_AF_NUM}){{0,40}}|[A-Za-z_]{{1,20}})')     # numbers (lists with | and /), or one short word


def audio_filter(chain):
    """a film's audio filter chain («highpass=f=320,acompressor=threshold=-18dB:ratio=4») → the same text, or Refused
    with what is not allowed in it"""
    chain = str(chain).strip()
    for part in chain.split(','):
        name, _, args = part.strip().partition('=')
        if name not in AUDIO_FILTERS:
            import difflib
            near = difflib.get_close_matches(name, sorted(AUDIO_FILTERS), n=1)
            raise Refused(f'audio_filter: фильтр «{name}» не из списка разрешённых' + (f' (может быть, {near[0]}?)' if near else '') +
                          f'; есть: {", ".join(sorted(AUDIO_FILTERS))}')
        for arg in (args.split(':') if args else []):
            key, eq, val = arg.partition('=')
            val = val if eq else key
            if (eq and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,30}', key)) or not _AF_VALUE.fullmatch(val):
                raise Refused(f'audio_filter: у «{name}» параметр «{arg}» — допустимы числа, единицы и короткие слова (имена файлов и адреса нельзя)')
    return chain


# ffmpeg and ffprobe read a film's media from the disk only: a «clip» that is really a playlist must not make them
# open an address on the network (put right before every -i that takes a film's file)
LOCAL = ['-protocol_whitelist', 'file,pipe']
