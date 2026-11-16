"""The block figure — `shape: figure` in a scene3d scene: a character from a few words, his strings included.

    - id: hero
      shape: figure
      position: [0, 0, 0]        where they stand (the soles); with `sits: true` — where they sit (the underside of the thighs)
      rotation: [0, 30, 0]
      height: 54                 up to the top of the head, in the scene's units (default 54.23: the series' hero)
      sits: true                 legs forward, as on a chair
      skin: "#e2a97f"
      top: "#ee3124"             the hoodie; pocket: true | false | its colour; stripes: "#ffffff" (two stripes on the chest);
                                 logo: picture.png (on the chest, its own transparency counts); sleeves: long | short
      pants: "#33508f"
      shoes: "#f1efe9"
      hair: "#2c1c13"            or {color, style: tufts | flat | long | bun | none, tufts: [[forward, tilt°], …]}
      hat: helmet                or {kind: helmet | cap | beanie, color} — comes down onto the head with the pose's `hat`
      eyes: "#0a0a0c"            brows: colour (the hair's by default); glasses: colour; sign: picture.png (shown in the eyes
                                 instead of the pupils while the pose's `sign` is ≥ .5 — dollars)
      build: {head: 1, wide: 1, arms: 1, legs: 1}      proportions: the head's size, the shoulders' width, how long the limbs are
      attach: {handR: [{shape: …}], head: […]}          anything hung on a part (in the figure's own units: the head is
                                 2.14 wide): a thing in a hand, a badge on the torso. Parts: torso, head, face, armL, armR,
                                 handL, handR, legL, legR, hat
      rest: {smile: .6, hat: 1}  how they are when no key says otherwise (the values below)
      pose: hero                 the name in film.poses that directs them (default: the id)
      voice: voice               the sound that opens the mouth: an id or a role in film.audio (default: the id; none — silent)
      seed: 7                    what is theirs by chance: the tufts, when they blink and breathe (default: from the id)

The parts are named after the figure: hero.torso, hero.head, hero.armL, hero.handR, … — a film may move them itself
(`nodes: {hero.head: {rotation: …}}`), though the strings below do it already.

A film directs a figure with poses, the way one tells an actor (film.poses):

    poses:
      hero: {from: figure, keys: [[0, 0, {smile: .5}], ["word:привет", .25, {armRf: 1.9, up: 1}]]}

the values: y (a hop, in head-heights' units), sag, lean / yaw / tilt (the body), armL / armR (out to the side, radians),
armLf / armRf (forward: 1.2 — hands on a desk, 1.9 — pointing up), legL / legR (forward: 1.57 — sitting), walk (0 … 1: steps
on the spot — move `position` yourself), type (typing, 0 … 1), hp / hy / hr (the head: nod, turn, tilt), lx / ly (where the
eyes look), lid (0 open … 1 shut; blinking is its own), up (brows raised), worry, smile, sign (≥ .5: the eyes' sign), hair
(blown up), hat (0 … 1: the hat comes down). The mouth opens with the voice. With no poses in the film a figure stays as
its `rest` says, breathing and blinking."""
import math, random, re, zlib

# how a figure is with nothing said: standing, arms down, a slight smile
DEFAULTS = dict(y=0, sag=0, lean=.08, yaw=0, tilt=0, armL=.14, armR=.14, armLf=.1, armRf=.1, legL=0, legR=0, walk=0, type=0, hp=.05, hy=0, hr=0,
                lx=0, ly=-.1, lid=.04, up=.2, worry=.1, smile=.3, sign=0, hair=0, hat=0)
KEYS = {'id', 'shape', 'position', 'rotation', 'scale', 'visible', 'height', 'sits', 'skin', 'top', 'pocket', 'stripes', 'logo', 'sleeves', 'pants', 'shoes',
        'hair', 'hat', 'eyes', 'brows', 'glasses', 'sign', 'build', 'attach', 'rest', 'pose', 'voice', 'seed'}
PARTS = ('torso', 'head', 'face', 'armL', 'armR', 'handL', 'handR', 'legL', 'legR', 'hat')
HAIR, HATS = ('tufts', 'flat', 'long', 'bun', 'none'), ('helmet', 'cap', 'beanie')
HAT_COLOR = {'helmet': '#f2b92b', 'cap': '#2b4fa8', 'beanie': '#3c8d5a'}
HEIGHT = 54.23                                                 # the series' hero: 11 cm to a unit


def _hex(v, k=(1, 1, 1)):
    """a colour as #rrggbb, each channel times k (a darker shade of the same cloth)"""
    from .doc import color
    return '#' + ''.join(f'{max(0, min(255, round(c * m))):02x}' for c, m in zip(color(v), k))


def units(build):
    """how tall the figure is in its own units, soles to the top of the head"""
    return 1.61 + 1.92 * build['head'] + 1.4 * build['legs']


def strings(fig, voice=None):
    """the figure's strings: what each part does as the pose's values change → {vars, nodes} (formulas a film reads).
    voice: the sound the mouth follows (None: it stays shut)"""
    N, P, v, g = fig['id'], fig['pose'], fig['var'], fig['geo']
    p = lambda k: f"pose('{P}', '{k}')"
    talk, opn, lid, smile, step = (f"var('{v}_{k}')" for k in ('talk', 'open', 'lid', 'smile', 'step'))
    ph = lambda k: f" + {g['phase'][k]:g}" if g['phase'][k] else ''
    V = {
        # how loud the voice is, as the series measures it: over a fifteenth of a second, a thirtieth ahead (the mouth
        # leads the sound a little), 1 — an ordinary loud syllable
        f'{v}_talk': f"=min(1, 1.4 * amp('{voice}', fall=0, window=0.0667, t=t + 0.0333))" if voice else '=0',
        f'{v}_open': f"=(lambda a: a * a * (3 - 2 * a))(clamp(({talk} - .1) / .75, 0, 1))",
        f'{v}_lid': f"=clamp({p('lid')} + (lambda q: sin(q / .13 * pi) if q < .13 else 0)((t + {g['blink'][0]:g}) % {g['blink'][1]:g}) * .95, 0, 1)",
        f'{v}_smile': f"={p('smile')} - {p('worry')} * .3",
        f'{v}_step': f"=sin(t * 7{ph('step')}) * {p('walk')}",
    }
    bx = g['brow_x']
    n = {
        'body': dict(position=f"=[0, {g['y0']:g} + {p('y')} - {p('sag')} * .109 + abs({step}) * .05, 0]"),
        'torso': dict(rotation=f"=[degrees({p('lean')} + {p('sag')} * .1 + sin(t * 1.9{ph('breath')}) * .006), degrees({p('yaw')}), degrees({p('tilt')})]",
                      scale=f"=[1, 1 + sin(t * 1.9{ph('breath')}) * .012, 1]"),
        'armL': dict(rotation=f"=[degrees(-{p('armLf')} + sin(t * 15) * {p('type')} * .09 + {step} * .45), 0, degrees(-{p('armL')})]"),
        'armR': dict(rotation=f"=[degrees(-{p('armRf')} + sin(t * 13 + 2) * {p('type')} * .09 - {step} * .45), 0, degrees({p('armR')})]"),
        'legL': dict(rotation=f"=[degrees(-{p('legL')} - {step} * .5), 0, 0]"),
        'legR': dict(rotation=f"=[degrees(-{p('legR')} + {step} * .5), 0, 0]"),
        'head': dict(rotation=f"=[degrees({p('hp')} + {talk} * sin(t * 8.5) * .022 + sin(t * 1.3{ph('sway')}) * .012), degrees({p('hy')} + sin(t * .7{ph('sway')}) * .03), "
                              f"degrees({p('hr')} + sin(t * .9{ph('sway')}) * .014)]"),
        'pupilL': dict(position=f"=[{p('lx')} * .1, {p('ly')} * .1, .05]", visible=f"={p('sign')} < .5" if g['sign'] else None),
        'pupilR': dict(position=f"=[{p('lx')} * .1, {p('ly')} * .1, .05]", visible=f"={p('sign')} < .5" if g['sign'] else None),
        'signL': dict(visible=f"={p('sign')} >= .5"), 'signR': dict(visible=f"={p('sign')} >= .5"),
        'lidL': dict(scale=f"=[1, max({lid} * .66, .001), 1]", position=f"=[0, .33 - {lid} * .33, .08]", visible=f"={lid} > .02"),
        'lidR': dict(scale=f"=[1, max({lid} * .66, .001), 1]", position=f"=[0, .33 - {lid} * .33, .08]", visible=f"={lid} > .02"),
        'browL': dict(rotation=f"=[0, 0, degrees({p('worry')} * .34 - {p('smile')} * .05)]", position=f"=[{-bx:g}, 1.42 + {p('up')} * .16 + {p('worry')} * .04, 1.1]"),
        'browR': dict(rotation=f"=[0, 0, degrees(-{p('worry')} * .34 + {p('smile')} * .05)]", position=f"=[{bx:g}, 1.42 + {p('up')} * .16 + {p('worry')} * .04, 1.1]"),
        'hair': dict(position=f"=[{p('hair')} * .5, {p('hair')} * 2.3, 0]", rotation=f"=[0, 0, degrees({p('hair')} * .5)]", visible=f"={p('hat')} < .95" if g['hat'] else None),
        'hat': dict(position=f"=[0, (1 - {p('hat')}) * 7, 0]", visible=f"={p('hat')} > .01"),
        'lipL': dict(position=f"=[-.16, {smile} * .03, 0]", rotation=f"=[0, 0, degrees(-{smile} * .36)]", visible=f"={opn} <= .06"),
        'lipR': dict(position=f"=[.16, {smile} * .03, 0]", rotation=f"=[0, 0, degrees({smile} * .36)]", visible=f"={opn} <= .06"),
        'mouthOpen': dict(scale=f"=[.8 + .2 * {opn}, .12 + .46 * {opn}, 1]", position=f"=[0, .12 - (.12 + .46 * {opn}) / 2, 0]", visible=f"={opn} > .06"),
        'teeth': dict(position="=[0, .03, .03]", scale=f"=[1, .5 + .5 * {opn}, 1]", visible=f"={opn} > .06"),
    }
    return dict(vars=V, nodes={f'{N}.{k}': {a: b for a, b in body.items() if b is not None} for k, body in n.items() if k in g['parts']})


def at_rest(fig):
    """what the strings say with the pose at rest, at the first moment, in silence → {part: {position, rotation, …}}"""
    S = strings(fig)
    env = dict(sin=math.sin, cos=math.cos, degrees=math.degrees, pi=math.pi, abs=abs, max=max, min=min, t=0.0,
               clamp=lambda x, lo, hi: max(lo, min(hi, x)), pose=lambda name, k: fig['rest'][k], amp=lambda *a, **k: 0.0)
    from . import safe
    calc = lambda f: eval(safe.compile_formula(f[1:], 'figure'), env)     # the same gate a film's formulas pass (kadr/safe.py)
    env['__builtins__'] = {}
    env['var'] = lambda name: calc(S['vars'][name])            # (as globals: a lambda inside a formula looks its names up there)
    out = {}
    for name, body in S['nodes'].items():
        out[name] = {k: calc(f) for k, f in body.items()}
    return out


def build(n, issue):
    """a figure's words → (the tree of plain nodes it is made of, what the film needs to direct it). issue(level, text,
    key=None, choices=None) takes the problems"""
    for k in n:
        if k not in KEYS:
            issue('error', f'фигурка: нет поля «{k}»', k, KEYS)
    N = str(n.get('id', 'figure'))
    if not re.fullmatch(r'[\w.\-]{1,64}', N):
        issue('error', f'фигурка: id «{N[:40]}» — нужны буквы, цифры, точка, дефис')
        N = 'figure'

    def num(key, dflt, lo, hi, src=n, what=''):
        try:
            x = float(src.get(key, dflt))
            if not lo <= x <= hi:
                raise ValueError
            return x
        except (TypeError, ValueError):
            issue('error', f'фигурка {N}: {what}{key} — число от {lo:g} до {hi:g}')
            return float(dflt)

    def hue(v, dflt, k=(1, 1, 1)):
        try:
            return _hex(dflt if v is None or v is True else v, k)
        except (TypeError, ValueError) as e:
            issue('error', f'фигурка {N}: {e}')
            return _hex(dflt, k)

    b = n.get('build') or {}
    if not isinstance(b, dict) or set(b) - {'head', 'wide', 'arms', 'legs'}:
        issue('error', f'фигурка {N}: build — {{head, wide, arms, legs}}, каждое — множитель около 1')
        b = {}
    B = dict(head=num('head', 1, .6, 1.6, b, 'build.'), wide=num('wide', 1, .7, 1.4, b, 'build.'), arms=num('arms', 1, .6, 1.5, b, 'build.'), legs=num('legs', 1, .5, 1.8, b, 'build.'))
    w, la, ll, sits = B['wide'], B['arms'], B['legs'], bool(n.get('sits', False))
    unit = num('scale', 0, 1e-6, 1e9) if 'scale' in n else num('height', HEIGHT, 1e-6, 1e9) / units(B)
    seed = n.get('seed')
    try:
        seed = zlib.crc32(N.encode()) % 100000 if seed is None else int(seed)
    except (TypeError, ValueError):
        issue('error', f'фигурка {N}: seed — целое число')
        seed = 0
    R = random.Random(seed)
    # what is the figure's by chance. Seed 0 is the series' hero himself: his blink and his breath, as the series drew them
    blink = (.4, 3.3) if seed == 0 else (round(R.uniform(0, 3), 2), round(R.uniform(2.8, 4.3), 2))
    phase = dict(breath=0, sway=0, step=0) if seed == 0 else dict(breath=round(R.uniform(0, 6.2), 2), sway=round(R.uniform(0, 6.2), 2), step=0)

    skin = hue(n.get('skin'), '#e2a97f')
    top = hue(n.get('top'), '#ee3124')
    pants, shoes = hue(n.get('pants'), '#33508f'), hue(n.get('shoes'), '#f1efe9')
    h = n.get('hair', '#2c1c13')
    h = {'color': h} if not isinstance(h, dict) else h
    if set(h) - {'color', 'style', 'tufts'}:
        issue('error', f'фигурка {N}: hair — цвет или {{color, style, tufts}}')
    style = str(h.get('style', 'tufts'))
    if h.get('color') in (False, 'none'):
        style = 'none'
    if style not in HAIR:
        issue('error', f'фигурка {N}: hair.style «{style}»', style, HAIR)
        style = 'tufts'
    hair = hue(h.get('color') if style != 'none' else None, '#2c1c13')
    brows = hue(n.get('brows'), hair)
    eyes = hue(n.get('eyes'), '#0a0a0c')
    hat = n.get('hat')
    hat = {'kind': hat} if isinstance(hat, str) else hat
    if hat is not None and (not isinstance(hat, dict) or set(hat) - {'kind', 'color'} or hat.get('kind', 'helmet') not in HATS):
        issue('error', f'фигурка {N}: hat — helmet, cap, beanie или {{kind, color}}', (hat or {}).get('kind') if isinstance(hat, dict) else None, HATS)
        hat = None
    sleeves = str(n.get('sleeves', 'long'))
    if sleeves not in ('long', 'short'):
        issue('error', f'фигурка {N}: sleeves — long или short', sleeves, ('long', 'short'))
        sleeves = 'long'

    rest = dict(DEFAULTS)
    if sits:
        rest.update(legL=1.5708, legR=1.5708)
    own = n.get('rest') or {}
    if not isinstance(own, dict):
        issue('error', f'фигурка {N}: rest — {{значение позы: число}}')
        own = {}
    for k, x in own.items():
        if k not in DEFAULTS:
            issue('error', f'фигурка {N}: в rest нет значения «{k}»', k, set(DEFAULTS))
            continue
        try:
            rest[k] = float(x)
        except (TypeError, ValueError):
            issue('error', f'фигурка {N}: rest.{k} — число')
    given = set(own) | ({'legL', 'legR'} if sits else set())

    P = lambda part: f'{N}.{part}'
    box = lambda part, size, r, pos, color, rough=None, **more: dict(id=P(part), shape='rbox', size=[round(float(x), 5) for x in size], radius=r, position=[round(float(x), 5) for x in pos], color=color,
                                                                      **({'roughness': rough} if rough is not None else {}), **more)
    # ---- the torso, the arms
    torso = [box('top', [2 * w, 1.5, 1.14], .18, [0, .75, 0], top, .8)]
    pk = n.get('pocket', True)
    if pk:
        torso.append(box('pocket', [1.3 * w, .5, .1], .05, [0, .28, .56], hue(None if pk is True else pk, top, (.8445, .857, .861) if pk is True else (1, 1, 1)), .85))
    if n.get('stripes'):
        st = hue(n['stripes'], '#ffffff')
        torso += [box('stripeL', [.18, .5, .05], .03, [-.16, 1, .58], st, .5), box('stripeR', [.18, .5, .05], .03, [.16, 1, .58], st, .5)]
    if n.get('logo'):
        torso.append(dict(id=P('logo'), shape='plane', size=[.9, .9], position=[0, 1, .585], map=str(n['logo']), alpha=True, unlit=True, cast=False, receive=False))
    ls = 1.42 * la                                              # a sleeve: from a little above the shoulder's pivot down
    for s, x in (('L', -1), ('R', 1)):
        if sleeves == 'long':
            arm = [box('sleeve' + s, [.62, ls, .72], .2, [0, .04 - ls / 2, 0], top, .8)]
        else:                                                   # a T-shirt: the cloth ends above the elbow
            c = .45 * ls
            arm = [box('sleeve' + s, [.62, c, .72], .2, [0, .04 - c / 2, 0], top, .8), box('forearm' + s, [.54, ls - c + .2, .62], .18, [0, .04 - c - (ls - c - .2) / 2, 0], skin, .48)]
        arm.append(box('hand' + s, [.54, .44, .58], .16, [0, .04 - ls - .14, 0], skin, .48))
        torso.append(dict(id=P('arm' + s), position=[x * (w + .3), 1.42, 0], children=arm))
    # ---- the head
    head = [box('face', [2.14, 1.92, 1.92], .34, [0, .96, .1], skin, .48)]
    locks, down = [], []
    if style != 'none':
        locks.append(box('hairCap', [2.3, .72, 2.08], .2, [0, 1.74, .02], hair, .62))
    if style == 'tufts':
        tf = h.get('tufts')
        try:
            tf = [(float(a), float(c)) for a, c in tf] if tf is not None else [(round(R.uniform(-.4, .5), 3), round(R.uniform(-22, 22), 3)) for _ in range(6)]
            if not tf:
                raise ValueError
        except (TypeError, ValueError):
            issue('error', f'фигурка {N}: hair.tufts — [[насколько вперёд, наклон в градусах], …]')
            tf = [(0, 0)] * 6
        for i, (z, tilt) in enumerate(tf):
            locks.append(box(f'tuft{i + 1}', [.62, .5, .62], .14, [-.92 + 1.85 * i / max(1, len(tf) - 1), 2.1, z], hair, .62, rotation=[0, 0, tilt]))
    elif style == 'bun':
        locks.append(dict(id=P('bun'), shape='sphere', radius=.5, position=[0, 2.2, -.62], color=hair, roughness=.62))
    elif style == 'long':                                       # hangs to the shoulders: stays when a hat is on
        down = [box('hairBack', [2.3, 1.7, .28], .12, [0, .9, -.94], hair, .62), box('hairSideL', [.2, 1.3, 1.1], .08, [-1.13, 1.05, -.42], hair, .62),
                box('hairSideR', [.2, 1.3, 1.1], .08, [1.13, 1.05, -.42], hair, .62)]
    if locks:
        head.append(dict(id=P('hair'), children=locks))
    if down:
        head.append(dict(id=P('hairDown'), children=down))
    sign = n.get('sign')
    for s, x in (('L', -.5), ('R', .5)):
        eye = [box('eyeWhite' + s, [.5, .6, .08], .04, [0, 0, 0], '#fbfbf7', .25), box('pupil' + s, [.26, .42, .05], .03, [0, -.01, .05], eyes, .15)]
        if sign:
            eye.append(dict(id=P('sign' + s), shape='plane', size=[.56, .56], position=[0, 0, .1], map=str(sign), alpha=True, unlit=True, visible=False, cast=False, receive=False))
        eye.append(box('lid' + s, [.56, 1, .06], .02, [0, .3168, .08], hue(None, skin, (.9425, .911, .882)), .5))
        head.append(dict(id=P('eye' + s), position=[x, .96, 1.06], children=eye))
    head += [box('browL', [.62, .13, .08], .04, [-.5, 1.456, 1.1], brows, .62), box('browR', [.62, .13, .08], .04, [.5, 1.456, 1.1], brows, .62)]
    head.append(dict(id=P('mouth'), position=[0, .36, 1.06], children=[
        box('lipL', [.36, .1, .05], .03, [-.16, 0, 0], '#4a2118'), box('lipR', [.36, .1, .05], .03, [.16, 0, 0], '#4a2118'),
        box('mouthOpen', [.66, 1, .05], .03, [0, .06, 0], '#3a1712', visible=False), box('teeth', [.56, .16, .05], .02, [0, .03, .03], '#fbfbf7', .3, visible=False)]))
    if n.get('glasses'):
        gl = hue(n['glasses'], '#15151a')
        ring = lambda s, x: dict(id=P('lens' + s), shape='torus', radius=.37, tube=.04, position=[x, .96, 1.2], color=gl, roughness=.35)
        head.append(dict(id=P('glasses'), children=[ring('L', -.5), ring('R', .5), box('bridge', [.3, .07, .06], .03, [0, 1.02, 1.2], gl, .35),
                                                   box('templeL', [.06, .07, 1.2], .03, [-1.09, 1.02, .62], gl, .35), box('templeR', [.06, .07, 1.2], .03, [1.09, 1.02, .62], gl, .35)]))
    if hat:
        kind = hat.get('kind', 'helmet')
        hc = hue(hat.get('color'), HAT_COLOR[kind])
        if kind == 'helmet':
            m = dict(color=hc, roughness=.34, clearcoat=.6)
            parts = [dict(id=P('hatDome'), shape='dome', radius=1.2, segments=[40, 20], position=[0, 1.98, .05], scale=[1.04, .8, 1.02], **m),
                     dict(id=P('hatBrim'), shape='cylinder', radius=[1.36, 1.42], height=.1, segments=40, position=[0, 1.98, .15], **m),
                     dict(id=P('hatRidge'), shape='rbox', size=[.36, .3, 2.2], radius=.1, position=[0, 2.9, .05], **m)]
        elif kind == 'cap':                                     # a band round the top of the head, the crown on it, a visor
            m = dict(color=hc, roughness=.8)
            parts = [dict(id=P('hatDome'), shape='dome', radius=1.2, segments=[40, 20], position=[0, 1.98, .08], scale=[1.1, .8, 1.08], **m),
                     dict(id=P('hatBand'), shape='cylinder', radius=[1.32, 1.34], height=.36, segments=40, position=[0, 1.83, .08], **m),
                     dict(id=P('hatVisor'), shape='rbox', size=[1.7, .09, 1.0], radius=.04, position=[0, 1.7, 1.8], **m),
                     dict(id=P('hatButton'), shape='sphere', radius=.13, position=[0, 2.95, .08], **m)]
        else:
            m = dict(color=hc, roughness=.95)
            parts = [dict(id=P('hatDome'), shape='dome', radius=1.2, segments=[40, 20], position=[0, 1.98, .05], scale=[1.04, .92, 1.02], **m),
                     dict(id=P('hatCuff'), shape='cylinder', radius=[1.3, 1.3], height=.36, segments=40, position=[0, 2.12, .05], color=_hex(hc, (.82, .82, .82)), roughness=.95),
                     dict(id=P('hatPom'), shape='sphere', radius=.3, position=[0, 3.3, .05], color=_hex(hc, (.82, .82, .82)), roughness=1)]
        head.append(dict(id=P('hat'), position=[0, 7, 0], visible=False, children=parts))
    torso.append(dict(id=P('head'), position=[0, 1.46, 0], **({'scale': B['head']} if B['head'] != 1 else {}), children=head))
    # ---- the legs: a leg is what an arm is — one block with a smaller block on its end, no knee; it hangs on the hip
    legs = []
    lg = 1.4 * ll
    for s, x in (('L', -1), ('R', 1)):
        legs.append(dict(id=P('leg' + s), position=[x * .5 * w, 1.07, -.05], children=[
            box('pant' + s, [.84, lg, .7], .2, [0, .35 - lg / 2, 0], pants, .88), box('shoe' + s, [.88, .28, .74], .13, [0, .35 - lg - .13, .01], shoes)]))
    tree = dict(id=N, scale=unit, children=[dict(id=P('body'), children=[dict(id=P('torso'), position=[0, 1.3, 0], children=torso), dict(id=P('legs'), children=legs)])])
    for k in ('position', 'rotation', 'visible'):
        if k in n:
            tree[k] = n[k]

    parts, byid = set(), {}
    def names(x):
        byid[x['id']] = x
        parts.add(x['id'][len(N) + 1:])
        for c in x.get('children') or []:
            names(c)
    names(tree)
    names_ = {'pose': str(n.get('pose', N)), 'voice': N if n.get('voice') in (None, False) else str(n['voice'])}
    for key, val in names_.items():                             # these go into formulas as text: a name must be nothing but a name
        if not re.fullmatch(r'[\w.\-]{1,64}', val):
            issue('error', f'фигурка {N}: {key} «{val[:40]}» — нужно имя: буквы, цифры, точка, дефис')
            names_[key] = N
    fig = dict(id=N, pose=names_['pose'], var=re.sub(r'\W', '_', N), voice=None if n.get('voice') in (None, False) else names_['voice'], voice_given='voice' in n,
               rest=rest, given=given,
               geo=dict(y0=-.72 if sits else lg - 1.15, brow_x=.5, blink=blink, phase=phase, sign=bool(sign), hat=bool(hat), parts=parts))
    for name, body in at_rest(fig).items():                     # each part as its strings hold it at rest
        for k, val in body.items():
            byid[name][k] = bool(val) if k == 'visible' else [round(float(x), 5) for x in val]
    # things hung on the figure's parts
    at = n.get('attach') or {}
    if not isinstance(at, dict):
        issue('error', f'фигурка {N}: attach — {{часть: [узлы]}}')
        at = {}
    for part, items in at.items():
        if part not in PARTS or P(part) not in byid:
            issue('error', f'фигурка {N}: attach: нет части «{part}»' + (' (у фигурки нет шапки)' if part == 'hat' else ''), part, PARTS)
            continue
        byid[P(part)].setdefault('children', []).extend(items if isinstance(items, list) else [items])
    return tree, fig


def wire(F, fig, where):
    """a film takes a figure on: its pose (made, if the film has none by that name), the mouth's voice, the strings'
    shared values → {part: {position, rotation, …}} for the layer to move"""
    from . import sound
    f, P = F, None
    while f is not None and P is None:                         # a comp's figure may be directed by its parent film
        P, f = f.poses.get(fig['pose']), f._parent_film
    if P is None:
        P = F.poses[fig['pose']] = dict(defaults={}, tracks={}, given=set())
    for k, x in fig['rest'].items():                           # at rest as the scene says — unless the film's own defaults say otherwise
        if k not in P['defaults'] or (k in fig['given'] and k not in P.get('given', ())):
            P['defaults'][k] = x
        P['tracks'].setdefault(k, [])
    voice = fig['voice'] if fig['voice_given'] else fig['id']
    if voice is not None:
        try:
            sound.sources(F, voice)
        except ValueError:
            if fig['voice_given'] and any(f_.audio for f_ in _chain(F)):
                F.issue('warn', f"фигурка {fig['id']}: звука «{voice}» в фильме нет — рот будет закрыт", where)
            voice = None
    S = strings(fig, voice)
    from .doc import Prop
    for k, x in S['vars'].items():
        if k not in F.vars:
            F.vars[k] = F.param(x, f'vars.{k}')
            if isinstance(F.vars[k], Prop):
                F.vars[k].film = F
    return S['nodes']


def _chain(F):
    while F is not None:
        yield F
        F = F._parent_film
