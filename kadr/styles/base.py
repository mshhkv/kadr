"""A channel style: how each kind of shot and overlay looks. A style turns one shot entry of the film document into the
plain layers Kadr renders (the way a motion-graphics template becomes a composition). Subclasses fill in the look."""
import math


class ShotError(Exception):
    pass


def r(x, n=3):
    return round(float(x), n)


class Style:
    name = ''
    ease = 'inOut'
    tail = 0.0              # extra seconds a move is timed over (an older engine rendered every card 0.5 s longer)

    # fields of the shot parts drawn by the style's methods; a style that takes more extends this (stamp: pos…)
    fields = {'stamp': ('line1', 'line2', 'rot', 'pos'), 'speaker': ('name', 'sub'), 'chapter': ('n', 'title', 'kicker'),
              'title': ('text', 'kicker', 'sub')}

    def __init__(self, fps):
        self.fps = fps
        self.film = None        # the film being built (set by Kadr): film.resolve(path), film.root, film.W, film.H
        self.prev = None        # (kind, shot, t0, t1) of the shot before the one being built, set by Kadr
        self.next = None        # the raw entry of the shot after it

    # -- helpers --
    def f2t(self, t0, n):
        return r(t0 + n / self.fps)

    def spring(self, n):
        """Remotion spring(damping 200, stiffness 100, mass 1): critically damped, frame n → 0..1"""
        t = max(0.0, n / self.fps)
        return 1 - (1 + 10 * t) * math.exp(-10 * t)

    def keys(self, t0, frames, fn, ease='linear', step=1):
        """a property sampled every `step` frames from frame 0 to `frames` (for springs and curves Kadr has no ease for)"""
        k = {self.f2t(t0, n): fn(n) for n in list(range(0, frames, step)) + [frames]}
        k['ease'] = ease
        return k

    def reveal(self, L, t0, delay, frames=18, dy=18, blur=14):
        """appear: opacity 0→1, blur → 0, a short rise, over `frames` after `delay` frames"""
        a, b = self.f2t(t0, delay), self.f2t(t0, delay + frames)
        x, y = L['position']
        L['opacity'] = {a: 0, b: L.get('opacity', 1), 'ease': self.ease}
        if blur:
            L['blur'] = {a: blur, b: 0, 'ease': self.ease}
        if dy:
            L['position'] = {a: [x, r(y + dy)], b: [x, y], 'ease': self.ease}
        return L

    def push(self, sid, src, t0, t1, z0, z1, ox_pct=50, oy_pct=50, **extra):
        """a full-frame picture pushing from z0 to z1 around the point (ox %, oy %), timed over the shot + tail; the point is
        both object-position (which part of an overflowing picture stays in frame) and transform-origin, as in CSS"""
        from PIL import Image
        iw, ih = Image.open(src).size
        k = max(1920 / iw, 1080 / ih)
        cx, cy = (1920 - iw * k) * ox_pct / 100 + iw * k / 2, (1080 - ih * k) * oy_pct / 100 + ih * k / 2
        ox, oy = 1920 * ox_pct / 100, 1080 * oy_pct / 100
        tail = t1 + self.tail
        pos = lambda z: [r(ox + z * (cx - ox)), r(oy + z * (cy - oy))]
        return {'id': sid, 'type': 'image', 'src': src, 'size': 'cover', 'in': r(t0), 'out': r(t1),
                'scale': {r(t0): r(z0, 4), r(tail): r(z1, 4), 'ease': 'linear'}, 'position': {r(t0): pos(z0), r(tail): pos(z1), 'ease': 'linear'}, **extra}

    def unsupported(self, what):
        raise ShotError(f'стиль {self.name} не умеет «{what}»')

    # -- what a style may implement --
    def photo(self, sh, t0, t1):
        self.unsupported('photo')

    def print(self, sh, t0, t1, aside=False):
        self.unsupported('print')

    def doc(self, sh, t0, t1):
        self.unsupported('doc')

    def chapter(self, sh, t0, t1):
        self.unsupported('chapter')

    def title(self, sh, t0, t1):
        self.unsupported('title')

    def stamp(self, sid, p, t0, t1):
        self.unsupported('stamp')

    def speaker(self, sid, p, t0, t1):
        self.unsupported('speaker')

    def source(self, sid, text, t0, t1):
        self.unsupported('source')

    footage_grade = {}
    omni_grade = {'saturation': 0.94, 'contrast': 1.03, 'gamma': 0.98}
    omni_loudness = (-30.0, 8.0)          # ambience of a generated take: LUFS, max gain
    line_loudness = -18.0                 # a presenter take
