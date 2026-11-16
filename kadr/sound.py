"""Sound as a source (AE.md, M8): what a sound does — how loud it is, in which frequencies — read by formulas and
drawn by two layers (After Effects: Convert Audio to Keyframes, Audio Spectrum, Audio Waveform; Trapcode Sound Keys).

    formulas         amp(src, fall=0.12, window=0.05, norm=True)   how loud the sound is at this moment
                     band(lo, hi, src, fall=0.12, norm=True)       how loud its frequencies lo … hi Hz are (20 … 12000)
                     Both give 0 … 1, where 1 is the loudest this sound ever gets (norm=False: the level itself, a
                     full-scale sine is 0.707). `fall`: seconds the value takes to drop after a hit (it rises at
                     once) — what makes a pulse readable; 0: none. `window`: seconds the loudness is measured over.
                     t=…: at another moment (t=t + 0.1 — a tenth of a second ahead of the sound).
                       scale: =1 + 0.3 * band(40, 120)              a disc that jumps on the kick
                       opacity: =amp("voice") > 0.08                on while the voice speaks
      src            the `id` of a film.audio entry; a role — voice | music | fx (all entries of that role together);
                     the id of a video layer (its own sound); a file. None: the music, else everything in film.audio.
                     Where the entry stands in the film (`at`, `from`, `to`) is taken into account; its volume is not.
                     beat(src)                                     where in the music's row of beats this moment is: 5 on the
                     fifth beat, 5.5 halfway to the sixth — `=1 + .2 * exp(-6 * (beat() % 1))` pulses on every beat
      times          beat:5 — the second the music's fifth beat falls on, anywhere a time is written (in: beat:9, keys,
                     cuts: «on the beat»); beat:8.5 — between two; beat:5+0.1 — a tenth later. `kadr beats music.mp3`
                     prints the tempo and every beat

    type: spectrum   the sound's frequencies as bars (Audio Spectrum)
      audio          src, as above
      size: [w, h]   the layer's plane (default: the frame); bars stand on its bottom edge and grow to its top
      bands: 48      how many;  range: [40, 10000] Hz;  axis: log | linear
      shape          bars | dots | line | area;  mirror: true — both ways from the middle line
      ring: r        around a circle of this radius in the middle of the plane, growing outwards (animates)
      symmetric      true: the picture is two mirrored halves — round a ring the low notes at the top and the high
                     ones at the bottom on both sides (a whole circle, not a lopsided one); flat — lows in the middle
      gap: 0.35      the share of a bar's step left empty;  round: true — round ends;  width: 3 (line thickness)
      color          a colour, #rrggbbaa or a gradient in the plane's px: {gradient: linear, from, to, stops}
      db: 48         the range shown: a band this many dB under the sound's loudest is at zero
      fall: 0.35     seconds a bar takes to fall its whole height;  floor: 2 (px a silent bar keeps);  gain (animates)

    type: wave       the sound's own curve (Audio Waveform)
      audio, size, color, width, gain, ring — as above
      window: 0.05   seconds of sound shown, centred on now (an oscilloscope); or
      span: [a, b]   a fixed stretch of the film drawn whole (an audiogram): with `played: colour` what is before
                     now takes that colour — a progress bar made of the sound itself
      shape          line | bars (`bars: n` of them, each the loudness of its slice; gap, round, mirror as above)
      norm: true     the sound's loudest reaches the plane's edge (false: full scale does)

A sound is read once: ffmpeg → 24 kHz mono, loudness every 5 ms and 96 bands (20 Hz … 12 kHz, equal steps of pitch)
from a 2048-sample Hann window, kept in ~/.cache/kadr/sound by the file's path, date and size. Every value is a
function of the file and the time — any process, starting anywhere, gets the same frame. Both renderers draw the
layers the same way: Skia in the layer's plane, then placed like a picture."""
import hashlib, json, math, os, subprocess, threading

import numpy as np
from PIL import Image

SR, HOP, NFFT, NB, F0, F1 = 24000, 120, 2048, 96, 20.0, 12000.0
RATE = SR / HOP                                                # 200 rows of analysis a second
VERSION = 1
CACHE = os.path.expanduser('~/.cache/kadr/sound')
EDGES = F0 * (F1 / F0) ** (np.arange(NB + 1) / NB)             # the analysis bands' edges, Hz
LE = np.log(EDGES)
_tracks, _lock = {}, threading.Lock()


def weights(lo, hi):
    """how much of each analysis band lies in lo … hi Hz (a band's power taken as even over its pitch) → (NB,)"""
    a, b = math.log(max(F0, min(F1, lo))), math.log(max(F0, min(F1, hi)))
    return np.clip(np.minimum(b, LE[1:]) - np.maximum(a, LE[:-1]), 0, None) / (LE[1:] - LE[:-1])


class Track:
    """one sound file, analysed: pcm (int16, 24 kHz mono), cum (running sum of each 5 ms row's mean square),
    mx (each row's largest sample), db (rows × 96 bands: power, dB)"""

    def __init__(self, path, st):
        self.path = path
        base = os.path.join(CACHE, hashlib.sha1(f'{path}|{st.st_mtime_ns}|{st.st_size}|{VERSION}'.encode()).hexdigest()[:20])
        if not os.path.exists(base + '.json'):
            self._analyse(path, base)
        meta = json.load(open(base + '.json', encoding='utf-8'))
        self.pcm = np.load(base + '.pcm.npy', mmap_mode='r')
        self.db = np.load(base + '.db.npy', mmap_mode='r')
        self.mx = np.load(base + '.mx.npy')
        self.cum = np.concatenate([[0.0], np.cumsum(np.load(base + '.e.npy'), dtype=np.float64)])
        self.n = len(self.mx)
        self.dur = len(self.pcm) / SR
        self.ref_amp, self.ref_db, self.peak = max(1e-6, meta['ref_amp']), meta['ref_db'], max(1e-6, meta['peak'])
        self._refs = {}

    @staticmethod
    def _analyse(path, base):
        r = subprocess.run(['ffmpeg', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-i', path, '-vn', '-ac', '1', '-ar', str(SR), '-f', 's16le', '-'], capture_output=True)
        pcm = np.frombuffer(r.stdout, np.int16)
        if r.returncode or len(pcm) < HOP:
            raise ValueError(f'в файле «{os.path.basename(path)}» нет звука, который можно прочитать')
        n = -(-len(pcm) // HOP)
        pad = np.zeros(NFFT // 2 + n * HOP + NFFT, np.int16)   # the file with half a window of silence before it
        pad[NFFT // 2:NFFT // 2 + len(pcm)] = pcm
        e, mx, db = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros((n, NB), np.float16)
        win = np.hanning(NFFT + 1)[:-1].astype(np.float32)
        df = SR / NFFT
        k = np.arange(NFFT // 2 + 1)
        lo_, hi_ = (k - .5) * df, (k + .5) * df                # each FFT bin's own stretch of Hz, shared out to the bands
        M = (np.clip(np.minimum(hi_[None], EDGES[1:, None]) - np.maximum(lo_[None], EDGES[:-1, None]), 0, None) / df).astype(np.float32)
        norm = np.where((k > 0) & (k < NFFT // 2), 2.0, 1.0) / (NFFT * float((win ** 2).sum())) / 32768.0 ** 2
        for k0 in range(0, n, 2048):
            m = min(2048, n - k0)
            x = pad[NFFT // 2 + k0 * HOP:NFFT // 2 + (k0 + m) * HOP].astype(np.float32).reshape(m, HOP) / 32768.0
            e[k0:k0 + m], mx[k0:k0 + m] = (x * x).mean(1), np.abs(x).max(1)
            s = k0 * HOP + HOP // 2                             # row k is the window centred on the middle of its 5 ms
            fr = np.lib.stride_tricks.sliding_window_view(pad[s:s + (m - 1) * HOP + NFFT], NFFT)[::HOP]
            P = np.abs(np.fft.rfft(fr.astype(np.float32) * win, axis=1)) ** 2 * norm
            db[k0:k0 + m] = 10 * np.log10(P.astype(np.float32) @ M.T + 1e-12)
        cum = np.concatenate([[0.0], np.cumsum(e, dtype=np.float64)])
        w = int(round(.05 * RATE))
        rms = np.sqrt((cum[w:] - cum[:-w]) / w) if n > w else np.sqrt(cum[-1:] / n)
        meta = dict(ref_amp=float(np.percentile(rms, 99.9)), ref_db=float(np.percentile(db.max(1).astype(np.float32), 99.9)),
                    peak=float(np.percentile(mx, 99.9)), n=n, src=path)
        os.makedirs(CACHE, exist_ok=True)
        for name, arr in (('pcm', pcm), ('e', e), ('mx', mx), ('db', db)):
            tmp = f'{base}.{name}.{os.getpid()}.tmp'
            with open(tmp, 'wb') as f:
                np.save(f, arr)
            os.replace(tmp, f'{base}.{name}.npy')
        tmp = f'{base}.json.{os.getpid()}.tmp'
        json.dump(meta, open(tmp, 'w', encoding='utf-8'))
        os.replace(tmp, base + '.json')                        # last: a cache without it is not read

    def ms(self, a, b):
        """the mean square of the sound between source seconds a and b (arrays), silence outside the file"""
        def at(s):
            r = np.clip(np.asarray(s, float) * RATE, 0, self.n)
            i = np.minimum(r.astype(int), self.n - 1)
            return self.cum[i] + (r - i) * (self.cum[i + 1] - self.cum[i])
        return (at(b) - at(a)) / np.maximum(1e-9, (np.asarray(b, float) - np.asarray(a, float)) * RATE)

    def rows(self, s):
        """the bands' power at source seconds s (array) → (len, NB), silence outside the file"""
        r = np.asarray(s, float) * RATE - .5
        i = np.floor(r).astype(int)
        f = (r - i)[:, None]
        ok = ((r > -1) & (r < self.n))[:, None]
        i0, i1 = np.clip(i, 0, self.n - 1), np.clip(i + 1, 0, self.n - 1)
        lo, hi = int(min(i0.min(), i1.min())), int(max(i0.max(), i1.max())) + 1
        blk = 10.0 ** (np.asarray(self.db[lo:hi], np.float32) / 10)   # one read of the stretch of rows asked for
        return np.where(ok, blk[i0 - lo] * (1 - f) + blk[i1 - lo] * f, 0.0)

    def band_ref(self, lo, hi):
        """the loudest the band lo … hi gets in this file (99.9 % of the time it is under it) — and never more than
        40 dB under the file's own loudest, so a band the sound does not have stays at zero, not at its noise"""
        key = (round(lo, 3), round(hi, 3))
        if key not in self._refs:
            w = weights(lo, hi).astype(np.float32)
            j = np.nonzero(w)[0]
            if not len(j):
                self._refs[key] = self.ref_amp
            else:
                top = 0.0
                parts = [np.sqrt(10.0 ** (np.asarray(self.db[k0:k0 + 20000, j[0]:j[-1] + 1], np.float32) / 10) @ w[j[0]:j[-1] + 1]) for k0 in range(0, self.n, 20000)]
                top = float(np.percentile(np.concatenate(parts), 99.9))
                self._refs[key] = max(top, self.ref_amp * .01)
        return self._refs[key]


LEAD = 0.030                                                   # an onset shows in the analysis this much before it sounds (its 85 ms window
                                                               # looks ahead); measured on click tracks: −30.0 ms, spread 0.7 ms


def _onsets(T):
    """how strongly something new begins at each 5 ms row: how much the bands grew louder since the row before, added over
    the bands (only growth counts, and only above 60 dB under the sound's loudest), less its own running average"""
    out = np.zeros(T.n, np.float32)
    prev = None
    for k0 in range(0, T.n, 20000):
        d = np.maximum(np.asarray(T.db[k0:k0 + 20000], np.float32), T.ref_db - 60)
        e = np.concatenate([prev, d]) if prev is not None else np.concatenate([np.full_like(d[:1], T.ref_db - 60), d])   # before the file: silence
        out[k0:k0 + len(d)] = np.maximum(np.diff(e, axis=0), 0).sum(1)
        prev = d[-1:]
    w = int(.5 * RATE)
    c = np.concatenate([[0.0], np.cumsum(out, dtype=np.float64)])
    i = np.arange(T.n)
    mean = (c[np.minimum(T.n, i + w)] - c[np.maximum(0, i - w)]) / (np.minimum(T.n, i + w) - np.maximum(0, i - w))
    out = np.maximum(out - mean, 0)
    g = np.exp(-.5 * (np.arange(-4, 5) / 1.5) ** 2)             # a hit is felt over a few rows
    out = np.convolve(out, g / g.sum(), 'same')
    return out / (out.std() + 1e-9)


def find_beats(T):
    """the music's pulse → (beats a minute, the seconds of the file each beat falls on). The tempo: the lag at which the
    onsets best repeat themselves (60 … 200 a minute, leaning to 120); the beats: the row of moments, about that far
    apart, that gathers the most onsets (Ellis, 2007)"""
    if getattr(T, '_beats', None) is not None:
        return T._beats
    env = _onsets(T).astype(np.float64)
    n = len(env)
    if n < 3 * RATE or env.max() <= 0:
        T._beats = (0.0, np.zeros(0))
        return T._beats
    lags = np.arange(int(RATE * 60 / 200), int(RATE * 60 / 50) + 1)
    m = min(n, int(90 * RATE))                                  # the first minute and a half is enough to hear the tempo
    seg = env[:m] - env[:m].mean()
    ac = np.array([float(seg[:m - l] @ seg[l:]) / (m - l) for l in lags])
    bpm = 60 * RATE / lags
    ac = ac * np.exp(-.5 * (np.log2(bpm / 120) / .9) ** 2)
    k = int(np.argmax(ac))
    if 0 < k < len(lags) - 1:                                   # between two lags: the top of the parabola through three
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        period = lags[k] + (.5 * (a - c) / (a - 2 * b + c) if abs(a - 2 * b + c) > 1e-12 else 0.0)
    else:
        period = float(lags[k])
    p = int(round(period))
    back = np.arange(-2 * p, -p // 2)                           # how far back the last beat may be
    cost = -100.0 * np.log(-back / period) ** 2
    score, prev = env.copy(), np.full(n, -1)
    for t in range(n):
        lo = t + back[0]
        if lo < 0:
            w = back[-lo:] if -lo < len(back) else back[:0]
            c = cost[-lo:] if -lo < len(back) else cost[:0]
        else:
            w, c = back, cost
        if len(w):
            v = score[t + w] + c
            j = int(np.argmax(v))
            if v[j] > 0:
                score[t] += v[j]
                prev[t] = t + w[j]
    t = int(np.argmax(score[max(0, n - p):])) + max(0, n - p)
    path = []
    while t >= 0:
        path.append(t)
        t = prev[t]
    rows = np.array(path[::-1], float)
    # each beat set on the top of the onset beside it (to a fraction of a row), then the row of beats evened out a little
    fine = []
    for r in rows:
        r = int(r)
        a, b = max(0, r - 3), min(n, r + 4)
        j = a + int(np.argmax(env[a:b]))
        if 0 < j < n - 1 and env[j] > 0:
            y0, y1, y2 = env[j - 1], env[j], env[j + 1]
            j = j + (.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if abs(y0 - 2 * y1 + y2) > 1e-12 else 0.0)
        fine.append(j)
    # the row of beats may run a step past the music at either end: a beat with nothing sounding on it there is dropped
    strength = np.array([env[max(0, int(r) - 3):min(n, int(r) + 4)].max() for r in rows])
    keep = np.ones(len(rows), bool)
    floor = .2 * np.median(strength)
    for side in (range(len(rows)), range(len(rows) - 1, -1, -1)):
        for i in side:
            if strength[i] >= floor:
                break
            keep[i] = False
    fine = [f for f, k in zip(fine, keep) if k]
    times = (np.array(fine) + .5) / RATE + LEAD
    tempo = 60.0 / float(np.polyfit(np.arange(len(times)), times, 1)[0]) if len(times) > 2 else 60 * RATE / period   # the row's own step
    T._beats = (tempo, times)
    return T._beats


def beats_of(F, name=None):
    """the beats of the film's music on the film's own clock → (tempo, seconds)"""
    out, tempo = [], 0.0
    root = F
    while root._parent_film is not None:
        root = root._parent_film
    for q in sources(root, name):
        T = track(q['path'])
        tempo, bt = find_beats(T)
        end = q['end'] if q['end'] is not None else T.dur
        bt = bt[(bt >= q['start'] - 1e-6) & (bt <= end)]
        out += list(q['at'] + (bt - q['start']) / q['speed'])
    return tempo, np.array(sorted(out))


def beat_pos(F, t, name=None):
    """where in the row of beats the moment t is: 5 on the fifth beat, 5.5 halfway to the sixth (before the first — under 1)"""
    cache = F.__dict__.setdefault('_beat_rows', {})
    if name not in cache:
        cache[name] = beats_of(F, name)[1]
    b = cache[name]
    if len(b) < 2:
        return 0.0
    step = float(np.median(np.diff(b)))
    if t <= b[0]:
        return 1 + (t - b[0]) / step
    if t >= b[-1]:
        return len(b) + (t - b[-1]) / step
    return float(np.interp(t, b, np.arange(len(b)) + 1.0))


def track(path):
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    with _lock:
        if key not in _tracks:
            _tracks[key] = Track(path, st)
        return _tracks[key]


# ---------------- which sound ----------------
def sources(F, name):
    """the sound a name stands for → [{path, at, start, end, speed, up}]: where each file stands, and on whose
    timeline — `up` films above this one (0: this comp's own audio or video layer, 1: its parent's, …). A comp
    looks in its own film.audio first, then in the films it is nested in"""
    chain, f = [], F
    while f is not None:
        chain.append(f)
        f = f._parent_film
    one = lambda a, up: dict(path=a['src'], at=a['at'], start=a['start'], end=a['end'], speed=1.0, up=up)
    if name is None:
        for up, f in enumerate(chain):
            if f.audio:
                return [one(a, up) for a in ([a for a in f.audio if a['role'] == 'music'] or f.audio)]
        raise ValueError('в film.audio нет звука: назовите источник — id видео-слоя или файл')
    name = str(name)
    for up, f in enumerate(chain):
        got = [a for a in f.audio if a.get('id') == name] or [a for a in f.audio if a['role'] == name]
        if got:
            return [one(a, up) for a in got]
        L = f._by_id.get(name)
        if L is not None and L['type'] == 'video':
            return [dict(path=L['src'], at=L['in'], start=L.get('from', 0.0), end=None, speed=L.get('speed', 1.0), up=up)]
    p = F.resolve(name)
    if os.path.isfile(p):
        return [dict(path=p, at=0.0, start=0.0, end=None, speed=1.0, up=len(chain) - 1)]
    import difflib
    known = sorted({a['id'] for f in chain for a in f.audio if a.get('id')} | {a['role'] for f in chain for a in f.audio} | {L['id'] for L in F.layers if L['type'] == 'video'})
    near = difflib.get_close_matches(name, known, n=1)
    raise ValueError(f'звук «{name}»: нет ни дорожки film.audio с таким id или ролью, ни видео-слоя, ни файла' +
                     (f' (может быть, {near[0]}?)' if near else '') + (f'; есть: {", ".join(known)}' if known else ''))


def _heard(F, name, ts):
    """the sources of `name` with the source seconds they are at, at this film's (this comp's) times ts →
    [(Track, source seconds, which of ts fall inside it)]"""
    cache = F.__dict__.setdefault('_sound_srcs', {})
    if name not in cache:
        cache[name] = sources(F, name)
    times, f = [np.asarray(ts, float)], F                      # ts on this film's timeline, its parent's, …
    out = []
    for q in cache[name]:
        while len(times) <= q['up']:                           # a comp's second → the second of the film it is in
            m = getattr(f, '_to_parent', None)
            if m is None:
                raise ValueError('звук во вставке, у которой время задано ключами (time): какому моменту фильма отвечает её секунда, неизвестно')
            times.append(m(times[-1]))
            f = f._parent_film
        T = track(q['path'])
        tt = times[q['up']]
        s = q['start'] + (tt - q['at']) * q['speed']
        ok = (tt >= q['at']) & (s <= (q['end'] if q['end'] is not None else T.dur))
        if ok.any():
            out.append((T, s, ok))
    return out


def _fall(v, fall):
    """the value now, of levels v[j] heard j rows ago: it rises at once and falls away in `fall` seconds"""
    return float((v * np.exp(-np.arange(len(v)) / (RATE * fall))).max()) if len(v) > 1 else float(v[0])


def amp(F, t, src=None, fall=.12, window=.05, norm=True):
    fall, window = min(5.0, max(0.0, float(fall))), max(.005, float(window))
    ts = float(t) - np.arange(int(math.ceil(7 * fall * RATE)) + 1) / RATE
    p, ref = np.zeros(len(ts)), 1e-6
    for T, s, ok in _heard(F, src, ts):
        p += np.where(ok, T.ms(s - window / 2, s + window / 2), 0.0)
        ref = max(ref, T.ref_amp)
    v = _fall(np.sqrt(np.maximum(p, 0)), fall)
    return min(4.0, v / ref) if norm else v


def band(F, t, lo, hi, src=None, fall=.12, norm=True):
    lo, hi = float(lo), float(hi)
    if not lo < hi:
        raise ValueError(f'band({lo:g}, {hi:g}): сначала нижняя частота, потом верхняя, Гц')
    fall = min(5.0, max(0.0, float(fall)))
    ts = float(t) - np.arange(int(math.ceil(7 * fall * RATE)) + 1) / RATE
    w = weights(lo, hi)
    p, ref = np.zeros(len(ts)), 1e-6
    for T, s, ok in _heard(F, src, ts):
        p += np.where(ok, T.rows(s) @ w, 0.0)
        ref = max(ref, T.band_ref(lo, hi))
    v = _fall(np.sqrt(p), fall)
    return min(4.0, v / ref) if norm else v


def check_formula(F, src, where):
    """a formula that names a sound: the name must stand for something (said when the document is read)"""
    import ast
    try:
        tree = ast.parse(src.lstrip('=').strip(), mode='eval')
    except SyntaxError:
        return
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('amp', 'band'):
            k = 0 if n.func.id == 'amp' else 2
            a = n.args[k] if len(n.args) > k else next((kw.value for kw in n.keywords if kw.arg == 'src'), None)
            for kw in n.keywords:
                if kw.arg not in ('src', 'fall', 'norm', 't') + (('window',) if n.func.id == 'amp' else ()):
                    F.issue('error', f'{n.func.id}(): нет параметра «{kw.arg}»', where, kw.arg, {'src', 'fall', 'window', 'norm', 't'})
            if a is None or (isinstance(a, ast.Constant) and (a.value is None or isinstance(a.value, str))):
                try:
                    sources(F, a.value if a is not None else None)
                except ValueError as e:
                    F.issue('error', str(e), where)


# ---------------- the layers ----------------
def _plane(F, L, lid, out):
    s = L.get('size', [F.W, F.H])
    if not (isinstance(s, (list, tuple)) and len(s) == 2):
        F.issue('error', 'size: нужно [w, h] — плоскость слоя', lid)
        s = [F.W, F.H]
    out['psize'] = (float(s[0]), float(s[1]))
    out['cell'] = 16.0
    out['audio'] = L.get('audio')
    try:
        sources(F, out['audio'])
    except ValueError as e:
        F.issue('error', str(e), f'{lid}.audio')
    out['color'] = F.paint(L.get('color', '#ffffff'), f'{lid}.color')
    out['width'] = float(L.get('width', 3))
    out['gain'] = F.prop(L.get('gain', 1), f'{lid}.gain')
    out['ring'] = F.prop(L.get('ring', 0), f'{lid}.ring')
    out['gap'] = min(.95, max(0.0, float(L.get('gap', .35))))
    out['round'] = bool(L.get('round', True))


def _choice(F, L, lid, key, default, allowed):
    v = L.get(key, default)
    if v not in allowed:
        F.issue('error', f'{key} «{v}»: нужно ' + ' | '.join(allowed), lid, v, set(allowed))
        v = default
    return v


def parse_spectrum(F, L, lid, out):
    _plane(F, L, lid, out)
    out['bands'] = int(L.get('bands', 48))
    if not 2 <= out['bands'] <= 512:
        F.issue('error', 'bands: от 2 до 512 полос', lid)
        out['bands'] = 48
    r = L.get('range', [40, 10000])
    if not (isinstance(r, (list, tuple)) and len(r) == 2 and F0 <= float(r[0]) < float(r[1]) <= F1):
        F.issue('error', f'range: [от, до] в герцах, в пределах {F0:g} … {F1:g}', lid)
        r = [40, 10000]
    out['range'] = (float(r[0]), float(r[1]))
    out['axis'] = _choice(F, L, lid, 'axis', 'log', ('log', 'linear'))
    out['shape'] = _choice(F, L, lid, 'shape', 'bars', ('bars', 'dots', 'line', 'area'))
    out['mirror'] = bool(L.get('mirror', False))
    out['symmetric'] = bool(L.get('symmetric', False))
    out['db'] = max(6.0, float(L.get('db', 48)))
    out['fall'] = min(5.0, max(0.0, float(L.get('fall', .35))))
    out['floor'] = float(L.get('floor', 2))
    lo, hi = out['range']
    n = out['bands']
    ed = lo * (hi / lo) ** (np.arange(n + 1) / n) if out['axis'] == 'log' else np.linspace(lo, hi, n + 1)
    out['wm'] = np.stack([weights(a, b) for a, b in zip(ed[:-1], ed[1:])])   # (bars, NB)
    out['wn'] = np.maximum(out['wm'].sum(1), 1e-6)             # how many analysis bands a bar is wide


def parse_wave(F, L, lid, out):
    _plane(F, L, lid, out)
    out['shape'] = _choice(F, L, lid, 'shape', 'line', ('line', 'bars'))
    out['window'] = max(.002, float(L.get('window', .05)))
    out['span'] = None
    if L.get('span') is not None:
        sp = L['span']
        if not (isinstance(sp, (list, tuple)) and len(sp) == 2):
            F.issue('error', 'span: [с какой секунды фильма, по какую]', lid)
        else:
            a, b = F.time(sp[0], f'{lid}.span'), F.time(sp[1], f'{lid}.span')
            if b <= a:
                F.issue('error', 'span: конец раньше начала', lid)
            else:
                out['span'] = (a, b)
    out['played'] = F.paint(L['played'], f'{lid}.played') if L.get('played') is not None else None
    if out['played'] is not None and out['span'] is None:
        F.issue('error', 'played красит то, что уже прозвучало, — ему нужен span: [от, до]', lid)
    out['bars'] = int(L.get('bars', max(8, round(out['psize'][0] / 10))))
    if not 2 <= out['bars'] <= 2000:
        F.issue('error', 'bars: от 2 до 2000', lid)
        out['bars'] = 64
    out['mirror'] = bool(L.get('mirror', True))
    out['norm'] = bool(L.get('norm', True))


def spectrum_heights(F, L, t):
    """each bar's height 0 … 1 at t: its power for its width, in dB under the sound's loudest, falling as `fall` says"""
    J = int(math.ceil(L['fall'] * RATE))
    ts = t - np.arange(J + 1) / RATE
    P, ref = np.zeros((J + 1, NB)), None
    for T, s, ok in _heard(F, L['audio'], ts):
        P += np.where(ok[:, None], T.rows(s), 0.0)
        ref = T.ref_db if ref is None else max(ref, T.ref_db)
    if ref is None:
        return np.zeros(L['bands'] * (2 if L['symmetric'] else 1))
    d = 10 * np.log10((P @ L['wm'].T) / L['wn'] + 1e-12)       # (J + 1, bars)
    hgt = np.clip((d - ref) / L['db'] + 1, 0, 1)
    if J:
        hgt = hgt - (np.arange(J + 1) / (RATE * L['fall']))[:, None]
    v = np.clip(hgt.max(0) * L['gain'].at(t), 0, 1)
    if L['symmetric']:                                         # two mirrored halves: lows at the top of a ring, in the middle of a row
        v = np.concatenate([v, v[::-1]]) if L['ring'].animated or L['ring'].value else np.concatenate([v[::-1], v])
    return v


def wave_data(F, L, t):
    """what the wave layer shows at t → ('line', lo, hi: (n,) −1 … 1 — one curve when lo is hi) or
    ('bars', v: (n,) 0 … 1, played: (n,) bool)"""
    a, b = L['span'] if L['span'] else (t - L['window'] / 2, t + L['window'] / 2)
    gain = L['gain'].at(t)
    if L['shape'] == 'bars':
        n = L['bars']
        ed = a + (b - a) * np.arange(n + 1) / n
        p, ref = np.zeros(n), 1e-6
        for T, s, ok in _heard(F, L['audio'], (ed[:-1] + ed[1:]) / 2):
            half = (s[1] - s[0]) / 2                           # half a bar's slice, in the source's own seconds
            p += np.where(ok, T.ms(s - half, s + half), 0.0)
            ref = max(ref, T.ref_amp)
        v = np.sqrt(np.maximum(p, 0)) / (ref if L['norm'] else 1.0) * gain
        return 'bars', np.clip(v, 0, 1), (ed[:-1] + ed[1:]) / 2 < t
    w = L['psize'][0]
    if b - a <= 2.0:                                           # the samples themselves, mixed on one grid
        n = max(2, int(round((b - a) * SR)))
        x, ref = np.zeros(n, np.float32), 1e-6
        for T, s, ok in _heard(F, L['audio'], a + np.arange(n) / SR):
            i = np.round(s * SR).astype(np.int64)
            ok = ok & (i >= 0) & (i < len(T.pcm))
            if ok.any():
                lo_ = int(i[ok].min())
                x[ok] += np.asarray(T.pcm[lo_:int(i[ok].max()) + 1], np.float32)[i[ok] - lo_] / 32768.0
            ref = max(ref, T.peak)
        x = x / (ref if L['norm'] else 1.0) * gain
        cols = int(min(n, max(2, round(w))))
        if n <= cols * 1.5:
            return 'line', np.clip(x, -1, 1), None
        cut = (np.arange(cols + 1) * n // cols)[:-1]
        return 'line', np.clip(np.minimum.reduceat(x, cut), -1, 1), np.clip(np.maximum.reduceat(x, cut), -1, 1)
    cols = int(max(2, round(w)))                               # a long stretch: each column's largest sample, both ways
    ed = a + (b - a) * np.arange(cols + 1) / cols
    top, ref = np.zeros(cols), 1e-6
    for T, s, ok in _heard(F, L['audio'], ed):
        r = np.clip(s * RATE, 0, T.n - 1).astype(int)
        top += np.where(ok[:-1], [T.mx[i:max(i + 1, j)].max() for i, j in zip(r[:-1], r[1:])], 0.0)
        ref = max(ref, T.peak)
    top = np.clip(top / (ref if L['norm'] else 1.0) * gain, 0, 1)
    return 'line', -top, top


def _curve(pts, closed):
    """a smooth path through the points (Catmull–Rom as cubics)"""
    import skia
    p = skia.Path()
    n = len(pts)
    if n < 2:
        return p
    g = (lambda i: pts[i % n]) if closed else (lambda i: pts[min(n - 1, max(0, i))])
    p.moveTo(*pts[0])
    for i in range(n if closed else n - 1):
        p0, p1, p2, p3 = g(i - 1), g(i), g(i + 1), g(i + 2)
        p.cubicTo(p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6, p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6, p2[0], p2[1])
    if closed:
        p.close()
    return p


def _poly(pts, closed):
    import skia
    p = skia.Path()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    if closed:
        p.close()
    return p


def render(F, L, t, sc):
    """the layer at t drawn in its own plane at `sc` px per content px → (x0, y0, w, h of the canvas in content px,
    premultiplied PIL RGBa picture of it) or None when there is nothing to draw"""
    import skia
    from .render import paint_of
    w, h = L['psize']
    W, H = int(math.ceil(w * sc)), int(math.ceil(h * sc))
    if W <= 0 or H <= 0 or W * H > 40e6:
        return None
    surf = skia.Surface.MakeRaster(skia.ImageInfo.Make(W, H, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType))
    c = surf.getCanvas()
    c.clear(skia.Color4f(0, 0, 0, 0))
    c.scale(sc, sc)
    box = skia.Rect.MakeWH(w, h)
    fill = paint_of(L['color'], box)
    line = paint_of(L['color'], box)
    line.setStyle(skia.Paint.kStroke_Style)
    line.setStrokeWidth(L['width'])
    line.setStrokeCap(skia.Paint.kRound_Cap)
    line.setStrokeJoin(skia.Paint.kRound_Join)
    ring = max(0.0, float(L['ring'].at(t)))
    cx, cy = w / 2, h / 2
    reach = (min(w, h) / 2 - ring) if ring else h              # how far a full bar goes

    def bars(v, paints, mirror, floor, as_dots=False):
        n = len(v)
        step = (2 * math.pi * ring / n) if ring else w / n
        bw = step * (1 - L['gap'])
        rr = bw / 2 if L['round'] else 0.0
        for i, x in enumerate(v):
            ln = max(floor, x * reach)
            p = paints[i] if isinstance(paints, list) else paints
            if ring:
                c.save()
                c.rotate(-90 + 360 * (i + .5) / n, cx, cy)
                a0 = ring - (ln / 2 if mirror else 0)
                r = skia.Rect.MakeLTRB(cx + a0, cy - bw / 2, cx + a0 + ln, cy + bw / 2)
                if as_dots:
                    c.drawCircle(cx + ring + x * reach, cy, bw / 2, p)
                else:
                    c.drawRRect(skia.RRect.MakeRectXY(r, min(rr, ln / 2), min(rr, ln / 2)), p)
                c.restore()
                continue
            x0 = i * step + (step - bw) / 2
            if as_dots:
                c.drawCircle(x0 + bw / 2, (h / 2 - x * h / 2) if mirror else (h - bw / 2 - x * (h - bw)), bw / 2, p)
                if mirror:
                    c.drawCircle(x0 + bw / 2, h / 2 + x * h / 2, bw / 2, p)
                continue
            r = skia.Rect.MakeLTRB(x0, h / 2 - ln / 2, x0 + bw, h / 2 + ln / 2) if mirror else skia.Rect.MakeLTRB(x0, h - ln, x0 + bw, h)
            c.drawRRect(skia.RRect.MakeRectXY(r, min(rr, ln / 2), min(rr, ln / 2)), p)

    def place(u, v):
        """a point of a curve: u 0 … 1 along it, v its height (−1 … 1 from the middle line, or 0 … 1 from the bottom)"""
        if ring:
            a = math.radians(-90 + 360 * u)
            return cx + (ring + v * reach) * math.cos(a), cy + (ring + v * reach) * math.sin(a)
        return u * w, v

    if L['type'] == 'spectrum':
        v = spectrum_heights(F, L, t)
        n = len(v)
        if L['shape'] in ('bars', 'dots'):
            bars(v, fill, L['mirror'], L['floor'], L['shape'] == 'dots')
        else:
            us = (np.arange(n) + .5) / n
            if ring:
                up = [place(u, x) for u, x in zip(us, v)]
                dn = [place(u, -x if L['mirror'] else 0.0) for u, x in zip(us, v)]
                if L['shape'] == 'area':
                    p = _curve(up, True)
                    p.addPath(_curve(dn, True))
                    p.setFillType(skia.PathFillType.kEvenOdd)
                    c.drawPath(p, fill)
                else:
                    c.drawPath(_curve(up, True), line)
                    if L['mirror']:
                        c.drawPath(_curve(dn, True), line)
            else:
                base = h / 2 if L['mirror'] else h
                k = h / 2 if L['mirror'] else h
                up = [(0.0, base - v[0] * k)] + [(u * w, base - x * k) for u, x in zip(us, v)] + [(w, base - v[-1] * k)]
                dn = [(x_, 2 * base - y_) for x_, y_ in up]
                if L['shape'] == 'area':
                    p = _curve(up, False)
                    if L['mirror']:
                        q = _curve(dn[::-1], False)
                        p.lineTo(*dn[-1])
                        p.addPath(q, skia.Path.kExtend_AddPathMode)
                    else:
                        p.lineTo(w, h)
                        p.lineTo(0, h)
                    p.close()
                    c.drawPath(p, fill)
                else:
                    c.drawPath(_curve(up, False), line)
                    if L['mirror']:
                        c.drawPath(_curve(dn, False), line)
    else:
        kind, a, b = wave_data(F, L, t)
        if kind == 'bars':
            n = len(a)
            step = (2 * math.pi * ring / n) if ring else w / n
            bw = step * (1 - L['gap'])
            if L['played'] is not None:
                done = paint_of(L['played'], box)
                bars(a, [done if b[i] else fill for i in range(n)], L['mirror'], bw if L['round'] else 1.0)
            else:
                bars(a, fill, L['mirror'], bw if L['round'] else 1.0)
        else:
            n = len(a)
            us = np.arange(n) / (n - 1 if not ring else n)
            k = reach if ring else h / 2
            pt = (lambda u, x: place(u, x * .5)) if ring else (lambda u, x: (u * w, h / 2 - x * k))
            if b is None:
                c.drawPath(_poly([pt(u, x) for u, x in zip(us, a)], bool(ring)), line)
            else:                                              # many samples a pixel: the band between each column's lowest and highest
                p = _poly([pt(u, x) for u, x in zip(us, b)] + [pt(u, x) for u, x in zip(us[::-1], a[::-1])], True)
                c.drawPath(p, fill)
                c.drawPath(p, line)
    a = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
    if not a[..., 3].any():
        return None
    return 0.0, 0.0, W / sc, H / sc, Image.fromarray(np.ascontiguousarray(a), 'RGBa')
