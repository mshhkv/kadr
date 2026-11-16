"""The sound mix: voice, music ducked under the voice, effects and footage sound, then one gain to the film loudness.
   Levels in LUFS (ffmpeg ebur128); ducking follows the voice-over words when the film has them, else the voice tracks' own energy."""
import functools, re, subprocess
import numpy as np
from .media import probe

SR = 48000
RAMP = 0.35            # s, duck attack/release
GAP = 0.3              # s, pauses shorter than this keep the music down
LIMITER = 'alimiter=limit=0.891:level=0:attack=5:release=60'   # −1 dBFS, no auto-level: the loudness gain stays


def decode(path, start=0.0, dur=None, tempo=1.0, af=None):
    """stereo float32 at SR; tempo ≠ 1 plays a sped-up / slowed footage sound in step with its picture (pitch kept)"""
    from . import safe
    af = safe.audio_filter(af) if af else af                    # only named filters with plain values (kadr/safe.py)
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{start:.6f}', *safe.LOCAL, '-i', path]
    if dur is not None:
        cmd += ['-t', f'{dur:.6f}']
    if abs(tempo - 1) > 1e-4:
        steps, x = [], tempo
        while x < 0.5 or x > 2:                                # atempo takes 0.5…2 per stage
            steps.append(0.5 if x < 0.5 else 2.0)
            x /= steps[-1]
        af = ','.join([f'atempo={v:.5f}' for v in steps + [x]] + ([af] if af else []))
    if af:
        cmd += ['-af', af]
    cmd += ['-vn', '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-']
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32).reshape(-1, 2).copy()


def lufs_of(a):
    """integrated loudness of a float32 stereo array (fed to ffmpeg in pieces: a whole film is ~1 GB, no copy of it)"""
    p = subprocess.Popen(['ffmpeg', '-v', 'info', '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-i', '-', '-af', 'ebur128', '-f', 'null', '-'],
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    import threading
    err = []
    th = threading.Thread(target=lambda: err.append(p.stderr.read()))
    th.start()
    a = np.ascontiguousarray(a, np.float32)
    for i in range(0, len(a), SR * 10):
        p.stdin.write(a[i:i + SR * 10].tobytes())
    p.stdin.close()
    p.wait()
    th.join()
    v = re.findall(rb'I:\s+(-?[\d.]+) LUFS', err[0] if err else b'')
    return float(v[-1]) if v else None


@functools.lru_cache(None)
def file_lufs(path):
    return lufs_of(decode(path))


def fade(a, fin, fout):
    n = len(a)
    for secs, rev in ((fin, False), (fout, True)):
        k = min(n, int(secs * SR))
        if k > 0:
            r = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k, dtype=np.float32)))[:, None]
            if rev:
                a[n - k:] *= r[::-1]
            else:
                a[:k] *= r
    return a


ENV_RATE = 1000         # the duck envelope is shaped at 1 ms steps, then spread to samples (a sample-rate convolution of a
                        # whole film is 10¹² multiplications)


def smooth_mask(on, n):
    """0/1 per millisecond → raised-cosine ramps of RAMP s, per sample"""
    k = int(RAMP * ENV_RATE)
    w = np.hanning(2 * k + 1).astype(np.float32)
    w /= w.sum()
    e = np.clip(np.convolve(on, w, mode='same'), 0, 1)
    return np.interp(np.arange(n, dtype=np.float64) * (ENV_RATE / SR), np.arange(len(e)), e).astype(np.float32)


def voice_activity(film, voice, n):
    m = int(np.ceil(n * ENV_RATE / SR)) + 1
    on = np.zeros(m, np.float32)
    if film.words:
        spans = []
        for _, s, e, _ in film.words:
            if spans and s - spans[-1][1] < GAP:
                spans[-1][1] = e
            else:
                spans.append([s, e])
        for s, e in spans:
            on[int(s * ENV_RATE):int(e * ENV_RATE)] = 1
    else:                                                     # no words: 20-ms windows of the voice tracks above −45 dBFS
        win, ms = int(0.02 * SR), SR // ENV_RATE
        for i in range(0, n - win, win):
            if np.abs(voice[i:i + win]).max() > 10 ** (-45 / 20):
                on[max(0, (i - int(GAP * SR)) // ms):(i + win + int(GAP * SR)) // ms] = 1
    return smooth_mask(on, n)


def tracks(film):
    """film.audio + footage sound of video layers with a volume"""
    out = [a for a in film.audio if not a.get('mute')]        # mute: only for amp() / band() / spectrum to hear (a voice already in the mix)
    for L in film.layers:
        if L['type'] == 'video' and (L.get('volume') is not None or L.get('loudness') is not None) and probe(L['src'])['audio'] \
                and L.get('remap') is None:                    # a time-remapped clip is silent: its sound cannot follow the curve
            end = L['from'] + (L['out'] - L['in']) * L['speed']
            if L.get('until') is not None:
                end = min(end, L['until'])
            if L.get('sound_to') is not None:
                end = min(end, L['sound_to'])
            out.append(dict(src=L['src'], role='fx', at=L['in'], start=L['from'], end=end, tempo=L['speed'],
                            volume=L.get('volume') or 0.0, loudness=L.get('loudness'), max_gain=L.get('max_gain', 99), duck=0.0,
                            fade_in=min(0.05, L['fade_in'] or 0.05), fade_out=min(0.12, L['fade_out'] or 0.05), where=L['id'],
                            audio_filter=L.get('audio_filter')))
    return out


def mix(film, t0=0.0, t1=None):
    """float32 stereo mix of [t0, t1) at SR (peaks are limited to −1 dBFS when muxed, see LIMITER)"""
    t1 = film.duration if t1 is None else t1
    n = int(round(film.duration * SR))
    buses = {r: np.zeros((n, 2), np.float32) for r in ('voice', 'music', 'fx')}
    env = None
    for a in sorted(tracks(film), key=lambda a: a['role'] == 'music'):     # voices first: music ducks under the finished voice bus
        dur = None if a['end'] is None else a['end'] - a['start']
        x = decode(a['src'], a['start'], dur, a.get('tempo', 1.0), a.get('audio_filter'))
        g = a['volume']
        if a['loudness'] is not None:
            lu = file_lufs(a['src'])
            if lu is not None and lu > -70:
                g += min(a['loudness'] - lu, a.get('max_gain', 99))
        x = fade(x * 10 ** (g / 20), a['fade_in'], a['fade_out'])
        i = int(round(a['at'] * SR))
        if i < 0:                                              # started before the film: play only what falls inside
            x, i = x[-i:], 0
        if i >= n or not len(x):
            continue
        x = x[:n - i]
        if a['role'] == 'music' and a['duck']:
            if env is None:
                env = voice_activity(film, buses['voice'], n)
            x *= (1 - env[i:i + len(x)] * (1 - 10 ** (a['duck'] / 20)))[:, None]
        buses[a['role']][i:i + len(x)] += x
    out = buses.pop('voice')                                  # summed in place: each bus of a whole film is ~0.8 GB
    out += buses.pop('music')
    out += buses.pop('fx')
    if film.loudness is not None and out.any():
        lu = lufs_of(out)
        if lu is not None and lu > -70:
            out *= 10 ** ((film.loudness - lu) / 20)
    return out[int(t0 * SR):int(t1 * SR)]
