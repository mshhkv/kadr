"""The sound of the AE 48 films: eight seconds of a made-up beat (kick, snare, hats, a bass line, an arpeggio), written
by this script so that it is the same file everywhere. python tests/ae/make_sound.py → tests/ae/beat.wav"""
import os, wave
import numpy as np
SR, BPM, SEC = 24000, 120, 8
rng = np.random.default_rng(7)
n = SR * SEC
x = np.zeros(n)
beat = 60 / BPM


def put(at, y, gain, rise=.003, tail=.02):
    """a sound laid at its second: it starts and ends at zero (a few ms in, 20 ms out) — a sound cut off mid-swing is
    heard as a click, and the first version of this file had one at the end of every note"""
    y = np.array(y, float)
    a, b = int(rise * SR), int(tail * SR)
    y[:a] *= .5 - .5 * np.cos(np.pi * np.arange(a) / a)
    y[-b:] *= .5 + .5 * np.cos(np.pi * np.arange(b) / b)
    i = int(round(at * SR))
    y = y[:max(0, n - i)]
    x[i:i + len(y)] += gain * y


def env(sec, tau):
    t = np.arange(int(sec * SR)) / SR
    return t, np.exp(-t / tau)


def hiss(m, lo):
    """noise without its low part (a moving average taken away), its largest sample at 1"""
    v = rng.standard_normal(m + lo)
    v = (v - np.convolve(v, np.ones(lo) / lo, 'same'))[lo // 2:lo // 2 + m]
    return v / np.abs(v).max()


for b in range(int(SEC / beat)):
    t, e = env(.45, .11)                                       # kick: a sine falling from 110 to 45 Hz
    put(b * beat, np.sin(2 * np.pi * (45 * t + (110 - 45) * .05 * (1 - np.exp(-t / .05)))) * e, .8, rise=.002)
    t, e = env(.22, .045)                                      # hat on the off-beat: a soft hiss, not a tick
    put(b * beat + beat / 2, hiss(len(t), 6) * e, .1, rise=.004)
    if b % 2 == 1:                                             # snare on two and four
        t, e = env(.3, .07)
        put(b * beat, (hiss(len(t), 40) * .55 + np.sin(2 * np.pi * 185 * t) * .5) * e, .3)
for k, hz in enumerate([55, 55, 65.4, 49] * 2):                # two bass notes a beat apart, a bar each
    for q in range(2):
        t, e = env(.45, .16)
        put((k * 2 + q) * beat + .02, sum(np.sin(2 * np.pi * hz * h * t) / h for h in (1, 2, 3)) * e, .2, rise=.008)
scale = [440, 523.25, 587.33, 659.25, 783.99, 880]
for i in range(int(SEC / (beat / 4))):                         # sixteenths from the second bar on
    if i * beat / 4 < 2:
        continue
    hz = scale[(i * 5 + i // 7) % len(scale)]
    t, e = env(.3, .07)
    put(i * beat / 4, (np.sin(2 * np.pi * hz * t) + .4 * np.sin(2 * np.pi * 2 * hz * t)) * e, .08, rise=.005)
x = x / np.abs(x).max() * .89                                  # no clipping of any kind: the loudest sample is at 0.89
with wave.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beat.wav'), 'wb') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
    f.writeframes((x * 32767).astype('<i2').tobytes())
