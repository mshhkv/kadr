"""A matte program in its smallest working form: the language Kadr speaks to the thing that cuts people and objects out
of pictures (kadr matte, `cutout`, kadr reframe, kadr puppet --pose). On macOS that thing is kadr/matte.swift (Apple
Vision). On Windows and Linux it has to be written once — copy this file to kadr/matte_tool.py and put a model where
`find()` is: this one calls «the thing» whatever is brighter than middle grey, which is enough to test the plumbing
(tests/ae/test_matte.py runs it) and nothing more.

    matte_tool.py W H person|subject|soft [x y]     frames in → mattes out
        person   the people of the frame, clean edges
        subject  whatever stands out; with x y (frame px) only the thing under that point — and the same thing in the
                 frames that follow, wherever it moves
        soft     the people with soft edges (hair, motion blur)
    matte_tool.py W H boxes person|subject          frames in → one line of numbers a frame (kadr reframe):
        cx cy x0 y0 x1 y1 share fx fy fw fh         the matte's middle, its box, the share of the frame it takes, the
                                                    largest face's box (left, top, width, height); −1 where there is none
    matte_tool.py W H pose                          frames in → for each frame lines «name x y confidence», then «end»:
        nose neck root eye_a eye_b ear_a ear_b shoulder_a shoulder_b elbow_a elbow_b wrist_a wrist_b hip_a hip_b knee_a
        knee_b ankle_a ankle_b (the person's own left is _a, right is _b; Kadr renames them by where they are), and of
        the largest face: face eye_a eye_b mouth chin crown. Only the joints that are seen.

    stdin   frames of W × H, four bytes a pixel: B, G, R, A — one after another until the input ends
    stdout  mattes of W × H, one byte a pixel (255 where the thing is), one for every frame read; or the lines above
    stderr  «frames N» now and then, and the last line, always:  done N empty E cover C
            (N frames read, E of them with nothing found, C the share of the frame the mattes took on average)

Rules for the real one: the model and its runtime must allow commercial use (read the licence of the weights, not only
of the code); nothing is sent anywhere — the pictures stay on the machine; the weights are fetched once, by a command
the user runs knowingly, at a pinned revision (as kadr depth --download does)."""
import sys

import numpy as np


def find(bgra, what, point):
    """the matte of one frame: an array H × W of 0…255. THE PART TO REPLACE WITH A MODEL"""
    gray = bgra[..., :3].mean(-1)
    m = np.where(gray > 127, 255, 0).astype(np.uint8)
    if point is not None and not m[min(max(point[1], 0), m.shape[0] - 1), min(max(point[0], 0), m.shape[1] - 1)]:
        m[:] = 0                                               # nothing under the point
    return m


def main():
    a = sys.argv[1:]
    w, h = int(a[0]), int(a[1])
    boxes, pose = a[2] == 'boxes', a[2] == 'pose'
    what = (a[3] if len(a) > 3 else 'person') if boxes else a[2]
    point = (int(a[3]), int(a[4])) if not boxes and not pose and len(a) > 4 else None
    src, out = sys.stdin.buffer, sys.stdout.buffer            # bytes, not text: on Windows the text streams would change them
    n = empty = 0
    cover = 0.0
    while True:
        raw = src.read(w * h * 4)
        if len(raw) < w * h * 4:
            break
        m = find(np.frombuffer(raw, np.uint8).reshape(h, w, 4), 'person' if pose else what, point)
        on = m > 127
        ys, xs = np.nonzero(on)
        n += 1
        empty += not len(xs)
        cover += len(xs) / (w * h)
        if pose:
            if len(xs):                                        # a stand-in: a real one names the joints it sees
                out.write(f'nose {xs.mean():.1f} {ys.mean():.1f} 1.000\n'.encode())
            out.write(b'end\n')
        elif boxes:
            row = f'{xs.mean():.1f} {ys.mean():.1f} {xs.min()} {ys.min()} {xs.max()} {ys.max()} {len(xs) / (w * h):.5f}' if len(xs) else '-1 -1 -1 -1 -1 -1 0'
            out.write((row + ' -1.0 -1.0 -1.0 -1.0\n').encode())      # no face finder here
        else:
            out.write(m.tobytes())
            if n % 25 == 0:
                print(f'frames {n}', file=sys.stderr)
    out.flush()
    print(f'done {n} empty {empty} cover {cover / n if n else 0}', file=sys.stderr)


if __name__ == '__main__':
    main()
