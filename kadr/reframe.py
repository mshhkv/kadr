"""kadr reframe — keeping the one who matters in the frame when the frame is narrower than the footage (a horizontal
film cut into vertical reels; After Effects / Premiere: Auto Reframe):

    kadr reframe talk.mp4               where the people are in every shot → talk.frame.json beside the clip
        --what subject                  …or whatever stands out (a product, an animal)
        --sheet out.jpg                 frames with the 9:16 window drawn where it would stand, to look at
    in a document (say 1080 × 1920, the clip 1920 × 1080):
        - {id: talk, type: video, src: talk.mp4, reframe: true}
        reframe: {smooth: 0.6}          seconds the window's move is evened over (it follows like a calm operator;
                                        where that would let the person out of the frame's middle part, the shot is
                                        followed closer by itself)
    The layer is laid as it would be (it covers the frame) and then slid so that the person — the face when one is
    seen, else the middle of the body — stands in the middle of the frame, never so far that the footage's edge shows.
    Each shot of the clip is followed by itself: the window jumps with the cut, it does not travel across it.

People and faces are found by what macOS has built in (the Vision framework, as kadr matte), every third frame."""
import json, math, os, subprocess

import numpy as np

STEP = 3


def run(src, what='person', step=STEP):
    """find the subject through the clip, write clip.frame.json → a report"""
    from .matte import tool
    from .media import probe
    from .track import find_cuts, sidecar
    if what not in ('person', 'subject'):
        raise RuntimeError('--what: person или subject')
    exe = tool()
    p = probe(src)
    k = min(1.0, 512 / max(p['w'], p['h']))
    w, h = max(16, int(round(p['w'] * k / 2)) * 2), max(16, int(round(p['h'] * k / 2)) * 2)
    dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-i', src, '-an', '-vf', f"select='not(mod(n\\,{step}))',scale={w}:{h}:flags=area", '-fps_mode', 'passthrough',
                            '-f', 'rawvideo', '-pix_fmt', 'bgra', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    r = subprocess.run(exe + [str(w), str(h), 'boxes', what], stdin=dec.stdout, capture_output=True)
    dec.stdout.close()
    dec.wait()
    rows = [[float(v) for v in ln.split()] for ln in r.stdout.decode().splitlines() if ln.strip()]
    if r.returncode or not rows:
        raise RuntimeError('не удалось найти людей в клипе: ' + r.stderr.decode('utf-8', 'replace')[-400:])
    a = np.array(rows)
    kx, ky = p['w'] / w, p['h'] / h
    a[:, [0, 2, 4, 7, 9]] = np.where(a[:, [0, 2, 4, 7, 9]] >= 0, a[:, [0, 2, 4, 7, 9]] * kx, -1)
    a[:, [1, 3, 5, 8, 10]] = np.where(a[:, [1, 3, 5, 8, 10]] >= 0, a[:, [1, 3, 5, 8, 10]] * ky, -1)
    cuts, fps, n = find_cuts(src)
    out = sidecar(src, 'frame')
    json.dump(dict(clip=dict(w=p['w'], h=p['h'], fps=fps, frames=n), what=what, step=step, cuts=[round(c, 4) for c in cuts], rows=[[round(float(v), 2) for v in row] for row in a]),
              open(out, 'w', encoding='utf-8'))
    seen = a[:, 0] >= 0
    return dict(out=out, samples=len(a), seen=int(seen.sum()), faces=int((a[:, 7] >= 0).sum()), shots=len(cuts) + 1,
                wander=float(a[seen, 0].max() - a[seen, 0].min()) if seen.any() else 0.0, size=(p['w'], p['h']))


_paths = {}


def path(src, smooth=.6, keep=None, steady=None, lim=None):
    """where the window looks on every frame of the clip: the subject's place, evened shot by shot → (frames, 2) clip px.
    keep (x, y), clip px: how far from the window's middle the subject may be — a shot where the evened path lets it
    further is evened half as much, and again, until it holds. steady (a name for the cache, the stabiliser's matrix
    for every frame): the clip is shown steadied — the subject is followed in the steadied picture (each place found
    is carried there before the evening, so the camera's shake taken out of the picture is not put back by the window).
    lim ((x from, x to), (y from, y to)), clip px: how far the window's middle can go before the footage's edge shows —
    the places found are held within it before the evening, so the window comes up to the edge gently (held after, it
    stopped there in one frame: a jerk of 1.7 px a frame, named by kadr check on the first clip tried)"""
    from .track import sidecar, _load
    sc = sidecar(src, 'frame')
    key = (sc, os.path.getmtime(sc), smooth, None if keep is None else (round(keep[0]), round(keep[1])), steady and steady[0],
           lim and tuple(None if a is None else (round(a[0], 1), round(a[1], 1)) for a in lim))
    if key in _paths:
        return _paths[key]
    d = _load(sc)
    a = np.array(d['rows'], float)
    n, step, fps = d['clip']['frames'], d['step'], d['clip']['fps']
    at = np.arange(len(a)) * step                              # the frames that were looked at
    edges = [0] + [int(round(c * fps)) for c in d['cuts']] + [n]
    out = np.zeros((n, 2))
    mid = np.array([d['clip']['w'] / 2, d['clip']['h'] / 2])
    for f0, f1 in zip(edges, edges[1:]):
        m = (at >= f0) & (at < f1)
        rows, fr = a[m], at[m]
        seen = rows[:, 0] >= 0 if len(rows) else np.zeros(0, bool)
        if not seen.any():
            out[f0:f1] = mid                                   # nobody in this shot: its middle
            continue
        face = rows[:, 7] >= 0
        if face.sum() >= .6 * seen.sum():                      # a face in most of it: the window follows the face
            pts, ok = np.stack([rows[:, 7] + rows[:, 9] / 2, rows[:, 8] + rows[:, 10] / 2], 1), face
        else:
            pts, ok = rows[:, :2], seen
        if steady is not None:
            A = steady[1][np.clip(fr[ok], 0, len(steady[1]) - 1)]
            pts = pts.copy()
            pts[ok] = np.einsum('nij,nj->ni', A, np.concatenate([pts[ok], np.ones((int(ok.sum()), 1))], 1))[:, :2]
        raw = np.stack([np.interp(fr, fr[ok], pts[ok, c]) for c in (0, 1)], 1)
        for c in (0, 1):
            if lim is not None and lim[c] is not None:
                raw[:, c] = np.clip(raw[:, c], lim[c][0], max(lim[c]))
        sg = smooth * fps / step
        for _ in range(6):
            x = raw
            if sg > .05:
                r = int(math.ceil(3 * sg))
                ker = np.exp(-.5 * (np.arange(-r, r + 1) / sg) ** 2)
                x = np.stack([np.convolve(np.pad(raw[:, c], (r, r), mode='edge'), ker / ker.sum(), 'valid') for c in (0, 1)], 1)
            if keep is None or (np.abs(raw - x).max(0) <= np.asarray(keep)).all():
                break
            sg /= 2
        full = np.arange(f0, f1)
        out[f0:f1] = np.stack([np.interp(full, fr, x[:, c]) for c in (0, 1)], 1)
    if len(_paths) > 16:
        _paths.clear()
    _paths[key] = out
    return out


def shift(film, L, t, px, py, sx, sy):
    """the layer's place moved so that the subject stands in the middle of the frame, the frame staying covered"""
    from .render import content_size, source_size, video_time, pivot_px
    from .media import probe
    cw, ch = content_size(film, L, t)
    iw, ih = source_size(L)
    steady = None
    if L.get('stab') is not None:
        from .track import plan_of, sidecar
        sp = sidecar(L.get('stab_src') or L['src'], 'stab')
        steady = ((sp, os.path.getmtime(sp), tuple(sorted(L['stab'].items())), L['from'], L['in'], L['out'], L['speed']), np.asarray(plan_of(L)['A']))
    off = L['crop'][:2] if L.get('crop') else (0.0, 0.0)
    lim = []
    for frame, s, size, isz, o in ((film.W, abs(sx), cw, iw, off[0]), (film.H, abs(sy), ch, ih, off[1])):
        half = frame / (2 * max(s, 1e-6))                      # half the frame, in the layer's own px
        lim.append(None if size * s <= frame + 1e-6 else (half * isz / size + o - .5, (size - half) * isz / size + o - .5))
    P = path(L.get('reframe_src') or L['src'], L['reframe']['smooth'], (.3 * film.W / max(1e-6, abs(sx)) * iw / cw, .3 * film.H / max(1e-6, abs(sy)) * ih / ch), steady, tuple(lim))
    n = int(math.floor(video_time(L, t) * probe(L['src'])['fps'] + 1e-4))
    sub = P[max(0, min(len(P) - 1, n))] + .5
    if L.get('crop'):
        sub = sub - np.array(L['crop'][:2])
    cx, cy = sub[0] * cw / iw, sub[1] * ch / ih
    if L.get('flip'):
        cx = cw - cx
    ax, ay = pivot_px(film, L, t)
    out = []
    for p, c, a, s, size, frame in ((px, cx, ax, abs(sx), cw, film.W), (py, cy, ay, abs(sy), ch, film.H)):
        if size * s <= frame + 1e-6:                           # nothing to spare this way: as it was laid
            out.append(p)
            continue
        want = frame / 2 - (c - a) * s                         # the subject in the middle
        out.append(min(a * s, max(frame - (size - a) * s, want)))
    return out[0], out[1]


def sheet(src, out, aspect=9 / 16, count=8):
    """frames of the clip with the window of this shape drawn where it would stand → a picture to look at"""
    from PIL import Image, ImageDraw
    from .media import VideoReader
    from .media import probe
    p = probe(src)
    P = path(src, keep=(.3 * min(p['w'], p['h'] * aspect), .3 * p['h']))
    r = VideoReader(src)
    picks = sorted(set(int(round(i)) for i in np.linspace(0, min(r.last, len(P) - 1), count)))
    k = min(1.0, 480 / r.w)
    tw, th = int(r.w * k), int(r.h * k)
    ww = min(r.w, r.h * aspect)
    page = Image.new('RGB', (4 * tw, -(-len(picks) // 4) * (th + 18)), '#111111')
    for j, i in enumerate(picks):
        fr = Image.fromarray(r.at((i + .5) / r.fps)).resize((tw, th))
        x0 = min(r.w - ww, max(0, P[i][0] - ww / 2))
        d = ImageDraw.Draw(fr)
        d.rectangle([x0 * k, 1, (x0 + ww) * k, th - 2], outline='#ffe000', width=2)
        d.ellipse([P[i][0] * k - 4, P[i][1] * k - 4, P[i][0] * k + 4, P[i][1] * k + 4], fill='#ff3030')
        page.paste(fr, ((j % 4) * tw, (j // 4) * (th + 18) + 18))
        ImageDraw.Draw(page).text(((j % 4) * tw + 4, (j // 4) * (th + 18) + 3), f'{i / r.fps:.2f} s', fill='#ffd060')
    r.close()
    page.save(out, quality=90)
    return out
