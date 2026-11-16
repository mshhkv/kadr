"""Following what moves in footage (AE.md, M7): a thing tracked through a clip, and a shaky camera steadied
(After Effects: point and planar tracking, Mocha, corner pin; Warp Stabilizer).

    kadr track shot.mp4 --name screen --at 1.2 --region "410,220 760,236 752,470 402,448"
        the region — four corners (top left, top right, bottom right, bottom left) or a rectangle x0,y0,x1,y1, in the
        clip's own px at second --at — is followed through the clip (--from / --to: only this stretch) and written to
        shot.track.json beside it. --surface "x,y x,y x,y x,y": the four corners a pinned layer is put on, when they
        are not the region's own (the region takes in a screen's frame — there is drawing to hold on to; the surface
        is the glass inside it). --model: plane (a flat thing seen from any side: a screen, a poster, a wall — the
        default for four corners), similar (place, size, turn — the default for a rectangle), point (place only),
        affine. --sheet out.jpg: frames with the found outline drawn, to look at.
    in a document, with `shot` a video layer of that clip:
        position:   =track("shot", "screen").position      where its middle is in the frame, through the layer's place
        corner_pin: =track("shot", "screen").corners       a layer pinned on it, corner to corner (a screen replaced)
        scale:      =track("shot", "screen").scale         its size against the second it was named at; .rotation — its turn
        opacity:    =1 - track("shot", "screen").lost      hidden while it is covered or out of the frame
        .map(x, y) — any point of the clip's px carried along with it;  .local — in the layer's own px (for a child);
        .q — how sure the match is, 0 … 1;  .at(t) — as at another moment

    kadr cuts clip.mp4                 where the clip jumps from one shot to another: the second each new shot begins at

    kadr stabilize shot.mp4            the camera's path through the clip → shot.stab.json beside it
        --region "x0,y0,x1,y1"         judge the camera by this part of the first frame only (the shore, a wall): for a
                                       scene half of which moves by itself — water, a crowd
    in a document:
        - {id: shot, type: video, src: shot.mp4, stabilize: true}
        stabilize: {smooth: 1.0, lock: false, zoom: auto}
            smooth — seconds the camera's path is evened over (what stays is a calm move);  lock: true — no move at
            all, a tripod;  zoom — how much the picture is enlarged so that its edges never show: auto (the least that
            hides them over the stretch the layer plays) or a number.
        A track made on the clip follows the steadied picture by itself.

How: frames are matched by their own picture — the warp (shift, shift + size + turn, affine, or the eight numbers of a
plane) under which a piece of one frame lies on the other with the least difference, found from coarse to fine
(Lucas–Kanade, inverse compositional), brightness and contrast taken out, what does not belong (a hand passing over,
a person walking through) weighted down. numpy only: no model, nothing to download."""
import json, math, os, subprocess

import numpy as np

MODELS = {'point': 2, 'similar': 4, 'affine': 6, 'plane': 8}


# ---------------- pictures ----------------
def half(a):
    """the picture at half the size: each pixel the mean of four"""
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) * .25


def pyramid(a, levels):
    out = [np.asarray(a, np.float32)]
    for _ in range(levels - 1):
        if min(out[-1].shape) < 16:
            break
        out.append(half(out[-1]))
    return out


def sample(img, x, y):
    """the picture at fractional pixels (bilinear) → values, and which of them lie inside it"""
    h, w = img.shape
    ok = (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
    x0 = np.clip(np.floor(x), 0, w - 2).astype(np.int64)
    y0 = np.clip(np.floor(y), 0, h - 2).astype(np.int64)
    fx, fy = np.clip(x - x0, 0, 1).astype(np.float32), np.clip(y - y0, 0, 1).astype(np.float32)
    i = y0 * w + x0
    f = img.ravel()
    v = (f[i] * (1 - fx) + f[i + 1] * fx) * (1 - fy) + (f[i + w] * (1 - fx) + f[i + w + 1] * fx) * fy
    return v, ok


def _jac(model, u, v):
    """how a point (u, v) moves with each number of the warp, at no warp → (n, 2, k)"""
    o, z = np.ones_like(u), np.zeros_like(u)
    if model == 'point':
        J = [[o, z], [z, o]]
    elif model == 'similar':                                   # u' = (1 + a) u − b v + tx,  v' = b u + (1 + a) v + ty
        J = [[u, -v, o, z], [v, u, z, o]]
    elif model == 'affine':
        J = [[u, v, o, z, z, z], [z, z, z, u, v, o]]
    else:
        J = [[u, v, o, z, z, z, -u * u, -u * v], [z, z, z, u, v, o, -u * v, -v * v]]
    return np.stack([np.stack(r, -1) for r in J], 1)


def _mat(model, p):
    """the warp of these numbers, as a 3×3 matrix"""
    M = np.eye(3)
    if model == 'point':
        M[0, 2], M[1, 2] = p
    elif model == 'similar':
        M[0, 0] += p[0]; M[1, 1] += p[0]; M[0, 1] -= p[1]; M[1, 0] += p[1]; M[0, 2], M[1, 2] = p[2], p[3]
    elif model == 'affine':
        M[0] += p[:3]; M[1] += p[3:6]
    else:
        M[0] += p[:3]; M[1] += p[3:6]; M[2, :2] += p[6:8]
    return M


class Template:
    """a piece of a frame to look for in others: the pixels of `quad` (4 points, full-size px) at every level of the
    pyramid, with what the inverse-compositional step needs of them"""

    def __init__(self, gray, quad, levels=None, cap=30000):
        quad = np.asarray(quad, float)
        self.quad = quad
        self.c = quad.mean(0)
        self.r = max(4.0, float(np.hypot(*(quad - self.c).T).max()))
        x0, y0 = np.floor(quad.min(0)).astype(int)
        x1, y1 = np.ceil(quad.max(0)).astype(int)
        side = min(x1 - x0, y1 - y0)
        if levels is None:
            levels = int(min(5, max(1, math.floor(math.log2(max(1.0, side / 12))) + 1)))
        self.levels = []
        pyr = pyramid(gray, levels)
        for l, im in enumerate(pyr):
            s = 2 ** l
            h, w = im.shape
            # the level's pixels inside the quad: level pixel i stands at full-size s (i + ½) − ½
            lx0, ly0 = max(1, int(math.floor((x0 + .5) / s - .5))), max(1, int(math.floor((y0 + .5) / s - .5)))
            lx1, ly1 = min(w - 2, int(math.ceil((x1 + .5) / s - .5))), min(h - 2, int(math.ceil((y1 + .5) / s - .5)))
            if lx1 - lx0 < 3 or ly1 - ly0 < 3:
                continue
            gy, gx = np.mgrid[ly0:ly1 + 1, lx0:lx1 + 1]
            X, Y = s * (gx + .5) - .5, s * (gy + .5) - .5
            inside = _in_quad(X, Y, quad)
            step = int(math.ceil(math.sqrt(max(1.0, inside.sum() / cap))))
            if step > 1:
                keep = np.zeros_like(inside)
                keep[::step, ::step] = True
                inside &= keep
            iy, ix = np.nonzero(inside)                        # each pixel's place in the lattice kept (for what lies around it)
            iy, ix, lat = iy // step, ix // step, (inside.shape[0] // step + 1, inside.shape[1] // step + 1)
            gy, gx, X, Y = gy[inside], gx[inside], X[inside], Y[inside]
            if len(X) < 24:
                continue
            T = im[gy, gx]
            Tx = (im[gy, gx + 1] - im[gy, gx - 1]) / (2 * s)   # per full-size px
            Ty = (im[gy + 1, gx] - im[gy - 1, gx]) / (2 * s)
            m, sd = float(T.mean()), float(T.std()) + 1e-6
            edge = np.hypot(Tx, Ty).astype(np.float64)         # where the piece has drawing: only there can a frame agree or not
            self.levels.append(dict(s=s, iy=iy, ix=ix, lat=lat, edge=edge / (edge.mean() + 1e-9), rel=np.ones(len(X)), X=X.astype(np.float64), Y=Y.astype(np.float64), T=((T - m) / sd).astype(np.float64),
                                    g=np.stack([Tx, Ty], -1).astype(np.float64) / sd * self.r))
        self.N = np.array([[1 / self.r, 0, -self.c[0] / self.r], [0, 1 / self.r, -self.c[1] / self.r], [0, 0, 1.0]])
        self.Ni = np.linalg.inv(self.N)
        self._sd = {}

    def learn(self, rate=.1):
        """after a frame that was accepted: which parts of the piece agreed with it. A part that keeps disagreeing
        while the rest holds is a part that changes by itself — a picture playing on a tracked screen, a reflection
        — and is trusted less and less (a monitor with moving curves was called lost while in plain view)"""
        for L in self.levels:
            if L.get('_ok') is not None:
                L['rel'] = (1 - rate) * L['rel'] + rate * L['_ok']
                L['_ok'] = None

    def sd(self, li, model):
        """each pixel's change with each number of the warp (the template's slope × how the point moves)"""
        if (li, model) not in self._sd:
            L = self.levels[li]
            u, v = (L['X'] - self.c[0]) / self.r, (L['Y'] - self.c[1]) / self.r
            self._sd[li, model] = np.einsum('nd,ndk->nk', L['g'], _jac(model, u, v))
        return self._sd[li, model]


def _in_quad(X, Y, quad):
    """which points lie inside the convex quad (corners in order, either way round)"""
    sign = None
    ok = np.ones(X.shape, bool)
    for i in range(4):
        (ax, ay), (bx, by) = quad[i], quad[(i + 1) % 4]
        cr = (bx - ax) * (Y - ay) - (by - ay) * (X - ax)
        if sign is None:
            sign = 1.0 if (quad[2][0] - ax) * (by - ay) - (quad[2][1] - ay) * (bx - ax) < 0 else -1.0
        ok &= cr * sign >= 0
    return ok


def _around(v, L, r=3):
    """the mean of v over each template pixel's neighbourhood (2r + 1 lattice cells square)"""
    h, w = L['lat']
    S, C = np.zeros((h + 1, w + 1)), np.zeros((h + 1, w + 1))
    np.add.at(S, (L['iy'] + 1, L['ix'] + 1), v)
    np.add.at(C, (L['iy'] + 1, L['ix'] + 1), 1.0)
    S, C = S.cumsum(0).cumsum(1), C.cumsum(0).cumsum(1)
    y0, y1 = np.clip(L['iy'] - r, 0, h), np.clip(L['iy'] + r + 1, 0, h)
    x0, x1 = np.clip(L['ix'] - r, 0, w), np.clip(L['ix'] + r + 1, 0, w)
    box = lambda A: A[y1, x1] - A[y0, x1] - A[y1, x0] + A[y0, x0]
    return box(S) / np.maximum(1.0, box(C))


def register(tpl, pyr, W, model, iters=(40, 20, 12, 8, 6), floor=0):
    """the warp W (3×3, the template's frame px → this frame's px) under which the template lies on the frame `pyr`
    (its pyramid), starting from the W given → (W, how alike they are −1 … 1, the share of the template seen in the
    frame, the share of it that agrees — the rest is covered by something else).
    What does not belong counts for nothing, not for less: at the fine levels a pixel beyond 4.7 typical differences
    has no say, and neither has a pixel in a neighbourhood that differs on the whole — under a hand a fifth of the
    pixels agree by chance, and those pulled the plane's corners 8 px off. Brightness and contrast are taken from the
    pixels that agree. → q is how alike the two are over the part that agrees"""
    W = np.array(W, float)
    q, seen, agree = -1.0, 0.0, 0.0
    n_lv = len(tpl.levels)
    for li in range(n_lv - 1, floor - 1, -1):
        L = tpl.levels[li]
        s = L['s']
        l_img = int(round(math.log2(s)))
        if l_img >= len(pyr):
            continue
        img = pyr[l_img]
        # a plane's eight numbers are found only where there is picture enough for them: coarse levels use fewer
        m = model if not (model == 'plane' and li >= n_lv - 1 and n_lv > 2) else 'similar'
        m = m if not (model in ('plane', 'affine', 'similar') and len(L['X']) < 60) else 'point'
        SD = tpl.sd(li, m)
        X1 = np.stack([L['X'], L['Y'], np.ones_like(L['X'])], 0)
        Wn = tpl.N @ W @ tpl.Ni
        wgt = None
        for it in range(iters[min(li, len(iters) - 1)]):
            Wf = tpl.Ni @ Wn @ tpl.N
            P = Wf @ X1
            x, y = P[0] / P[2], P[1] / P[2]
            a, ok = sample(img, (x + .5) / s - .5, (y + .5) / s - .5)
            seen = float(ok.mean())
            if ok.sum() < 24:
                return tpl.Ni @ Wn @ tpl.N, -1.0, seen, 0.0
            a = a.astype(np.float64)
            w0 = ok.astype(np.float64) if wgt is None or (wgt > .5).sum() < 24 else wgt
            mu = (w0 * a).sum() / w0.sum()
            sd = math.sqrt((w0 * (a - mu) ** 2).sum() / w0.sum()) + 1e-6
            T = L['T']
            if w0 is not ok:                                   # the template's own level over the same pixels
                tm = (w0 * T).sum() / w0.sum()
                T = (T - tm) / (math.sqrt((w0 * (T - tm) ** 2).sum() / w0.sum()) + 1e-6)
            an = (a - mu) / sd
            e = np.where(ok, an - T, 0.0)
            sig = min(.4, max(.05, 1.4826 * float(np.median(np.abs(e[ok] - np.median(e[ok]))))))
            u = e / (4.685 * sig)
            loc = _around(np.where(ok, np.abs(e), 2.0), L)     # how much each pixel's neighbourhood differs, in the template's own contrast
            # 1 where the neighbourhood agrees, 0 where it does not — by this match's own noise, and by an absolute
            # measure too: when everything differs the noise estimate grows with it and nothing would look out of place
            near = np.clip(3.5 - loc / sig, 0, 1) * (loc < .6)
            tuk = np.where(ok & (np.abs(u) < 1), (1 - u * u) ** 2, 0.0) * near * L['rel']
            wgt = tuk if li <= 1 else np.where(ok, np.minimum(1.0, 1.345 * sig / (np.abs(e) + 1e-9)), 0.0)
            # the share of the piece's drawing that lies on the frame as it should. Counted where the piece has drawing:
            # a dark jacket over a dark screen differs from it nowhere but at the screen's edges, which are gone
            say = L['edge'] * L['rel']                         # …and where that drawing is known to stay as it is
            agree = float((say * (loc < .6))[ok].sum() / max(1e-9, say[ok].sum()))
            L['_ok'] = np.where(ok, loc < .6, L['rel'] > .5).astype(float)
            q = float((tuk * an * T).sum() / max(1e-9, tuk.sum())) if tuk.sum() > 12 else -1.0
            A = SD * wgt[:, None]
            Hm = A.T @ SD + np.eye(len(SD[0])) * 1e-9
            try:
                dp = np.linalg.solve(Hm, A.T @ e)
            except np.linalg.LinAlgError:
                break
            Wn = Wn @ np.linalg.inv(_mat(m, dp))
            Wn /= Wn[2, 2]
            if np.abs(dp).max() < 2e-4 / max(1.0, s / 2):
                break
        W = tpl.Ni @ Wn @ tpl.N
    return W / W[2, 2], q, seen, agree


def coarse_shift(tpl, pyr, W, reach):
    """a jump too far for the fine search: the template slid over the frame at its coarsest level, up to `reach`
    full-size px each way → W moved to where it fits best"""
    L = tpl.levels[-1]
    s = L['s']
    img = pyr[min(len(pyr) - 1, int(round(math.log2(s))))]
    s_img = 2 ** min(len(pyr) - 1, int(round(math.log2(s))))
    P = W @ np.stack([L['X'], L['Y'], np.ones_like(L['X'])], 0)
    x, y = (P[0] / P[2] + .5) / s_img - .5, (P[1] / P[2] + .5) / s_img - .5
    R = max(1, int(round(reach / s_img)))
    best, at = -2.0, (0, 0)
    step = max(1, int(round(s / s_img / 2)) or 1)
    for dy in range(-R, R + 1, step):
        for dx in range(-R, R + 1, step):
            a, ok = sample(img, x + dx, y + dy)
            if ok.mean() < .6:
                continue
            a = a[ok].astype(np.float64)
            q = float((((a - a.mean()) / (a.std() + 1e-6)) * L['T'][ok]).mean())
            if q > best:
                best, at = q, (dx, dy)
    T = np.eye(3)
    T[0, 2], T[1, 2] = at[0] * s_img, at[1] * s_img
    return T @ W, best


# ---------------- a clip's frames ----------------
def frames(path, i0, i1, long_side):
    """gray frames i0 … i1 of the clip, the long side at most `long_side` px → (array (n, h, w) uint8 kept in a file
    on disk, S: clip px → these px as a 3×3 matrix, the file to remove afterwards)"""
    import tempfile
    from .media import probe
    p = probe(path)
    k = min(1.0, long_side / max(p['w'], p['h']))
    w, h = max(16, int(round(p['w'] * k / 2)) * 2), max(16, int(round(p['h'] * k / 2)) * 2)
    n = i1 - i0 + 1
    tmp = tempfile.NamedTemporaryFile(prefix='kadr_track_', suffix='.gray', delete=False)
    tmp.close()
    out = np.memmap(tmp.name, np.uint8, 'w+', shape=(n, h, w))
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-i', path, '-an', '-vf', f'scale={w}:{h}:flags=area,format=gray', '-frames:v', str(i1 + 1),
                             '-f', 'rawvideo', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 4)
    got = 0
    for i in range(i1 + 1):
        buf = proc.stdout.read(w * h)
        if len(buf) < w * h:
            break
        if i >= i0:
            out[i - i0] = np.frombuffer(buf, np.uint8).reshape(h, w)
            got += 1
    proc.stdout.close()
    proc.wait()
    kx, ky = w / p['w'], h / p['h']
    S = np.array([[kx, 0, (kx - 1) / 2], [0, ky, (ky - 1) / 2], [0, 0, 1.0]])   # pixel centres: clip i ↔ (i + ½) k − ½
    return out[:got], S, tmp.name


def sidecar(src, kind):
    """the file beside a clip that holds what was found in it: shot.mp4 → shot.track.json, shot.stab.json"""
    return os.path.splitext(src)[0] + f'.{kind}.json'


def region(text):
    """«x0,y0,x1,y1» or «x,y x,y x,y x,y» (top left, top right, bottom right, bottom left) → 4 corners"""
    nums = [float(v) for v in str(text).replace(',', ' ').split()]
    if len(nums) == 4:
        x0, y0, x1, y1 = nums
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], False
    if len(nums) == 8:
        return [[nums[i], nums[i + 1]] for i in range(0, 8, 2)], True
    raise ValueError('область: «x0,y0,x1,y1» или четыре угла «x,y x,y x,y x,y» (левый верхний, правый верхний, правый нижний, левый нижний)')


def _area(q):
    x, y = np.asarray(q, float).T
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2


def _pts(H, q):
    P = H @ np.c_[np.asarray(q, float), np.ones(len(q))].T
    return (P[:2] / P[2]).T


def _follow(G, ref, quad, model, order, reach):
    """the quad of frame `ref` through the frames `order` (indices into G, each next to the last) → {i: (W, q, lost)}"""
    tpl, Wkey = Template(G[ref].astype(np.float32), quad), np.eye(3)
    a0 = _area(quad)
    W, step, out = np.eye(3), np.eye(3), {}
    for i in order:
        pyr = pyramid(G[i].astype(np.float32), 6)
        Ki = np.linalg.inv(Wkey)
        best = None
        for guess in (step @ W, W):                            # as it was moving; else where it last was
            got = register(tpl, pyr, guess @ Ki, model)
            if best is None or got[1] > best[1]:
                best = got
            if got[1] > .8:
                break
        if best[1] < .6:                                       # a jump: look around for it, coarsely, then finely again
            Wc, qc = coarse_shift(tpl, pyr, W @ Ki, reach)
            if qc > .3:
                got = register(tpl, pyr, Wc, model)
                if got[1] > best[1]:
                    best = got
        if model in ('plane', 'affine') and best[3] < .85 and best[1] > .6:
            # part of it is covered. What is left may be too little to tell a plane's tilt: with a shoulder over a
            # monitor's side its free corner swam 40 px. Found again as place, size and turn only, from where it was
            # heading: if the two answers part by more than 3 % of its size, the stiffer one is taken
            stiff = register(tpl, pyr, step @ W @ Ki, 'similar')
            if stiff[1] > .6:
                gap = float(np.hypot(*(_pts(best[0], tpl.quad) - _pts(stiff[0], tpl.quad)).T).max())
                if gap > .03 * math.sqrt(_area(_pts(stiff[0], tpl.quad))):
                    best = stiff
        Wn, q, seen, agree = best
        Wn = Wn @ Wkey
        ratio = _area(_pts(Wn, quad)) / a0
        lost = q < .6 or seen < .35 or agree < .3 or not (.04 < ratio < 25) or not np.isfinite(Wn).all()
        if not lost and agree < .9:
            # a doubtful match must also be where the thing could be: a plane's eight numbers will find some look-alike
            # patch for a covered card (65–150 px away in the test) — a jump of more than a seventh of its size from
            # where it was heading, with part of it not agreeing, is a loss, not a move
            jump = float(np.hypot(*(_pts(Wn, quad) - _pts(step @ W, quad)).T).max())
            lost = jump > .15 * math.sqrt(_area(_pts(W, quad)))
        if lost:
            out[i] = (W, max(0.0, q), True)                    # held where it was last seen
            step = np.eye(3)
            continue
        step, W = Wn @ np.linalg.inv(W), Wn
        step /= step[2, 2]
        out[i] = (W, q, False)
        tpl.learn()
        rel = _area(_pts(W @ Ki, tpl.quad)) / _area(tpl.quad)
        if q > .85 and agree > .95 and seen > .98 and not (.55 < rel < 1.8):   # much nearer or farther than the piece it is compared with:
            tpl, Wkey = Template(G[i].astype(np.float32), _pts(W, quad)), W    # compare with how it looks now
    return out


def track_clip(path, name, at, quad, model=None, t_from=None, t_to=None, long_side=1600, surface=None):
    """follow the quad (clip px at second `at`) through the clip, write it into the clip's .track.json → a report"""
    from .media import probe
    p = probe(path)
    fps, last = p['fps'], max(0, int(p['duration'] * p['fps']) - 1)
    idx = lambda t: min(last, max(0, int(math.floor(t * fps + 1e-4))))
    r = idx(at)
    i0, i1 = (idx(t_from) if t_from is not None else 0), (idx(t_to) if t_to is not None else last)
    if not i0 <= r <= i1:
        raise ValueError(f'секунда --at {at:g} не внутри отрезка --from … --to')
    quad = np.asarray(quad, float)
    if model not in MODELS:
        raise ValueError(f'модель «{model}»: нужно ' + ' | '.join(MODELS))
    G, S, tmp = frames(path, i0, i1, long_side)
    try:
        i1 = i0 + len(G) - 1
        qa = _pts(S, quad)
        reach = .2 * max(G.shape[1:])
        res = {r - i0: (np.eye(3), 1.0, False)}
        res.update(_follow(G, r - i0, qa, model, range(r - i0 + 1, len(G)), reach))
        res.update(_follow(G, r - i0, qa, model, range(r - i0 - 1, -1, -1), reach))
    finally:
        del G
        os.remove(tmp)
    Si = np.linalg.inv(S)
    H, Q, lost = [], [], []
    for i in range(i1 - i0 + 1):
        W, q, ls = res[i]
        W = Si @ W @ S
        W /= W[2, 2]
        H.append([round(float(v), 7) for v in W.ravel()[:8]])
        Q.append(round(float(min(1.0, q)), 3))
        lost.append(int(ls))
    sc = sidecar(path, 'track')
    doc = json.load(open(sc, encoding='utf-8')) if os.path.exists(sc) else {'tracks': {}}
    doc['clip'] = dict(w=p['w'], h=p['h'], fps=fps, frames=last + 1)
    doc['tracks'][name] = dict(model=model, at=at, ref=r, first=i0, quad=[[float(x), float(y)] for x, y in quad], H=H, q=Q, lost=lost)
    if surface is not None:
        doc['tracks'][name]['surface'] = [[float(x), float(y)] for x, y in surface]
    json.dump(doc, open(sc, 'w', encoding='utf-8'))
    return dict(out=sc, frames=len(H), lost=int(sum(lost)), q=float(np.mean([q for q, l in zip(Q, lost) if not l] or [0])), first=i0, last=i1, fps=fps)


def stabilize_clip(path, long_side=640, region=None):
    """the camera's own motion through the clip — where a point of the first frame is in each later one, as a shift,
    a size and a turn — written into the clip's .stab.json → a report. Frames are compared with a key frame, not each
    with the last (errors would add up), and the key moves on every half a second at the latest, or sooner when much
    of it no longer agrees: over two seconds of a boat towing a cage across water the scene itself had changed so much
    that the match jumped from the shore to the cage and back (14 px in one frame).
    region: four corners in the first frame's px — the part of the picture that is the still world (a shore, a wall). The
    camera is then judged by it alone, wherever it goes: for a scene half of which moves by itself (water, a crowd), where
    the picture's greater part is not the ground"""
    from .media import probe
    p = probe(path)
    last = max(0, int(p['duration'] * p['fps']) - 1)
    G, S, tmp = frames(path, 0, last, long_side)
    try:
        h, w = G.shape[1:]
        inset = [[w * .06, h * .06], [w * .94, h * .06], [w * .94, h * .94], [w * .06, h * .94]]
        if region is not None:
            inset = _pts(S, np.asarray(region, float)).tolist()

        def part(Ck):                                          # the region where it is now, kept inside the frame
            if region is None:
                return inset
            return np.clip(_pts(Ck, inset), [2, 2], [w - 3, h - 3]).tolist()
        tpl, Ckey, key = Template(G[0].astype(np.float32), inset, cap=16000), np.eye(3), 0
        C, Q = [np.eye(3)], [1.0]
        step = np.eye(3)
        span = max(2, int(round(p['fps'] / 2)))
        for i in range(1, len(G)):
            pyr = pyramid(G[i].astype(np.float32), 6)
            for attempt in range(2):
                guess = step @ C[-1] @ np.linalg.inv(Ckey)
                Wr, q, seen, agree = register(tpl, pyr, guess, 'similar')
                if q < .6:
                    Wc, qc = coarse_shift(tpl, pyr, C[-1] @ np.linalg.inv(Ckey), .15 * max(w, h))
                    W2, q2, seen2, agree2 = register(tpl, pyr, Wc, 'similar')
                    if q2 > q:
                        Wr, q, seen, agree = W2, q2, seen2, agree2
                if (seen < .75 or q < .7 or agree < .7 or i - key > span) and key != i - 1:   # the last frame becomes the key
                    tpl, Ckey, key = Template(G[i - 1].astype(np.float32), part(C[-1]), cap=16000), C[-1], i - 1
                    continue
                break
            Cn = Wr @ Ckey if q > .3 else C[-1]                # nothing to hold on to (a cut, a flash): the camera stays
            step = Cn @ np.linalg.inv(C[-1])
            C.append(Cn)
            Q.append(min(1.0, max(0.0, q)))
    finally:
        del G
        os.remove(tmp)
    Si = np.linalg.inv(S)
    rows = []
    for M in C:
        M = Si @ M @ S
        rows.append([round(float(v), 7) for v in ((M[0, 0] + M[1, 1]) / 2, (M[1, 0] - M[0, 1]) / 2, M[0, 2], M[1, 2])])
    sc = sidecar(path, 'stab')
    json.dump(dict(clip=dict(w=p['w'], h=p['h'], fps=p['fps'], frames=last + 1), C=rows, q=[round(q, 3) for q in Q]), open(sc, 'w', encoding='utf-8'))
    A = np.array(rows)
    c0 = np.array([p['w'] / 2, p['h'] / 2])
    mid = np.stack([A[:, 0] * c0[0] - A[:, 1] * c0[1] + A[:, 2], A[:, 1] * c0[0] + A[:, 0] * c0[1] + A[:, 3]], 1)
    rep = dict(out=sc, frames=len(rows), fps=p['fps'], q=float(np.mean(Q)), weak=int((np.array(Q) < .5).sum()),
               shift=float(np.hypot(*(mid - c0).T).max()), turn=float(np.degrees(np.abs(np.arctan2(A[:, 1], A[:, 0])).max())),
               size=(float(np.hypot(A[:, 0], A[:, 1]).min()), float(np.hypot(A[:, 0], A[:, 1]).max())))
    for label, kw in (('smooth', dict(smooth=1.0, lock=False)), ('lock', dict(smooth=1.0, lock=True))):
        rep['zoom_' + label] = float(_plan(sc, os.path.getmtime(sc), kw['smooth'], kw['lock'], 'auto', 0, len(rows) - 1)['zoom'])
    return rep


def find_cuts(path):
    """where a clip jumps from one shot to another → (the seconds each new shot begins at, frames a second, frames).
    Each frame is made small and compared with the one before: a cut is a difference several times what the frames
    around it differ by (a pan or a zoom changes every frame about as much — a cut is one frame apart from its
    neighbours), and large in itself"""
    from .media import probe
    p = probe(path)
    w, h = 64, 36
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-i', path, '-an', '-vf', f'scale={w}:{h}:flags=area,format=gray', '-f', 'rawvideo', '-'],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 64)
    raw = np.frombuffer(proc.stdout.read(), np.uint8)
    proc.wait()
    fr = raw[:len(raw) // (w * h) * w * h].reshape(-1, h, w).astype(np.float32)
    if len(fr) < 3:
        return [], p['fps'], len(fr)
    d = np.abs(np.diff(fr, axis=0)).mean((1, 2))               # d[k]: frame k + 1 against frame k
    # the picture's own make-up as well: a cut changes how much of each brightness there is, a move within a shot does not
    hist = np.stack([np.histogram(f, 16, (0, 256))[0] for f in fr]).astype(np.float32) / (w * h)
    hd = np.abs(np.diff(hist, axis=0)).sum(1)
    cuts = []
    for k in range(len(d)):
        a, b = max(0, k - 6), min(len(d), k + 7)
        around = np.concatenate([d[a:k], d[k + 1:b]])
        base = float(np.median(around)) if len(around) else 0.0
        if d[k] > max(10.0, 3.5 * base + 4.0) and hd[k] > .25 and d[k] >= d[a:b].max() - 1e-6:
            cuts.append(k + 1)
    return [c / p['fps'] for c in cuts], p['fps'], len(fr)


# ---------------- reading them back ----------------
_cache = {}


def _load(path):
    mt = os.path.getmtime(path)
    if _cache.get(path, (None,))[0] != mt:
        _cache[path] = (mt, json.load(open(path, encoding='utf-8')))
    return _cache[path][1]


_plans = {}


def _plan(path, mtime, smooth, lock, zoom, i0, i1):
    """what to do with each frame of a clip to steady it: A[k] — frame k's px → the steadied picture's px"""
    key = (path, mtime, smooth, lock, zoom, i0, i1)
    if key in _plans:
        return _plans[key]
    d = _load(path)
    w, h, fps = d['clip']['w'], d['clip']['h'], d['clip']['fps']
    A = np.array(d['C'], float)
    n = len(A)
    i0, i1 = max(0, min(n - 1, i0)), max(0, min(n - 1, i1))
    c0 = np.array([(w - 1) / 2, (h - 1) / 2])
    sz, th = np.log(np.hypot(A[:, 0], A[:, 1])), np.unwrap(np.arctan2(A[:, 1], A[:, 0]))
    mid = np.stack([A[:, 0] * c0[0] - A[:, 1] * c0[1] + A[:, 2], A[:, 1] * c0[0] + A[:, 0] * c0[1] + A[:, 3]], 1)   # where the first frame's middle is
    if lock:
        tm, ts, tt = np.repeat(mid[i0:i0 + 1], n, 0), np.full(n, sz[i0]), np.full(n, th[i0])
    else:
        sg = max(1e-3, smooth * fps)
        r = int(math.ceil(3 * sg))
        ker = np.exp(-.5 * (np.arange(-r, r + 1) / sg) ** 2)

        def even(v):                                           # within the stretch that plays: nothing outside it pulls the path
            seg = v[i0:i1 + 1]
            pad = np.pad(seg, (r, r), mode='edge')
            out = v.copy()
            out[i0:i1 + 1] = np.convolve(pad, ker / ker.sum(), 'valid')
            return out
        tm, ts, tt = np.stack([even(mid[:, 0]), even(mid[:, 1])], 1), even(sz), even(th)

    def sim(s, t, m):                                          # the first frame's px → a frame whose middle is at m
        a, b = math.exp(s) * math.cos(t), math.exp(s) * math.sin(t)
        return np.array([[a, -b, m[0] - a * c0[0] + b * c0[1]], [b, a, m[1] - b * c0[0] - a * c0[1]], [0, 0, 1.0]])
    K = np.stack([sim(ts[k], tt[k], tm[k]) @ np.linalg.inv(sim(sz[k], th[k], mid[k])) for k in range(n)])
    corners = np.array([[-.5, -.5, 1], [w - .5, -.5, 1], [w - .5, h - .5, 1], [-.5, h - .5, 1]], float)
    Ki = np.linalg.inv(K[i0:i1 + 1])

    def covers(z):                                             # does every corner of the picture shown come from inside its frame?
        o = np.c_[c0 + (corners[:, :2] - c0) / z, np.ones(4)]
        src = np.einsum('kij,cj->kci', Ki, o)
        return bool(((src[..., 0] >= -.5 - 1e-6) & (src[..., 0] <= w - .5 + 1e-6) & (src[..., 1] >= -.5 - 1e-6) & (src[..., 1] <= h - .5 + 1e-6)).all())
    if zoom == 'auto':
        lo, hi = 1.0, 3.0
        if covers(1.0):
            hi = 1.0
        elif covers(hi):
            for _ in range(24):
                z = (lo + hi) / 2
                lo, hi = (lo, z) if covers(z) else (z, hi)
        z = hi
    else:
        z = float(zoom)
    Z = np.array([[z, 0, c0[0] * (1 - z)], [0, z, c0[1] * (1 - z)], [0, 0, 1.0]])
    out = dict(A=np.einsum('ij,kjl->kil', Z, K), zoom=z, covered=covers(z), w=w, h=h, n=n)
    if len(_plans) > 16:
        _plans.clear()
    _plans[key] = out
    return out


def plan_of(L):
    """the steadying plan of a video layer with `stabilize` (the stretch it plays decides the zoom)"""
    st = L['stab']
    sc = sidecar(L.get('stab_src') or L['src'], 'stab')   # a cut-out's matte is steadied by the footage's own path
    d = _load(sc)
    fps, n = d['clip']['fps'], len(d['C'])
    if L.get('remap') is not None or L.get('loop'):
        i0, i1 = 0, n - 1
    else:
        i0 = int(math.floor(L['from'] * fps + 1e-4))
        end = L['from'] + (L['out'] - L['in']) * L['speed'] if L['out'] < 1e8 else n / fps
        if L.get('until') is not None:
            end = min(end, L['until'])
        i1 = int(math.floor(end * fps + 1e-4))
    return _plan(sc, os.path.getmtime(sc), st['smooth'], st['lock'], st['zoom'], i0, i1)


def steady(L, frame, n):
    """frame n of the layer's clip, steadied"""
    from PIL import Image
    P = plan_of(L)
    M = np.linalg.inv(P['A'][max(0, min(P['n'] - 1, n))])     # the steadied picture's px → the frame's
    a, b, c, d, e, f = M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2]
    co = (a, b, c + .5 * (a + b) - .5, d, e, f + .5 * (d + e) - .5)   # PIL counts from a pixel's corner, the plan from its middle
    return np.asarray(Image.fromarray(frame).transform((frame.shape[1], frame.shape[0]), Image.AFFINE, co, Image.BICUBIC))


class TrackRef:
    """a tracked thing inside a formula: track("video layer", "name") — see the module's notes"""

    def __init__(self, film, L, tr, t):
        self._film, self._L, self._tr, self._t = film, L, tr, t

    def at(self, t):
        return TrackRef(self._film, self._L, self._tr, float(t))

    def _frame(self):
        """the clip's frame shown now → (index into the track clamped, the frame's number, is it outside the track)"""
        from .render import video_time
        from .media import probe
        tr = self._tr
        n = int(math.floor(video_time(self._L, self._t) * probe(self._L['src'])['fps'] + 1e-4))
        i = n - tr['first']
        return max(0, min(len(tr['H']) - 1, i)), n, not 0 <= i < len(tr['H'])

    def _H(self):
        i, n, out = self._frame()
        return np.append(np.array(self._tr['H'][i], float), 1.0).reshape(3, 3), n

    def _clip(self, pts):
        """points of the reference frame → where they are in the frame shown now, the clip's px (steadied if the layer is)"""
        H, n = self._H()
        if self._L.get('stab') is not None:
            P = plan_of(self._L)
            H = P['A'][max(0, min(P['n'] - 1, n))] @ H
        return _pts(H, pts)

    def _local(self, pts):
        from .render import content_size, source_size
        L = self._L
        p = self._clip(pts) + .5                               # a pixel's middle → the picture's own continuous px
        if L.get('crop'):
            p = p - np.array(L['crop'][:2])
        cw, ch = content_size(self._film, L, self._t)
        iw, ih = source_size(L)
        p = p * np.array([cw / iw, ch / ih])
        if L.get('flip'):
            p[:, 0] = cw - p[:, 0]
        return p

    def _world(self, pts):
        from .render import world_mat
        M = world_mat(self._film, self._L, self._t)
        p = self._local(pts)
        return p if M is None else _pts(M, p)

    def map(self, x, y):
        return tuple(float(v) for v in self._world([[float(x), float(y)]])[0])

    def __getattr__(self, name):
        tr = self._tr
        quad = np.array(tr['quad'], float)
        if name in ('position', 'x', 'y'):
            p = self._world([quad.mean(0)])[0]
            return (float(p[0]), float(p[1])) if name == 'position' else float(p[0 if name == 'x' else 1])
        if name == 'local':
            return tuple(float(v) for v in self._local([quad.mean(0)])[0])
        if name == 'corners':                                  # corner_pin's order: top left, top right, bottom left, bottom right
            return tuple(float(v) for v in self._world(np.array(tr.get('surface') or tr['quad'], float)[[0, 1, 3, 2]]).ravel())
        if name == 'scale':
            H, _ = self._H()
            return math.sqrt(_area(_pts(H, quad)) / max(1e-9, _area(quad)))
        if name == 'rotation':
            H, _ = self._H()
            a, b = _pts(H, quad[:2])
            return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]) - math.atan2(quad[1, 1] - quad[0, 1], quad[1, 0] - quad[0, 0]))
        if name in ('q', 'lost'):
            i, _, out = self._frame()
            return (0.0 if out else float(tr['q'][i])) if name == 'q' else float(bool(out or tr['lost'][i]))
        raise ValueError(f'у трека нет свойства «{name}»: есть position, x, y, corners, scale, rotation, lost, q, local, map(x, y), at(t)')


PROPS = ('position', 'x', 'y', 'corners', 'scale', 'rotation', 'lost', 'q', 'local', 'map', 'at')


def find(film, lid, name):
    """the video layer and the track a formula names → (layer, track); ValueError says what is missing"""
    L = film._by_id.get(str(lid))
    if L is None or L['type'] != 'video':
        raise ValueError(f'track("{lid}", …): нужен id видео-слоя этой композиции' + ('' if L is None else f', а «{lid}» — {L["type"]}'))
    sc = sidecar(L['src'], 'track')
    if not os.path.exists(sc):
        raise ValueError(f'у клипа {os.path.basename(L["src"])} нет трека: сначала kadr track {os.path.basename(L["src"])} --name … --at … --region …')
    tracks = _load(sc)['tracks']
    if name is None and len(tracks) == 1:
        name = next(iter(tracks))
    if name not in tracks:
        import difflib
        near = difflib.get_close_matches(str(name), sorted(tracks), n=1)
        raise ValueError(f'в {os.path.basename(sc)} нет трека «{name}»' + (f' (может быть, {near[0]}?)' if near else '') + f'; есть: {", ".join(sorted(tracks))}')
    return L, tracks[name]


def ref(film, lid, name, t):
    L, tr = find(film, lid, name)
    return TrackRef(film, L, tr, t)


def sheet(path, name, out, count=8):
    """frames of the clip with the tracked outline drawn (yellow; red where it was lost) → a picture to look at"""
    from PIL import Image, ImageDraw
    from .media import VideoReader
    tr = _load(sidecar(path, 'track'))['tracks'][name]
    r = VideoReader(path)
    n = len(tr['H'])
    picks = sorted(set(int(round(i)) for i in np.linspace(0, n - 1, count)))
    cols = 4
    k = min(1.0, 480 / r.w)
    tw, th = int(r.w * k), int(r.h * k)
    page = Image.new('RGB', (cols * tw, -(-len(picks) // cols) * (th + 18)), '#111111')
    for j, i in enumerate(picks):
        fr = Image.fromarray(r.at((tr['first'] + i + .5) / r.fps)).resize((tw, th))
        H = np.append(np.array(tr['H'][i], float), 1.0).reshape(3, 3)
        q = [(float(x + .5) * k, float(y + .5) * k) for x, y in _pts(H, tr['quad'])]
        ImageDraw.Draw(fr).line(q + [q[0]], fill='#ff3030' if tr['lost'][i] else '#ffe000', width=2)
        page.paste(fr, ((j % cols) * tw, (j // cols) * (th + 18) + 18))
        ImageDraw.Draw(page).text(((j % cols) * tw + 4, (j // cols) * (th + 18) + 3), f'{(tr["first"] + i) / r.fps:.2f} s  q {tr["q"][i]:.2f}' + ('  LOST' if tr['lost'][i] else ''), fill='#ffd060')
    r.close()
    page.save(out, quality=90)
    return out
