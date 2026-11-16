"""What a 3D scene's bodies reflect and are lit by from all around (kadr/scene3d.py, `environment`): a panorama — a
photograph of everything around a point, 2:1 — or the built-in studio, baked into the layout the core's shader reads
(three.js' PMREM «CubeUV»: the six faces of a cube at eleven degrees of blur, one for each roughness).

    environment: studio                    a dark room with softboxes: one overhead, a key at the left, a cool fill
                                           at the right, strips behind — what a product shot stands in
    environment: room.hdr | pano.jpg       a panorama (Radiance .hdr keeps the lamps' true brightness; a .jpg / .png
                                           is taken as it is). Its middle is straight ahead of a camera looking along −z
    environment: {src: studio, intensity: 1, turn: 30}        turn: degrees about the vertical (what was ahead goes to the right)

A panorama's pixel (u, v) looks at:  yaw = (u − ½) · 360° from −z towards +x,  pitch = (½ − v) · 180° up.
Baking takes a second or two and is kept in ~/.cache/kadr/env by the file's path, date, size and turn."""
import hashlib, math, os

import numpy as np

VERSION = 2
CUBE = 256                                                     # the sharpest face; the atlas is 768 × 1024
EXTRA = (0.125, 0.215, 0.35, 0.446, 0.526, 0.582)              # the blur (radians) of the six lods past the 16 px face
CACHE = os.path.expanduser('~/.cache/kadr/env')


def directions(w, h):
    """where each pixel of a w × h panorama looks → (h, w, 3)"""
    u, v = (np.arange(w) + .5) / w, (np.arange(h) + .5) / h
    yaw, pitch = (u - .5) * 2 * math.pi, (.5 - v) * math.pi
    cp = np.cos(pitch)[:, None]
    return np.stack([np.sin(yaw)[None] * cp, np.repeat(np.sin(pitch)[:, None], w, 1), -np.cos(yaw)[None] * cp], -1)


def lookup(eq, d):
    """the panorama in the directions d (…, 3) → (…, 3): between its four nearest pixels, round the seam"""
    h, w = eq.shape[:2]
    x = (np.arctan2(d[..., 0], -d[..., 2]) / (2 * math.pi) + .5) * w - .5
    y = (.5 - np.arcsin(np.clip(d[..., 1], -1, 1)) / math.pi) * h - .5
    x0, y0 = np.floor(x).astype(np.int64), np.clip(np.floor(y).astype(np.int64), 0, h - 2)
    fx, fy = (x - x0)[..., None].astype(np.float32), np.clip(y - y0, 0, 1)[..., None].astype(np.float32)
    xa, xb = x0 % w, (x0 + 1) % w
    return (eq[y0, xa] * (1 - fx) + eq[y0, xb] * fx) * (1 - fy) + (eq[y0 + 1, xa] * (1 - fx) + eq[y0 + 1, xb] * fx) * fy


def shrink(eq, w):
    """the panorama at a smaller width (each pixel the mean of what it covers)"""
    from PIL import Image
    h = max(2, w // 2)
    if w >= eq.shape[1]:
        return eq
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(eq[..., c]), 'F').resize((w, h), Image.BOX)) for c in range(3)], -1)


def _face_dirs(fs):
    """the direction of every texel of every face at face size fs, borders included → (6, fs, fs, 3), unit length.
    The inverse of the shader's cube_uv: texel centre c ↔ face coordinate (c − 1) / (fs − 2)"""
    c = ((np.arange(fs) + .5) - 1) / (fs - 2) * 2 - 1
    a, b = np.meshgrid(c, c)                                   # a along x of the face, b along y
    o = np.ones_like(a)
    f = [np.stack(v, -1) for v in ((o, b, a), (-a, o, -b), (-a, b, o), (-o, b, -a), (-a, -o, b), (a, b, -o))]
    d = np.stack(f)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _lobe(sigma, rings=10, spokes=16):
    """directions around +z spread as a bell of `sigma` radians in angle (equal weights): each ring holds an equal
    share of the bell's light → (rings · spokes, 3)"""
    th = np.linspace(0, min(3 * sigma, math.pi), 4000)
    pdf = np.exp(-.5 * (th / sigma) ** 2) * np.sin(th)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    tq = np.interp((np.arange(rings) + .5) / rings, cdf, th)
    out = []
    for r, t in enumerate(tq):
        ph = (np.arange(spokes) + (r * 0.618034) % 1) / spokes * 2 * math.pi
        out.append(np.stack([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.full(spokes, np.cos(t))], -1))
    return np.concatenate(out)


def bake(eq, cube=CUBE):
    """a panorama (h, w, 3) float, linear light → the atlas (4 · cube, 3 · max(cube, 112), 4) as half floats (uint16)"""
    eq = np.asarray(eq, np.float32)
    top = int(round(math.log2(cube)))
    W, H = 3 * max(cube, 112), 4 * cube
    atlas = np.zeros((H, W, 4), np.float32)
    atlas[..., 3] = 1
    lods = [(2 ** m, 0.0 if m == top else 1.0 / 2 ** m, 0) for m in range(top, 3, -1)] + [(16, s, k + 1) for k, s in enumerate(EXTRA)]
    for fs, sigma, extra in lods:
        d = _face_dirs(fs)                                     # (6, fs, fs, 3)
        if sigma <= 0:
            col = lookup(shrink(eq, 4 * fs), d)
        else:
            src = shrink(eq, int(min(eq.shape[1], max(64, 2 ** math.ceil(math.log2(4 * math.pi / sigma))))))
            lobe = _lobe(sigma).astype(np.float32)             # (K, 3) about +z
            col = np.zeros((6, fs, fs, 3), np.float32)
            for f in range(6):
                n = d[f].reshape(-1, 3)
                up = np.where(np.abs(n[:, 1:2]) < .99, np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]]))
                t1 = np.cross(up, n)
                t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
                t2 = np.cross(n, t1)
                s = lobe[None, :, 0:1] * t1[:, None] + lobe[None, :, 1:2] * t2[:, None] + lobe[None, :, 2:3] * n[:, None]   # (T, K, 3)
                col[f] = lookup(src, s).mean(1).reshape(fs, fs, 3)
        x0, y0 = 3 * 16 * extra, 4 * (cube - fs)
        for f in range(6):
            atlas[y0 + (f // 3) * fs:y0 + (f // 3 + 1) * fs, x0 + (f % 3) * fs:x0 + (f % 3 + 1) * fs, :3] = col[f]
    return np.clip(atlas, 0, 65000).astype(np.float16).view(np.uint16)


def studio(w=1024):
    """the built-in studio as a panorama: a dark room whose walls carry softboxes"""
    d = directions(w, w // 2)
    p = d / np.abs(d).max(-1, keepdims=True)                   # where the look meets a cube room, −1 … 1
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    out = np.zeros(d.shape, np.float32)
    out += (.05 + .05 * (y + 1) / 2)[..., None] * np.array([1.0, 1.0, 1.05])                # walls: a little light from above
    out[y <= -.999] = np.array([.035, .032, .03])                                             # the floor

    def box(mask, colour, edge):                               # a softbox: bright, falling off softly at its rim
        out[...] = out * (1 - edge[..., None] * mask[..., None]) + np.array(colour, np.float32) * (edge * mask)[..., None]

    def soft(a, lo, hi, s=.06):                                # 1 inside lo … hi, easing to 0 over s
        return np.clip((a - lo) / s, 0, 1) * np.clip((hi - a) / s, 0, 1)
    box(y >= .999, (9.0, 9.0, 9.0), soft(x, -.55, .55) * soft(z, -.45, .65))                   # overhead
    box(x <= -.999, (6.5, 6.0, 5.4), soft(y, -.15, .75) * soft(z, -.5, .55))                   # key, warm, at the left
    box(x >= .999, (1.9, 2.2, 2.8), soft(y, -.3, .6) * soft(z, -.35, .45))                     # fill, cool, at the right
    box(z <= -.999, (3.2, 3.2, 3.4), soft(y, -.1, .8) * (soft(x, -.9, -.62) + soft(x, .62, .9)))   # two strips behind the subject
    box(z >= .999, (1.3, 1.3, 1.3), soft(y, -.4, .7) * soft(x, -.75, .75))                     # a broad soft front, behind the camera
    return out


def read_hdr(path):
    """a Radiance .hdr picture → (h, w, 3) float32, linear light"""
    with open(path, 'rb') as f:
        if not f.readline().startswith(b'#?'):
            raise ValueError(f'{os.path.basename(path)}: не Radiance .hdr')
        while True:
            ln = f.readline()
            if not ln:
                raise ValueError(f'{os.path.basename(path)}: в .hdr нет размеров')
            if ln.strip() == b'':
                break
        dims = f.readline().split()
        if len(dims) != 4 or dims[0] not in (b'-Y', b'+Y') or dims[2] != b'+X':
            raise ValueError(f'{os.path.basename(path)}: такой порядок строк .hdr не читается')
        h, w = int(dims[1]), int(dims[3])
        if not (0 < h <= 16384 and 0 < w <= 32768):
            raise ValueError(f'{os.path.basename(path)}: панорама {w}×{h} — больше, чем читается (до 32768×16384)')
        raw = np.frombuffer(f.read(), np.uint8)
    img = np.zeros((h, w, 4), np.uint8)
    p = 0
    for y in range(h):
        if w < 8 or w > 32767 or raw[p] != 2 or raw[p + 1] != 2 or (int(raw[p + 2]) << 8 | int(raw[p + 3])) != w:
            img.reshape(-1, 4)[y * w:] = raw[p:p + (h - y) * w * 4].reshape(-1, 4)       # not run-length coded: the rest as it is
            break
        p += 4
        for c in range(4):
            x = 0
            while x < w:
                n = int(raw[p]); p += 1
                if n > 128:
                    n -= 128
                    img[y, x:x + n, c] = raw[p]; p += 1
                else:
                    img[y, x:x + n, c] = raw[p:p + n]; p += n
                x += n
    e = img[..., 3].astype(np.int32)
    out = img[..., :3].astype(np.float32) * np.where(e > 0, np.ldexp(1.0, e - 136), 0.0)[..., None].astype(np.float32)
    return out[::-1] if dims[0] == b'+Y' else out


def room(colour, lamps, w=1024, soft=0.0):
    """a room of one colour with glowing balls in it, seen from its middle, as a panorama. lamps: ((r, g, b) — the light
    it gives, linear; (x, y, z) — where its centre is from the middle; its radius). A ball nearer hides one behind it;
    one that reaches the middle itself is not seen (the look starts inside it). soft: how blurred the balls' edges are,
    degrees (a ball is a disc to the eye in the middle; its edge is eased over this angle)"""
    from math import erf
    d = directions(w, w // 2).astype(np.float64)
    out = np.tile(np.array(colour, np.float32), (*d.shape[:2], 1))
    sig = math.radians(max(float(soft), 1e-4))
    ease = np.vectorize(lambda v: .5 * (1 + erf(v)), otypes=[np.float32])
    for c, pos, r in sorted(lamps, key=lambda q: -float(np.linalg.norm(q[1]))):      # the farthest first, the nearer over it
        pos = np.array(pos, float)
        dist = float(np.linalg.norm(pos))
        if dist <= r:
            continue
        al = math.asin(r / dist)                               # the disc's radius to the eye; eased evenly both ways its edge would
        al -= sig * sig / (2 * math.tan(al)) if soft else 0.0  # gain light (there is more sky outside it than inside): drawn in by that
        gap = al - np.arccos(np.clip(d @ (pos / dist), -1, 1)) # how far inside the disc's edge, radians
        near = np.abs(gap) < 4 * sig
        cov = (gap > 0).astype(np.float32)
        cov[near] = ease(gap[near] / (sig * math.sqrt(2)))
        out = out * (1 - cov[..., None]) + np.array(c, np.float32) * cov[..., None]
    return out.astype(np.float32)


def panorama(src, turn=0.0):
    """the panorama of an `environment` → (h, w, 3) float32 linear, turned about the vertical"""
    if src == 'studio':
        eq = studio()
    elif src.lower().endswith('.hdr'):
        eq = read_hdr(src)
    else:
        from PIL import Image
        a = np.asarray(Image.open(src).convert('RGB'), np.float32) / 255
        eq = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)
    if eq.shape[1] > 2048:
        eq = shrink(eq, 2048)
    if turn:
        eq = np.roll(eq, int(round(turn / 360 * eq.shape[1])), 1)      # what was ahead goes to the right
    return eq


def atlas(src, turn=0.0, made=None):
    """the baked environment of `src` (studio, a file, or 'room' with made = (colour, lamps)), from the cache when it is there"""
    st = os.stat(src) if src not in ('studio', 'room') else None
    key = hashlib.sha1(f'{src}|{st.st_mtime_ns if st else 0}|{st.st_size if st else 0}|{turn}|{CUBE}|{VERSION}|{made!r}'.encode()).hexdigest()[:20]
    p = os.path.join(CACHE, key + '.npy')
    if os.path.exists(p):
        return np.load(p)
    if src == 'room':
        eq = room(made[0], made[1], soft=made[2] if len(made) > 2 else 0.0)
        eq = np.roll(eq, int(round(turn / 360 * eq.shape[1])), 1) if turn else eq
    else:
        eq = panorama(src, turn)
    a = bake(eq)
    os.makedirs(CACHE, exist_ok=True)
    tmp = f'{p}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        np.save(f, a)
    os.replace(tmp, p)
    return a
