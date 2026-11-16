"""The `scene3d` layer (AE.md, M6): a 3D scene drawn by the core — models, physically based materials, lights, one
soft shadow, an environment, depth of field and motion blur from several samples a frame, glow and a filmic curve.

    type: scene3d
    src: scene.yaml        the scene in Kadr's own words (what `scene:` below takes), kept in a file of its own: a room, a
                           character — written once, used by many films; its pictures and models are looked for beside it
    src: scene.json        or a scene exported from another renderer: nodes (a tree of models, groups and sprites),
                           materials, pictures, lights, the environment, the look — with scene.bin (the models),
                           env.bin, tex_*.png beside it
    clip: clip.bin         what moves, sampled at a fixed rate: nodes (place, turn, size, visibility), materials
                           (colour, opacity), the camera (place, target, lens); `cuts` in the scene — the moments the
                           camera jumps (a frame never mixes two shots)
    size: [w, h]           the picture (default: the frame)
    samples: 16            samples a frame: each has its own moment in the shutter, point on the lens, sub-pixel shift
                           and place of the lamp — motion blur, depth of field, smooth edges and soft shadows in one
    time: …                the scene's own time (default: the film's), as a comp's `time`
    nodes: {name: {position, rotation, scale, visible}}       a file's named nodes (in Kadr's words: their `id`) moved
                           from the document, by keys and formulas (a puppet: pose("hero", "smile"), amp("voice")) —
                           over the clip, or with no clip
    camera: {position, look_at, fov, lens, focus}             the camera set in the document; cuts: [times it jumps]
    lights: {name: {intensity}}; materials: {name: {opacity, glow, shade}}      a file's named lamps and materials
                           turned from the document (day and night in a room); shade: its colour times this
    rig: file.yaml         a file with `nodes` (and `camera`, `lights`, `materials`) to start from: a character's
                           strings, written once

A scene can also be written in the document itself, from simple bodies — no file, no other program:

    scene:
      camera: {position: [0, 40, 160], look_at: [0, 20, 0], fov: 35, lens: 0.8}     # lens: the aperture's radius (0: all sharp)
      lights:
        - {type: spot, position: […], target: […], color: "#ffd2a6", intensity: 60000, angle: 34, penumbra: .8, shadow: true, soft: 10}
        - {type: point, position: […], color: …, intensity: …}        # intensity of spot / point: candela, units are the scene's
        - {type: sun, position: […], target: […], intensity: 2}
        - {type: sky, color: "#a7acbf", ground: "#3a2415", intensity: .6}
        - {type: panel, position: […], look_at: […], size: [36, 23], intensity: 2.6}      # a glowing rectangle
      look: {exposure: 1, glow: .3, tint: [1, 1, 1], filmic: .82, vignette: 0, grain: 0, raw: false}
      environment: studio                                             # what bodies reflect: studio, a panorama file,
                                                                      # {src, intensity, turn}, or a room described here:
                                                                      # {room: "#0d0b10", lamps: [{color, intensity, position,
                                                                      # radius}, …], soft: 1.7 (degrees), intensity} — glowing
                                                                      # balls round the scene's middle (kadr/envmap.py)
      nodes:
        - {id: floor, shape: box, size: [200, 4, 200], position: [0, -2, 0], color: "#888888", roughness: .8}
        - {id: ball, shape: sphere, radius: 12, position: {0: [0, 12, 0], 2: [40, 12, 0]}, color: "#ee3124", clearcoat: .5}
        - {id: arm, position: […], rotation: [0, 30, 0], children: [{shape: rbox, size: [4, 20, 4], radius: 1, …}]}

shapes: box, rbox (rounded, `radius`), sphere, dome (its upper half), cylinder (`radius` — or [top, bottom]: a cone's
slice — and `height`), lathe (a body of revolution: `profile: [[radius, height], …]` turned about the vertical — a mug, a
vase, a lampshade), torus (a ring in its own xy plane: `radius`, `tube`), disc (`radius`, faces +z), plane (`size: [w, h]`,
faces +z; `segments: [across, up]` and `relief: "sin(x * .3) * .7"` — its height off the plane as a formula of x and y —
bend it: a flag, a cloth); `segments` on a round body: how many pieces round (sphere: [round, down]; torus: [round, across]);
glow — a soft spot of light that faces the camera and adds to what is behind it: {shape: glow, size: 30, color, glow: .3};
text — lettering with thickness: {shape: text, text: "KADR", font: Arial, weight: 900, size: 40 (an em, scene units),
depth: 8, bevel: 1 (the edge rounded off by this much), align, tracking, line_height}, centred on its own box;
extrude — any outline: {shape: extrude, d: "M … Z" (an SVG path) | src: logo.svg / logo.png (the file's shape, wherever it
is not transparent), height: 30, depth: 6, bevel: .6};
model — a glTF file (.glb / .gltf, kadr/gltf.py): {shape: model, src: chair.glb, height: 40 (scaled to this height),
origin: base | center | model (what sits at the node's position: the middle of its underside by default), animation:
its name or number (the first by default; false: none), loop: true, speed: 1}; color / roughness / metal given on the
node repaint the whole model. A node has
position, rotation (degrees about x, y, z), scale, visible; a material: color, roughness, metal, clearcoat, emissive,
opacity, `env` (how much of the surroundings it takes: 1 — all, 0 — none), and `map: picture.png` — a picture laid over the
body (upright on a plane and on each side of a box, wrapped round a sphere and a cylinder; `tile: [3, 2]` repeats it; with
`unlit: true` it shows as it is — a screen, a poster; `alpha: true` — its own transparency counts: a sticker). A light has an
`id` too (a film turns it by that name: `lights: {lamp: {intensity: …}}`) and `shadow: true` or {near, far}. Positions,
rotations, the camera and the lights take keys and formulas like any property.
figure — a block character from a few words, with his strings (poses, a face, a mouth that follows a voice):
{shape: figure, id: bob, position: where they stand, height: 54, sits: true, top: "#2b6", pants, shoes, skin, hair, hat, …}
— all of it in kadr/figure.py (`kadr catalog figure`).
`accept: [[mug, laptop], [fly]]` beside `nodes` — pairs of bodies or things (top nodes) that are meant to run into each
other, and bodies meant to hang in the air: the scene's check (`kadr scene`, kadr/scenecheck.py) says nothing of them.

The drawing itself is core/src/scene3d.rs (Rust + wgpu); this file decides what is where at every sample. Both
renderers call the same core, so they agree to the bit. The lighting follows three.js r185 (MeshStandard / Physical),
the renderer the first scene came from, checked frame against frame."""
import functools, json, math, os, threading
from struct import error as struct_error

import numpy as np
from PIL import Image

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_tls = threading.local()


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """a camera's world matrix: it looks down its −z at the target"""
    z = np.asarray(eye, float) - np.asarray(target, float)
    z /= np.linalg.norm(z) or 1.0
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross((0.0, 0.0, 1.0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def perspective(fov, aspect, near, far):
    """OpenGL's projection (z −1…1), as three.js builds it; fov: vertical, degrees"""
    top = near * math.tan(math.radians(fov) / 2)
    r = top * aspect
    return np.array([[near / r, 0, 0, 0], [0, near / top, 0, 0], [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]], float)


ZFIX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, .5, .5], [0, 0, 0, 1]], float)   # z −1…1 → 0…1, the GPU's depth


def trs(p, q, s):
    """(…, 3), (…, 4 xyzw), (…, 3) → (…, 4, 4)"""
    p, q, s = np.asarray(p, float), np.asarray(q, float), np.asarray(s, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.zeros(p.shape[:-1] + (4, 4))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - z * w); m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w); m[..., 2, 1] = 2 * (y * z + x * w); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    m[..., :3, :3] *= s[..., None, :]
    m[..., :3, 3] = p
    m[..., 3, 3] = 1
    return m


def mips(a, srgb):
    """the smaller copies of a picture (h, w, 4) uint8, averaged in linear light when it is sRGB"""
    f = a.astype(np.float32) / 255
    if srgb:
        f[..., :3] = np.where(f[..., :3] <= 0.04045, f[..., :3] / 12.92, ((f[..., :3] + 0.055) / 1.055) ** 2.4)
    out = [a]
    while max(f.shape[0], f.shape[1]) > 1:
        h, w = max(1, f.shape[0] // 2), max(1, f.shape[1] // 2)
        f = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(f[..., c]), 'F').resize((w, h), Image.BOX)) for c in range(4)], -1)
        g = f.copy()
        if srgb:
            g[..., :3] = np.where(g[..., :3] <= 0.0031308, g[..., :3] * 12.92, 1.055 * np.maximum(g[..., :3], 0) ** (1 / 2.4) - 0.055)
        out.append(np.ascontiguousarray(np.clip(np.round(g * 255), 0, 255).astype(np.uint8)))
    return out


def srgb(c):
    """a colour of the document (0..255, sRGB) → linear light 0..1"""
    c = np.asarray(c, float)[:3] / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def euler(deg):
    """degrees about x, y, z (applied in that order, as three.js' default) → a quaternion xyzw"""
    x, y, z = (math.radians(v) / 2 for v in deg)
    c1, c2, c3, s1, s2, s3 = math.cos(x), math.cos(y), math.cos(z), math.sin(x), math.sin(y), math.sin(z)
    return [s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3]


def _grid(n):
    """two triangles for every cell of an (n + 1) × (n + 1) grid of vertices"""
    i = np.arange(n * n)
    a = i // n * (n + 1) + i % n
    return np.stack([a, a + 1, a + n + 1, a + 1, a + n + 2, a + n + 1], -1).ravel()     # counter-clockwise from outside


def box_mesh(w, h, d, radius=0.0, seg=1):
    """a box centred at the origin → (vertices (n, 8), indices). With a radius its edges and corners are rounded: each
    face is its flat middle and, on every side of it, `seg` equal steps round the edge (6 unless more are asked for) —
    however small the radius is beside the box. (Until 16.11.2026 a face was an even grid pushed out from the box
    shrunk by the radius: a thin radius on a long box fell between two grid lines and came out as one wide chamfer.)"""
    half = np.array([w, h, d], float) / 2
    radius = min(radius, half.min() - 1e-4) if radius > 0 else 0.0
    if radius > 0:
        n = max(6, int(seg))
        t = np.tan(np.linspace(0, math.pi / 4, n + 1))         # where on a cube's face the look from its middle turns by equal steps
        u = np.concatenate([-t[::-1], t])                      # the two zeros are the flat middle's two ends
        sg = np.concatenate([-np.ones(n + 1), np.ones(n + 1)])
    else:
        n = max(1, int(seg))
        u = np.linspace(-1, 1, n + 1)
        sg = np.sign(u)
    k = len(u)
    a, b = np.meshgrid(u, u)
    sa, sb = np.meshgrid(sg, sg)
    cell = np.arange((k - 1) * (k - 1))
    q = cell // (k - 1) * k + cell % (k - 1)
    quad = np.stack([q, q + 1, q + k, q + 1, q + k + 1, q + k], -1).ravel()
    vs, ix = [], []
    for axis in range(3):
        for sign in (1, -1):
            o = [(1, 2), (2, 0), (0, 1)][axis]
            p, sgn = np.zeros((k, k, 3)), np.zeros((k, k, 3))
            p[..., axis] = sgn[..., axis] = sign
            p[..., o[0]], sgn[..., o[0]] = (a, sa) if sign > 0 else (-a, -sa)
            p[..., o[1]], sgn[..., o[1]] = b, sb
            p, sgn = p.reshape(-1, 3), sgn.reshape(-1, 3)
            if radius > 0:
                nrm = p / np.linalg.norm(p, axis=1, keepdims=True)
                pos = sgn * (half - radius) + nrm * radius
            else:
                nrm = np.tile(np.eye(3)[axis] * sign, (len(p), 1))
                pos = p * half
            fa, fb = pos[:, o[0]] * (1 if sign > 0 else -1), pos[:, o[1]]      # a picture lies on the face by where the point is
            uv = np.stack([fa / (2 * half[o[0]]) + .5, fb / (2 * half[o[1]]) + .5], -1)
            ix.append(quad + sum(len(v) for v in vs))
            vs.append(np.concatenate([pos, nrm, uv], 1))
    return np.concatenate(vs).astype(np.float32), np.concatenate(ix).astype(np.uint32)


def sphere_mesh(r, seg=48, rows=None, part=1.0):
    """part: how much of it from the top down (.5 — a dome, open below)"""
    rows = int(rows or seg // 2)
    th, ph = np.meshgrid(np.linspace(0, 2 * math.pi, seg + 1), np.linspace(0, math.pi * part, rows + 1))
    n = np.stack([-np.sin(th) * np.sin(ph), np.cos(ph), -np.cos(th) * np.sin(ph)], -1).reshape(-1, 3)   # u = ½ (a picture's middle) faces +z
    uv = np.stack([th.ravel() / (2 * math.pi), 1 - ph.ravel() / math.pi], -1)
    cols = seg
    i = np.arange(rows * cols)
    a = i // cols * (cols + 1) + i % cols
    ix = np.stack([a, a + cols + 1, a + 1, a + 1, a + cols + 1, a + cols + 2], -1).ravel()
    return np.concatenate([n * r, n, uv], 1).astype(np.float32), ix.astype(np.uint32)


def cylinder_mesh(r, h, seg=48):
    """r: one radius, or (top, bottom) — a cone's slice"""
    rt, rb = (float(r[0]), float(r[1])) if np.ndim(r) else (float(r), float(r))
    th = np.linspace(0, 2 * math.pi, seg + 1)
    c, s = -np.sin(th), -np.cos(th)                            # u = ½ (a picture's middle) faces +z
    k = (rb - rt) / h                                          # the wall leans: its normal looks up or down by this much
    ln = math.sqrt(1 + k * k)
    side = np.concatenate([np.stack([c * r_, np.full_like(c, y), s * r_, c / ln, np.full_like(c, k / ln), s / ln, th / (2 * math.pi), np.full_like(c, v)], -1)
                           for y, v, r_ in ((h / 2, 1, rt), (-h / 2, 0, rb))])
    i = np.arange(seg)
    ix = [np.stack([i, i + seg + 1, i + 1, i + 1, i + seg + 1, i + seg + 2], -1).ravel()]
    vs = [side]
    for y, ny, r_ in ((h / 2, 1.0, rt), (-h / 2, -1.0, rb)):
        if r_ <= 0:
            continue
        base = sum(len(v) for v in vs)
        cap = np.stack([c * r_, np.full_like(c, y), s * r_, np.zeros_like(c), np.full_like(c, ny), np.zeros_like(c), c / 2 + .5, s / 2 + .5], -1)
        vs.append(np.concatenate([[[0, y, 0, 0, ny, 0, .5, .5]], cap]))
        tri = np.stack([np.full(seg, base), base + 1 + i, base + 2 + i], -1) if ny > 0 else np.stack([np.full(seg, base), base + 2 + i, base + 1 + i], -1)
        ix.append(tri.ravel())
    return np.concatenate(vs).astype(np.float32), np.concatenate(ix).astype(np.uint32)


def lathe_mesh(profile, seg=48):
    """a body of revolution: the profile [(radius, height), …] turned about the vertical. Where the profile bends the
    surface is smooth (a point's normal is the mean of its two segments'), as on a potter's wheel"""
    P = np.asarray(profile, float).reshape(-1, 2)
    if len(P) < 2:
        raise ValueError('profile: нужны хотя бы две точки [радиус, высота]')
    d = np.diff(P, axis=0)
    sn = np.stack([d[:, 1], -d[:, 0]], -1)                     # each segment's normal in the profile's plane (outwards, up)
    nn = np.concatenate([sn[:1], sn[:-1] + sn[1:], sn[-1:]])
    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
    ph = np.linspace(0, 2 * math.pi, seg + 1)
    sp, cp = np.sin(ph)[:, None], np.cos(ph)[:, None]
    n = len(P)
    v = np.stack([P[None, :, 0] * sp, np.repeat(P[None, :, 1], seg + 1, 0), P[None, :, 0] * cp, nn[None, :, 0] * sp, np.repeat(nn[None, :, 1], seg + 1, 0), nn[None, :, 0] * cp,
                  np.repeat(ph[:, None] / (2 * math.pi), n, 1), np.repeat(np.linspace(0, 1, n)[None], seg + 1, 0)], -1).reshape(-1, 8)
    i, j = np.meshgrid(np.arange(seg), np.arange(n - 1), indexing='ij')
    a = (j + i * n).ravel()
    b, c, e = a + n, a + n + 1, a + 1
    return v.astype(np.float32), np.stack([a, b, e, c, e, b], -1).ravel().astype(np.uint32)


def torus_mesh(R, tube, seg=32, around=14):
    """a ring lying in the node's own xy plane (its hole looks along z): R — the ring's radius, tube — its thickness' radius"""
    u, w = np.meshgrid(np.linspace(0, 2 * math.pi, seg + 1), np.linspace(0, 2 * math.pi, around + 1))
    n = np.stack([np.cos(w) * np.cos(u), np.cos(w) * np.sin(u), np.sin(w)], -1)
    p = np.stack([R * np.cos(u), R * np.sin(u), np.zeros_like(u)], -1) + n * tube
    v = np.concatenate([p, n, (u / (2 * math.pi))[..., None], (w / (2 * math.pi))[..., None]], -1).reshape(-1, 8)
    j, i = np.meshgrid(np.arange(1, around + 1), np.arange(1, seg + 1), indexing='ij')
    a, b = ((seg + 1) * j + i - 1).ravel(), ((seg + 1) * (j - 1) + i - 1).ravel()
    c, e = b + 1, a + 1
    return v.astype(np.float32), np.stack([a, b, e, b, c, e], -1).ravel().astype(np.uint32)


def disc_mesh(r, seg=32):
    """a flat round plate facing +z"""
    th = np.linspace(0, 2 * math.pi, seg + 1)
    rim = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th), np.zeros_like(th), np.zeros_like(th), np.ones_like(th), np.cos(th) / 2 + .5, np.sin(th) / 2 + .5], -1)
    i = np.arange(1, seg + 1)
    return np.concatenate([[[0, 0, 0, 0, 0, 1, .5, .5]], rim]).astype(np.float32), np.stack([i, i + 1, np.zeros_like(i)], -1).ravel().astype(np.uint32)


def _safe_formula(src):
    from . import safe
    return safe.compile_formula(src, 'relief')


RELIEF = {k: getattr(np, k) for k in ('sin', 'cos', 'tan', 'sqrt', 'exp', 'abs', 'minimum', 'maximum', 'hypot')} | {'pi': math.pi}


def plane_mesh(w, h, seg=None, relief=None):
    """a sheet facing +z. seg: (across, up) — how many pieces it is cut into; relief: its height off the plane as a formula
    of x and y (the sheet's own units, 0 in its middle) — a flag's folds, a wavy cloth; it needs seg to bend along"""
    if not seg and relief is None:
        v = np.array([[-w / 2, -h / 2, 0, 0, 0, 1, 0, 0], [w / 2, -h / 2, 0, 0, 0, 1, 1, 0], [w / 2, h / 2, 0, 0, 0, 1, 1, 1], [-w / 2, h / 2, 0, 0, 0, 1, 0, 1]], np.float32)
        return v, np.array([0, 1, 2, 0, 2, 3], np.uint32)
    nx, ny = (int(seg[0]), int(seg[1])) if seg else (32, 32)
    x, y = np.meshgrid(np.linspace(-w / 2, w / 2, nx + 1), np.linspace(h / 2, -h / 2, ny + 1))
    z = np.zeros_like(x)
    if relief is not None:
        try:
            z = np.broadcast_to(np.asarray(eval(_safe_formula(str(relief).lstrip('=')), {'__builtins__': {}, **RELIEF, 'x': x, 'y': y}), float), x.shape)
        except Exception as e:
            raise ValueError(f'relief: «{relief}» не считается ({e}); в формуле x, y, sin, cos, sqrt, abs, exp, pi')
    p = np.stack([x, y, z], -1).reshape(-1, 3)
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    a = (ix + (nx + 1) * iy).ravel()
    b, c, e = a + nx + 1, a + nx + 2, a + 1
    tri = np.stack([a, b, e, b, c, e], -1).reshape(-1, 3)
    fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])      # a face's normal, as long as the face is large
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    uv = np.stack([(x / w + .5).ravel(), (y / h + .5).ravel()], -1)
    return np.concatenate([p, n, uv], 1).astype(np.float32), tri.ravel().astype(np.uint32)


def soft_dot(n=128):
    """a glow's picture: white, opaque in the middle, gone at the rim (the same falloff the first scene's lamps had)"""
    c = (np.arange(n) + .5 - n / 2) / (n / 2)
    r = np.hypot(c[None], c[:, None])
    a = np.interp(r, [0, .55, 1], [1, .75, 0])
    return np.dstack([np.full((n, n, 3), 255.0), a * 255]).round().astype(np.uint8)


NODE = {'id', 'shape', 'size', 'radius', 'height', 'position', 'rotation', 'scale', 'visible', 'children', 'color', 'roughness', 'metal', 'clearcoat', 'clearcoat_roughness',
        'emissive', 'glow', 'opacity', 'cast', 'receive', 'unlit', 'sides', 'map', 'tile', 'alpha', 'env',
        'segments', 'profile', 'tube', 'relief',
        'text', 'font', 'weight', 'depth', 'bevel', 'bevel_steps', 'd', 'align', 'tracking', 'line_height',
        'src', 'origin', 'animation', 'loop', 'speed'}
LIGHT = {'id', 'type', 'position', 'target', 'look_at', 'color', 'ground', 'intensity', 'angle', 'penumbra', 'distance', 'decay', 'shadow', 'soft', 'size'}
CAMERA = {'position', 'look_at', 'fov', 'lens', 'focus', 'near', 'far'}
LOOK = {'exposure', 'glow', 'glow_from', 'tint', 'filmic', 'vignette', 'grain', 'raw', 'shutter'}
SHAPES = ('box', 'rbox', 'sphere', 'dome', 'cylinder', 'lathe', 'torus', 'disc', 'plane', 'glow', 'text', 'extrude', 'model')
KINDS = {'spot': 'SpotLight', 'point': 'PointLight', 'sun': 'DirectionalLight', 'sky': 'HemisphereLight', 'panel': 'RectAreaLight'}


def parse(F, d, where, base=None):
    """a scene written in the document → what `Scene` reads (the same structure an exported scene has) and the
    properties that move; problems go to the film's issues. base: the folder of the scene's own file (its pictures and
    models are looked for beside it), when it is not written in the film itself"""
    from .doc import color
    resolve = F.resolve if base is None else (lambda q: F.resolve(os.path.expanduser(str(q)) if os.path.isabs(os.path.expanduser(str(q))) else os.path.join(base, str(q))))
    if not isinstance(d, dict):
        F.issue('error', 'scene: нужен словарь с nodes, lights, camera', where)
        d = {}
    for k in d:
        if k not in ('camera', 'lights', 'nodes', 'look', 'environment', 'accept'):
            F.issue('error', f'scene: нет поля «{k}»', where, k, {'camera', 'lights', 'nodes', 'look', 'environment', 'accept'})
    def col(v, dflt):
        try:
            return list(srgb(color(v if v is not None else dflt)))
        except (ValueError, TypeError) as e:
            F.issue('error', str(e), where)
            return [1.0, 1.0, 1.0]
    env = None
    ev = d.get('environment')
    if ev:                                                     # what the bodies reflect and are lit by from all around (kadr/envmap.py)
        ev = {'src': ev} if isinstance(ev, str) else ev
        if not isinstance(ev, dict) or any(k not in ('src', 'intensity', 'turn', 'room', 'lamps', 'soft') for k in ev):
            F.issue('error', 'scene.environment: studio, файл панорамы (.hdr, .jpg, .png), {src, intensity, turn} или {room, lamps, intensity}', where)
            ev = {}
        if 'lamps' in ev or 'room' in ev:                      # a dark room with glowing balls in it, described here
            lamps = []
            for q in ev.get('lamps') or []:
                try:
                    lamps.append((tuple(float(v) * float(q.get('intensity', 1)) for v in col(q.get('color'), '#ffffff')), tuple(float(v) for v in q['position']), float(q.get('radius', 1))))
                    if len(lamps[-1][1]) != 3 or set(q) - {'color', 'intensity', 'position', 'radius'}:
                        raise ValueError
                except (TypeError, ValueError, KeyError, AttributeError):
                    F.issue('error', 'scene.environment.lamps: каждая — {color, intensity, position: [x, y, z], radius}', where)
            env = dict(w=768, h=1024, intensity=float(ev.get('intensity', 1)), src='room', turn=float(ev.get('turn', 0)), room=(tuple(col(ev.get('room'), '#000000')), tuple(lamps), float(ev.get('soft', 0))))
        else:
            src = str(ev.get('src', 'studio'))
            if src != 'studio':
                src = resolve(src)
                if not os.path.isfile(src):
                    F.issue('error', f"scene.environment: нет файла {ev.get('src')}", where)
                    src = 'studio'
            env = dict(w=768, h=1024, intensity=float(ev.get('intensity', 1)), src=src, turn=float(ev.get('turn', 0)))

    def keys(x, allowed, what):
        for k in x:
            if k not in allowed:
                F.issue('error', f'{what}: нет поля «{k}»', where, k, allowed)

    nodes, geos, mats, props = [], [], [], []
    meshes, texs, figures = [], [], []

    def picture(n):
        """the node's `map` → its number among the scene's pictures (one file is read once), or −1"""
        if not n.get('map'):
            return -1
        path = resolve(str(n['map']))
        if not os.path.isfile(path):
            F.issue('error', f"scene: нет файла картинки {n['map']}", where)
            return -1
        tile = n.get('tile')
        try:
            rp = [float(tile[0]), float(tile[1])] if tile is not None else [1.0, 1.0]
        except (TypeError, ValueError, IndexError):
            F.issue('error', 'scene: tile — [сколько раз по ширине, по высоте]', where)
            rp = [1.0, 1.0]
        key = (path, tuple(rp))
        for t in texs:
            if t['key'] == key:
                return t['id']
        texs.append(dict(id=len(texs), key=key, path=path, srgb=True, repeat=tile is not None, rep=rp, flipY=True))
        return texs[-1]['id']

    def model(n, at, w0):
        from . import gltf
        if not n.get('src'):
            raise ValueError('нужен src: файл .glb или .gltf')
        path = resolve(str(n['src']))
        if not os.path.isfile(path):
            raise ValueError('нет такого файла')
        M = gltf.cached(path, getattr(F, 'may_read', None))
        lo, hi = gltf.bounds(M)
        k = float(n['height']) / max(1e-9, hi[1] - lo[1]) if n.get('height') is not None else 1.0
        origin = n.get('origin', 'base')
        if origin not in ('base', 'center', 'model'):
            raise ValueError('origin: base (середина низа), center (середина) или model (как в файле)')
        off = {'base': [-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2], 'center': list(-(lo + hi) / 2), 'model': [0, 0, 0]}[origin]
        fit = len(nodes)                                       # a node that brings the model to size and sets its origin
        nodes.append(dict(id=fit, name=f"{n.get('id', at)}.fit", parent=at, kind='Group', p=[v * k for v in off], q=[0, 0, 0, 1], s=[k, k, k], visible=True, cast=False, receive=False, geo=-1, mat=-1))
        tex0 = len(texs)
        for t in M['texs']:
            texs.append(dict(id=len(texs), key=('model', path, len(texs)), data=t['data'], srgb=t['srgb'], repeat=t['repeat'], rep=[1.0, 1.0], flipY=False))
        tx = lambda i: (tex0 + i) if i >= 0 else -1
        mat0 = len(mats)
        for m in M['mats']:
            c = col(n['color'], '#ffffff') if n.get('color') is not None else m['color']
            mats.append(dict(id=len(mats), kind='basic' if (m['unlit'] or n.get('unlit')) else 'standard', color=list(c), roughness=float(n.get('roughness', m['roughness'])),
                             metalness=float(n.get('metal', m['metal'])), emissive=m['emissive'], emissiveIntensity=1.0, opacity=m['opacity'] * float(n.get('opacity', 1)),
                             transparent=m['transparent'] or float(n.get('opacity', 1)) < 1, side=2 if m['two'] else 0, map=tx(m['map']), map_mr=tx(m['map_mr']), map_n=tx(m['map_n']), map_e=tx(m['map_e']), ao=m['ao'], envMapIntensity=1,
                             clearcoat=float(n.get('clearcoat', m['clearcoat'])), clearcoatRoughness=float(n.get('clearcoat_roughness', m['clearcoat_roughness'])), additive=False,
                             depthWrite=not (m['transparent'] or float(n.get('opacity', 1)) < 1), depthTest=True))
        geo0 = len(meshes)
        meshes.extend(M['meshes'])
        ours = {}
        order = [j for j in range(len(M['nodes'])) if M['keep'][j]]
        done = set()

        def place(j):                                          # parents first
            if j in done:
                return
            g = M['nodes'][j]
            if g['parent'] >= 0 and M['keep'][g['parent']]:
                place(g['parent'])
            done.add(j)
            ours[j] = len(nodes)
            nodes.append(dict(id=len(nodes), name=g['name'], parent=ours.get(g['parent'], fit), kind='Group', p=g['p'], q=g['q'], s=g['s'], visible=True, cast=False, receive=False, geo=-1, mat=-1))
            for mesh, mat in g['draws']:
                nodes.append(dict(id=len(nodes), name=g['name'], parent=ours[j], kind='Mesh', p=[0, 0, 0], q=[0, 0, 0, 1], s=[1, 1, 1], visible=True,
                                  cast=bool(n.get('cast', True)), receive=bool(n.get('receive', True)), geo=geo0 + mesh, mat=mat0 + mat))
        for j in order:
            place(j)
        which = n.get('animation', 0)
        if which is not False and which is not None and M['anims']:
            names = [a['name'] for a in M['anims']]
            if isinstance(which, str) and which not in names:
                raise ValueError(f'нет анимации «{which}»; есть: {", ".join(names)}')
            a = M['anims'][names.index(which) if isinstance(which, str) else int(which) % len(M['anims'])]
            for node, what, times, vals, step in a['tracks']:
                if node in ours:
                    props.append((ours[node], {'translation': 'position', 'rotation': 'quat', 'scale': 'scale'}[what],
                                  gltf.Track(times, vals, turn=what == 'rotation', step=step, loop=bool(n.get('loop', True)), speed=float(n.get('speed', 1)), length=a['length'])))
        if M['left_out']:
            F.issue('warn', f"модель {n['src']}: не читается — " + ', '.join(M['left_out']), w0)

    def walk(items, parent):
        for n in items or []:
            if not isinstance(n, dict):
                F.issue('error', 'scene.nodes: каждый узел — словарь', where)
                continue
            if n.get('shape') == 'figure':                     # a block figure: a few words → its parts and its strings (kadr/figure.py)
                from . import figure
                tree, fig = figure.build(n, lambda level, msg, key=None, choices=None: F.issue(level, msg, where, key, choices))
                if any(q['id'] == fig['id'] for q in figures):
                    F.issue('error', f"scene: две фигурки с одним id «{fig['id']}»", where)
                figures.append(fig)
                walk([tree], parent)
                continue
            keys(n, NODE, 'узел сцены')
            i = len(nodes)
            w0 = f"{where}.scene.{n.get('id', i)}"
            node = dict(id=i, name=str(n.get('id', i)), parent=parent, kind='Group', p=[0, 0, 0], q=[0, 0, 0, 1], s=[1, 1, 1], visible=True, cast=False, receive=False, geo=-1, mat=-1)
            nodes.append(node)
            for key, dflt, dim in (('position', [0, 0, 0], 3), ('rotation', [0, 0, 0], 3), ('scale', 1, None), ('visible', 1, None)):
                if key in n:
                    props.append((i, key, F.prop(n[key] if not isinstance(n[key], bool) else float(n[key]), f'{w0}.{key}', dim)))
            sh = n.get('shape')
            if sh == 'model':                                  # a model made elsewhere hangs on this node: kadr/gltf.py
                try:
                    model(n, i, w0)
                except (ValueError, OSError, KeyError, IndexError, struct_error) as e:
                    F.issue('error', f'scene: модель {n.get("src")}: {e}', where)
                walk(n.get('children'), i)
                continue
            if sh == 'glow':                                   # a soft spot of light that always faces the camera and adds to what is behind it
                c = col(n.get('color'), '#ffffff')
                k = float(n.get('glow', 1))
                for t in texs:
                    if t.get('key') == 'soft dot':
                        break
                else:
                    texs.append(dict(id=len(texs), key='soft dot', data=soft_dot(), srgb=True, repeat=False, rep=[1.0, 1.0], flipY=True))
                    t = texs[-1]
                mats.append(dict(id=len(mats), name=node['name'], kind='sprite', color=[v * k for v in c], roughness=1.0, metalness=0.0, emissive=[0, 0, 0], emissiveIntensity=0.0, opacity=1.0,
                                 transparent=True, side=0, map=t['id'], envMapIntensity=1, clearcoat=0.0, clearcoatRoughness=0.0, additive=True, depthWrite=False, depthTest=True))
                try:
                    size = float(n.get('size', 10))
                except (TypeError, ValueError):
                    F.issue('error', 'scene: у glow size — одно число (поперечник пятна)', where)
                    size = 10.0
                node.update(kind='Sprite', mat=len(mats) - 1, s=[size, size, size])
                walk(n.get('children'), i)
                continue
            if sh is not None:
                if sh not in SHAPES:
                    F.issue('error', f'scene: shape «{sh}»', where, sh, SHAPES)
                    sh = 'box'
                sz = n.get('size', [10, 10, 10])
                sg = n.get('segments')
                try:
                    sg = None if sg is None else ([int(v) for v in sg] if isinstance(sg, (list, tuple)) else [int(sg)])
                    if sg is not None and min(sg) < 3 and sh != 'plane':
                        raise ValueError
                    if sh in ('box', 'rbox'):
                        r_ = float(n.get('radius', 1 if sh == 'rbox' else 0)) if sh == 'rbox' else 0.0
                        m = box_mesh(*[float(v) for v in sz], radius=r_)
                        node['solid'] = ('box', [float(v) / 2 for v in sz], min(r_, min(float(v) for v in sz) / 2))
                    elif sh in ('sphere', 'dome'):
                        m = sphere_mesh(float(n.get('radius', 5)), *(sg or [48])[:2], **({'part': .5} if sh == 'dome' else {}))
                        rr = float(n.get('radius', 5))
                        node['solid'] = ('ball', rr) if sh == 'sphere' else ('shell', [[rr * math.sin(a), rr * math.cos(a)] for a in np.linspace(0, math.pi / 2, 13)])
                    elif sh == 'cylinder':
                        r_ = n.get('radius', 5)
                        r_ = [float(v) for v in r_] if isinstance(r_, (list, tuple)) else [float(r_)] * 2
                        hh = float(n.get('height', 10))
                        m = cylinder_mesh(r_, hh, (sg or [48])[0])
                        node['solid'] = ('lathe', [[0, -hh / 2], [r_[1], -hh / 2], [r_[0], hh / 2], [0, hh / 2]])
                    elif sh == 'lathe':
                        m = lathe_mesh(n.get('profile') or [], (sg or [48])[0])
                        pf = [[float(a), float(b)] for a, b in n.get('profile')]
                        node['solid'] = ('lathe' if abs(pf[0][0]) < 1e-9 and abs(pf[-1][0]) < 1e-9 else 'shell', pf)     # shut on the axis at both ends: a solid
                    elif sh == 'torus':
                        m = torus_mesh(float(n.get('radius', 5)), float(n.get('tube', 1)), *(sg or [32, 14])[:2])
                        node['solid'] = ('torus', float(n.get('radius', 5)), float(n.get('tube', 1)))
                    elif sh == 'disc':
                        m = disc_mesh(float(n.get('radius', 5)), (sg or [32])[0])
                        node['solid'] = ('disc', float(n.get('radius', 5)))
                    elif sh in ('text', 'extrude'):            # lettering and logos given thickness: kadr/extrude.py
                        from . import extrude
                        if sh == 'text':
                            fonts = n.get('font', 'Arial')
                            fonts = [str(x) for x in (fonts if isinstance(fonts, list) else [fonts])]
                            if 'weight' in n:
                                fonts = [f"{x.partition('#')[0]}#wght={float(n['weight']):g}" for x in fonts]
                            size = float(n.get('size', 20))
                            outline = extrude.text_outline(str(n.get('text', '')), '|'.join(fonts), size, str(n.get('align', 'center')),
                                                           float(n.get('tracking', 0)), float(n.get('line_height', 1.2)))
                        else:
                            size = float(n.get('height', 20))
                            if n.get('src'):                   # the shape of a file: an SVG, or any picture's own transparency
                                from . import svg
                                fp = resolve(str(n['src']))
                                if not os.path.isfile(fp):
                                    raise ValueError(f"нет файла {n['src']}")
                                outline = svg.outline(fp, size)
                            elif n.get('d'):
                                outline = extrude.path_outline(str(n['d']), size)
                            else:
                                raise ValueError('у extrude нужен контур: d: "M … Z" (как в SVG) или src: файл .svg / .png')
                        m = extrude.mesh(outline, float(n.get('depth', size * .2)), float(n.get('bevel', 0)), int(n.get('bevel_steps', 3)))
                        if not len(m[1]):
                            raise ValueError('пустой контур: нет ни одной буквы или замкнутой фигуры')
                    else:
                        m = plane_mesh(float(sz[0]), float(sz[1]), sg if sg is None or len(sg) > 1 else sg * 2, n.get('relief'))
                        node['solid'] = ('sheet', float(sz[0]) / 2, float(sz[1]) / 2)
                except (TypeError, ValueError, IndexError) as e:
                    F.issue('error', f'scene: «{sh}»: {e}' if sh in ('text', 'extrude') or str(e).startswith(('relief', 'profile')) else
                            f'scene: у «{sh}» неверные размеры (box / rbox: size [w, h, d]; sphere / dome / disc: radius; cylinder: radius (или [верх, низ]), height; lathe: profile [[радиус, высота], …]; '
                            'torus: radius, tube; plane: size [w, h]; segments — целое число от 3)', where)
                    m = box_mesh(10, 10, 10)
                meshes.append(m)
                em = n.get('emissive')
                # alpha: the picture's own transparency counts (a sticker, a sign cut out of its sheet); env: how much of
                # the surroundings the body takes (1 — as they are; a table that should not mirror the lamps: .04)
                mats.append(dict(id=len(mats), name=node['name'], kind='basic' if n.get('unlit') else 'standard', color=col(n.get('color'), '#ffffff' if n.get('map') else '#cccccc'), roughness=float(n.get('roughness', .6)),
                                 metalness=float(n.get('metal', 0)), emissive=col(em, '#000000') if em is not None else [0, 0, 0], emissiveIntensity=float(n.get('glow', 1 if em is not None else 0)),
                                 opacity=float(n.get('opacity', 1)), transparent=float(n.get('opacity', 1)) < 1 or bool(n.get('alpha')), side=2 if (n.get('sides') == 2 or sh == 'plane') else 0, map=picture(n),
                                 envMapIntensity=float(n.get('env', 1)),
                                 clearcoat=float(n.get('clearcoat', 0)), clearcoatRoughness=float(n.get('clearcoat_roughness', 0)), additive=False, depthWrite=float(n.get('opacity', 1)) >= 1, depthTest=True))
                node.update(kind='Mesh', geo=len(meshes) - 1, mat=len(mats) - 1, cast=bool(n.get('cast', True)), receive=bool(n.get('receive', True)))
            walk(n.get('children'), i)
    walk(d.get('nodes'), -1)
    lights, lprops = [], []
    for k, L in enumerate(d.get('lights') or []):
        keys(L, LIGHT, 'свет')
        ty = L.get('type', 'point')
        if ty not in KINDS:
            F.issue('error', f'scene: свет type «{ty}»', where, ty, set(KINDS))
            continue
        w0 = f'{where}.scene.lights[{k}]'
        sh_ = L.get('shadow', False)                           # true, or {near, far}: between which distances from the lamp things
        if isinstance(sh_, dict):                              # cast shadows (the tighter, the finer the shadow's edge)
            keys(sh_, {'near', 'far'}, 'свет.shadow')
        one = dict(kind=KINDS[ty], name=str(L['id']) if L.get('id') is not None else '', color=col(L.get('color'), '#ffffff'), intensity=1.0, pos=[0, 100, 0], target=[0, 0, 0],
                   distance=float(L.get('distance', 0)), decay=float(L.get('decay', 2)),
                   angle=math.radians(float(L.get('angle', 30))), penumbra=float(L.get('penumbra', .5)), cast=bool(sh_), jitter=float(L.get('soft', 0)),
                   shadow=dict(size=2048, bias=-0.0004, normalBias=0.3, near=float(sh_.get('near', 1)) if isinstance(sh_, dict) else 1.0,
                               far=float(sh_.get('far', 5000)) if isinstance(sh_, dict) else 5000.0, radius=1, focus=1, given=isinstance(sh_, dict)), ground=col(L.get('ground'), '#000000'))
        if ty == 'sky':
            one['pos'] = [0, 1, 0]
        if ty == 'panel':
            sz = L.get('size', [20, 20])
            one.update(width=float(sz[0]), height=float(sz[1]))
            lprops.append((k, 'target', F.prop(L.get('look_at', L.get('target', [0, 0, 0])), w0 + '.look_at', 3)))
        elif 'target' in L or 'look_at' in L:
            lprops.append((k, 'target', F.prop(L.get('target', L.get('look_at')), w0 + '.target', 3)))
        if 'position' in L:
            lprops.append((k, 'pos', F.prop(L['position'], w0 + '.position', 3)))
        lprops.append((k, 'intensity', F.prop(L.get('intensity', 1), w0 + '.intensity')))
        lights.append(one)
    c = d.get('camera') or {}
    keys(c, CAMERA, 'camera')
    cam = dict(position=F.prop(c.get('position', [0, 30, 150]), f'{where}.scene.camera.position', 3), look_at=F.prop(c.get('look_at', [0, 0, 0]), f'{where}.scene.camera.look_at', 3),
               fov=F.prop(c.get('fov', 35), f'{where}.scene.camera.fov'), lens=F.prop(c.get('lens', 0), f'{where}.scene.camera.lens'),
               focus=F.prop(c['focus'], f'{where}.scene.camera.focus') if 'focus' in c else None)
    k = d.get('look') or {}
    keys(k, LOOK, 'look')
    vg = k.get('vignette', 0)
    look = dict(exposure=float(k.get('exposure', 1)), bloom=float(k.get('glow', .3)), glow_from=float(k.get('glow_from', 1)), tint=[float(v) for v in k.get('tint', [1, 1, 1])],
                aces=float(k.get('filmic', .82)), vignette=[.42, .86, float(vg)], grain=float(k.get('grain', 0)), lampJitter=0, spp=16, shutter=float(k.get('shutter', .5)), raw=bool(k.get('raw', False)))   # shutter: the share of a frame it stays open (0: no blur)
    accept = []                                                # pairs the scene's check is told are meant to meet (kadr/scenecheck.py)
    known = {n['name'] for n in nodes}
    for pair in d.get('accept') or []:
        pair = [pair] if isinstance(pair, str) else pair
        if not isinstance(pair, (list, tuple)) or len(pair) not in (1, 2):
            F.issue('error', 'scene.accept: пары имён, которым положено сходиться, и имена тех, кому положено висеть — [[кружка, ноутбук], [муха]]', where)
            continue
        for nm in pair:
            if str(nm) not in known:
                F.issue('error', f'scene.accept: в сцене нет узла «{nm}»', where, str(nm), known)
        accept.append(tuple(str(v) for v in pair))
    J = dict(units='scene', accept=accept, nodes=nodes, geos=[], mats=mats, texs=texs, lights=lights, env=env, camera=dict(near=float(c.get('near', 1)), far=float(c.get('far', 5000)), aspect=1), look=look)
    return dict(J=J, meshes=meshes, props=props, lprops=lprops, cam=cam, figures=figures)


_uids = __import__('itertools').count()

class Scene:
    """a scene read once: its models and pictures (what the core keeps on the GPU) and its tree"""

    def __init__(self, path=None, clip=None, made=None):
        self.made = made
        vs, ixs, self.geos = [], [], []
        nv = ni = 0
        if made is not None:                                   # written in the document (parse)
            J = self.J = made['J']
            self.dir = ''
            for v, ix in made['meshes']:
                self.geos.append((ni, len(ix), nv))
                vs.append(v); ixs.append(ix)
                nv += len(v); ni += len(ix)
        else:                                                  # exported from another renderer: scene.json + scene.bin
            self.dir = os.path.dirname(os.path.abspath(path))
            J = self.J = json.load(open(path, encoding='utf-8'))
            raw = open(os.path.join(self.dir, 'scene.bin'), 'rb').read()
            for g in J['geos']:
                v = np.frombuffer(raw, np.float32, g['verts'] * 8, g['v']).reshape(-1, 8)
                ix = np.frombuffer(raw, np.uint32, g['tris'] * 3, g['i'])
                self.geos.append((ni, len(ix), nv))
                vs.append(v); ixs.append(ix)
                nv += len(v); ni += len(ix)
            for L in J['lights']:
                L.setdefault('jitter', J['look'].get('lampJitter', 0) if L['kind'] == 'SpotLight' else 0)
            for m in J['mats']:                                # three.js takes a material's own share of the surroundings only with the
                m['envMapIntensity'] = 1                       # material's own map; lit by the scene's, as these are, it ignored the number
        # a sprite's quad
        self.quad = len(self.geos)
        self.geos.append((ni, 6, nv))
        vs.append(np.array([[-.5, -.5, 0, 0, 0, 1, 0, 0], [.5, -.5, 0, 0, 0, 1, 1, 0], [.5, .5, 0, 0, 0, 1, 1, 1], [-.5, .5, 0, 0, 0, 1, 0, 1]], np.float32))
        ixs.append(np.array([0, 1, 2, 0, 2, 3], np.uint32))
        self.verts, self.indices = np.ascontiguousarray(np.concatenate(vs)), np.ascontiguousarray(np.concatenate(ixs))
        self.maps = []
        for t in J['texs']:
            a = t['data'] if t.get('data') is not None else np.asarray(Image.open(t.get('path') or os.path.join(self.dir, f"tex_{t['id']}.png")).convert('RGBA'))
            if max(a.shape[:2]) > 4096:                        # a photograph straight from a camera: more than a body can show
                k = 4096 / max(a.shape[:2])
                a = np.asarray(Image.fromarray(a).resize((max(1, int(a.shape[1] * k)), max(1, int(a.shape[0] * k))), Image.LANCZOS))
            self.maps.append((mips(np.ascontiguousarray(a), t['srgb']), bool(t['srgb']), bool(t['repeat'])))
        e = J['env']
        if e and e.get('src'):                                 # baked here from the studio, a panorama or a room of lamps
            from . import envmap
            self.env = envmap.atlas(e['src'], e.get('turn', 0.0), e.get('room'))
        else:
            self.env = np.fromfile(os.path.join(self.dir, 'env.bin'), np.uint16).reshape(e['h'], e['w'], 4) if e else np.zeros((4, 4, 4), np.uint16)
        self.nodes = J['nodes']
        self.order = list(range(len(self.nodes)))             # parents come first (a traversal)
        self.p0 = np.array([n['p'] for n in self.nodes], float).reshape(-1, 3); self.q0 = np.array([n['q'] for n in self.nodes], float).reshape(-1, 4)
        self.s0 = np.array([n['s'] for n in self.nodes], float).reshape(-1, 3)
        self.vis0 = np.array([1.0 if n['visible'] else 0.0 for n in self.nodes])
        self.parent = [n['parent'] for n in self.nodes]
        self.clip = None
        C = J.get('clip')
        if clip and C:
            self.clip = np.fromfile(clip, np.float32).reshape(C['n'], C['width'])
            self.C = C
        self.draw_nodes = [i for i, n in enumerate(self.nodes) if n['kind'] in ('Mesh', 'Sprite')]

    def gpu(self):
        """the scene on this thread's GPU device (each render thread has its own)"""
        from kadr_core import Scene3d
        cache = _tls.__dict__.setdefault('scenes', {})          # by the scene's own number, not id(): a freed scene's id is
        key = self.__dict__.setdefault('uid', next(_uids))      # handed to the next one, which would then be drawn with the old bodies
        if key not in cache:
            while len(cache) >= 6:                             # a long session of many films does not keep every scene on the GPU
                cache.pop(next(iter(cache)))
            ltc = [np.fromfile(os.path.join(DATA, f'ltc_{k}.bin'), np.float32).astype(np.float16).view(np.uint16).reshape(64, 64, 4) for k in (1, 2)]
            d = np.fromfile(os.path.join(DATA, 'dfg.bin'), np.uint16).reshape(16, 16, 2)
            dfg = np.zeros((16, 16, 4), np.uint16); dfg[..., :2] = d
            sets = [[m['map'], m.get('map_mr', -1), m.get('map_n', -1), m.get('map_e', -1)] for m in self.J['mats']]   # a material's four pictures
            cache[key] = Scene3d(self.verts, self.indices, self.geos, self.maps, sets, np.ascontiguousarray(self.env), np.ascontiguousarray(ltc[0]), np.ascontiguousarray(ltc[1]), dfg)
        return cache[key]

    # ---- the clip: what moves
    def rows(self, ts, t_frame):
        """the clip at each sample's time (linear between its rows; across a camera cut — the row on the sample's own
        side, and no sample of a frame is taken from beyond a cut that falls inside the frame)"""
        C, clip = self.C, self.clip
        out = []
        for t in ts:
            for c in C['cuts']:
                if t_frame < c <= t:
                    t = c - 1e-6
            k = min(max(t * C['rate'], 0.0), len(clip) - 1.001)
            i = int(k)
            a = k - i
            t0, t1 = i / C['rate'], (i + 1) / C['rate']
            cut = next((c for c in C['cuts'] if t0 < c <= t1), None)
            if cut is not None:
                out.append(clip[i] if t < cut else clip[i + 1])
            else:
                out.append(clip[i] * (1 - a) + clip[i + 1] * a)
        return np.array(out)

    def frame(self, t, w, h, fps, samples=None, look=None, flags=None, over=None, cam_over=None, cuts=None, lit=None, mts=None):
        """everything the core needs to draw the frame at t → (globals, insts, mats, draws, look). over: named nodes moved
        from the document [(name, what, property)]; cam_over: the camera set there; cuts: when it jumps"""
        J = self.J
        K = dict(J['look'], **(look or {}))
        flags = flags or {}
        still = flags.get('still')
        S = max(1, min(256, int(samples or K['spp'])))
        shutter = K['shutter'] / fps
        ts = [t + (i + .5) / S * shutter for i in range(S)] if not still else [t] * S
        for c in cuts or ():                                   # no sample of a frame is taken from beyond a cut inside it
            if t < c:
                ts = [min(x, c - 1e-6) for x in ts]
        P, Q, Sc = np.repeat(self.p0[None], S, 0), np.repeat(self.q0[None], S, 0), np.repeat(self.s0[None], S, 0)
        vis = np.repeat(self.vis0[None], S, 0)
        mcol = np.array([m['color'] + [m['opacity'], m['emissiveIntensity']] for m in J['mats']], float).reshape(-1, 5)
        cam = np.zeros((S, 9))
        lights = [[dict(l) for l in J['lights']] for _ in range(S)]
        if self.clip is not None:                              # what an exported scene's clip says
            R = self.rows(ts, t)
            C = self.C
            o = 0
            for nid in C['nodes']:
                P[:, nid], Q[:, nid], Sc[:, nid], vis[:, nid] = R[:, o:o + 3], R[:, o + 3:o + 7], R[:, o + 7:o + 10], np.round(R[:, o + 10])
                o += 11
            for mid in C['mats']:
                mcol[mid] = R[0, o:o + 5]
                o += 5
            o += 7 * len(C['lights'])
            cam[:] = R[:, o:o + 9]
        elif self.made is not None:                            # what the document's own keys and formulas say
            M = self.made
            for s_, ts_ in enumerate(ts):
                for nid, key, pr in M['props']:
                    v = pr.at(ts_)
                    if key == 'position':
                        P[s_, nid] = v
                    elif key == 'rotation':
                        Q[s_, nid] = euler(v)
                    elif key == 'quat':                        # a model's own turn, as it is written in its file
                        Q[s_, nid] = v
                    elif key == 'scale':
                        Sc[s_, nid] = v if np.ndim(v) else (v, v, v)
                    else:
                        vis[s_, nid] = 1.0 if v > .5 else 0.0
                for k, key, pr in M['lprops']:
                    v = pr.at(ts_)
                    lights[s_][k][key] = list(v) if isinstance(v, tuple) else v
                c = M['cam']
                pos, at = np.array(c['position'].at(ts_)), np.array(c['look_at'].at(ts_))
                cam[s_] = [*pos, *at, c['fov'].at(ts_), c['lens'].at(ts_), c['focus'].at(ts_) if c['focus'] is not None else np.linalg.norm(pos - at)]
        for name, key, pr in lit or ():                        # named lamps turned up and down from the document
            k = self.__dict__.setdefault('light_by_name', {l.get('name'): i for i, l in enumerate(J['lights']) if l.get('name')})[name]
            for s_, ts_ in enumerate(ts):
                lights[s_][k]['intensity'] = float(pr.at(ts_))
        for name, key, pr in mts or ():                        # named materials: see-through, glow, a darker colour
            k = self.__dict__.setdefault('mat_by_name', {m.get('name'): i for i, m in enumerate(J['mats']) if m.get('name')})[name]
            v = float(pr.at(ts[0]))
            if key == 'opacity':
                mcol[k, 3] = v
            elif key == 'glow':
                mcol[k, 4] = v
            else:
                mcol[k, :3] = np.array(J['mats'][k]['color'], float) * v
        if over:                                               # the document's own say over named nodes: a puppet's strings
            ids = self.__dict__.setdefault('by_name', {n.get('name'): i for i, n in enumerate(self.nodes) if n.get('name')})
            for s_, ts_ in enumerate(ts):
                for name, key, pr in over:
                    nid, v = ids[name], pr.at(ts_)
                    if key == 'position':
                        P[s_, nid] = v
                    elif key == 'rotation':
                        Q[s_, nid] = euler(v)
                    elif key == 'scale':
                        Sc[s_, nid] = v if isinstance(v, tuple) else (v, v, v)
                    else:
                        vis[s_, nid] = 1.0 if v > .5 else 0.0
        if flags.get('nodes') is not None:                     # only these bodies are drawn (a part looked at on its own)
            keep = set(flags['nodes'])
            vis[:, [i for i in self.draw_nodes if i not in keep]] = 0.0
        if flags.get('lights') is not None:                    # the scene's own lamps put aside for these (an even light to judge a shape by)
            lights = [[dict(l) for l in flags['lights']] for _ in range(S)]
        if cam_over:
            for s_, ts_ in enumerate(ts):
                pos, at = np.array(cam_over['position'].at(ts_)), np.array(cam_over['look_at'].at(ts_))
                cam[s_, :6] = [*pos, *at]
                cam[s_, 6] = cam_over['fov'].at(ts_) if cam_over['fov'] is not None else (cam[s_, 6] or 35.0)
                cam[s_, 7] = cam_over['lens'].at(ts_) if cam_over['lens'] is not None else cam[s_, 7]
                cam[s_, 8] = cam_over['focus'].at(ts_) if cam_over['focus'] is not None else float(np.linalg.norm(pos - at))
        elif self.clip is None and self.made is None:
            raise ValueError('у сцены из файла нет ни записи движений (clip), ни камеры: задайте camera: {position, look_at, fov}')
        local = trs(P, Q, Sc)                                  # (S, N, 4, 4)
        world = np.zeros_like(local)
        seen = vis.copy()
        for i in self.order:
            p = self.parent[i]
            if p < 0:
                world[:, i] = local[:, i]
            else:
                world[:, i] = world[:, p] @ local[:, i]
                seen[:, i] = seen[:, i] * seen[:, p]
        if flags.get('probe') is not None:                     # where everything stands at this moment, for whoever asks
            flags['probe'].update(world=world[0].copy(), seen=seen[0].copy())
        aspect = w / h
        e = J['env']
        has_env = e is not None and not flags.get('noenv')
        G = np.zeros((S, 288), np.float32)
        mats = np.zeros((max(1, len(J['mats'])), 20), np.float32)
        for k, m in enumerate(J['mats']):
            basic = m['kind'] != 'standard'
            mats[k] = [*mcol[k][:3], mcol[k][3], *(np.array(m['emissive']) * mcol[k][4]), m.get('envMapIntensity', 1), m['roughness'], m['metalness'], m['clearcoat'], m['clearcoatRoughness'],
                       1.0 if basic else 0.0, 1.0 if m['side'] == 2 else 0.0, 1.0 if m['map'] >= 0 else 0.0, 0.0 if m['transparent'] else 1.0, 1, 1, 1, 0]
            first = next((m.get(key, -1) for key in ('map', 'map_mr', 'map_n', 'map_e') if m.get(key, -1) >= 0), -1)
            if first >= 0:                                     # how its pictures lie: as the first one it has
                tx = J['texs'][first]
                mats[k, 16:19] = [tx['rep'][0], tx['rep'][1], 1.0 if tx['flipY'] else 0.0]
            mats[k, 19] = (1 if m.get('map_mr', -1) >= 0 else 0) | (2 if m.get('map_n', -1) >= 0 else 0) | (4 if m.get('map_e', -1) >= 0 else 0) | (8 if m.get('ao') else 0)
        nd = len(self.draw_nodes)
        insts = np.zeros((max(1, S * nd), 36), np.float32)
        draws = []
        for s in range(S):
            i1 = s + 1
            pos, at, fov, lens, focus = cam[s, :3].copy(), cam[s, 3:6], float(cam[s, 6]), float(cam[s, 7]), float(cam[s, 8])
            cw = look_at(pos, at)
            u, v = halton(i1, 2), halton(i1, 3)
            r, th = (0.0 if still else math.sqrt(u) * lens), v * 2 * math.pi
            dx, dy = r * math.cos(th), r * math.sin(th)
            cw[:3, 3] += cw[:3, 0] * dx + cw[:3, 1] * dy       # a point on the lens; the picture stays sharp at `focus`
            proj = perspective(fov, aspect, *(flags.get('clip') or (J['camera']['near'], J['camera']['far'])))
            jx, jy = (0.0, 0.0) if still else ((halton(i1, 5) - .5) * 2 / w, (halton(i1, 7) - .5) * 2 / h)
            proj[0, 2] += -proj[0, 0] * dx / max(focus, 1e-6) + jx
            proj[1, 2] += -proj[1, 1] * dy / max(focus, 1e-6) + jy
            view = np.linalg.inv(cw)
            vp_gl = proj @ view
            g = G[s]
            g[0:16] = (ZFIX @ vp_gl).T.ravel(); g[16:32] = view.T.ravel(); g[32:48] = (ZFIX @ proj).T.ravel()
            g[64:68] = [*cw[:3, 3], e['intensity'] if has_env else 0.0]
            nl = nh = nr = 0
            jit = np.array([halton(i1, 11) - .5, halton(i1, 13) - .5, halton(i1, 17) - .5]) * 2
            for L in lights[s]:
                kind = L['kind']
                if flags.get('only') and kind not in flags['only']:
                    continue
                col = np.array(L['color']) * L['intensity']
                if kind in ('PointLight', 'SpotLight', 'DirectionalLight') and nl < 8:
                    o = 80 + nl * 16
                    lp = np.array(L['pos'], float)
                    if kind == 'SpotLight':
                        if not still:
                            lp = lp + jit * L.get('jitter', 0)   # the lamp is somewhere else at each sample: a soft shadow
                        d = lp - np.array(L['target']); dist = np.linalg.norm(d); d /= dist
                        cast = 1.0 if (L.get('cast') and not flags.get('noshadow')) else 0.0
                        g[o:o + 16] = [*lp, 1, *col, L['distance'], *d, L['decay'], math.cos(L['angle']), math.cos(L['angle'] * (1 - L['penumbra'])), cast, 0]
                        if cast:
                            sh = L['shadow']
                            near, far = (sh['near'], L['distance'] or sh['far']) if self.made is None else (sh['near'], sh['far']) if sh.get('given') else (dist * .1, dist * 6)
                            sp = perspective(math.degrees(2 * L['angle'] * sh.get('focus', 1)), 1.0, near, far)
                            g[48:64] = (ZFIX @ sp @ np.linalg.inv(look_at(lp, L['target']))).T.ravel()
                            g[68:72] = [sh['bias'], sh['normalBias'], sh['radius'], sh['size']]
                    elif kind == 'PointLight':
                        g[o:o + 16] = [*lp, 0, *col, L['distance'], 0, 0, 0, L['decay'], 0, 0, 0, 0]
                    else:
                        d = lp - np.array(L['target']); d /= np.linalg.norm(d)
                        g[o:o + 16] = [0, 0, 0, 2, *col, 0, *d, 0, 0, 0, 0, 0]
                    nl += 1
                elif kind == 'HemisphereLight' and nh < 4:
                    o = 208 + nh * 12
                    d = np.array(L['pos'], float); d /= np.linalg.norm(d)
                    g[o:o + 12] = [*d, 0, *col, 0, *(np.array(L['ground']) * L['intensity']), 0]
                    nh += 1
                elif kind == 'RectAreaLight' and nr < 2:
                    o = 256 + nr * 16
                    rot = trs(np.zeros(3), np.array(L['quat']), np.ones(3))[:3, :3] if 'quat' in L else look_at(L['pos'], L['target'])[:3, :3]
                    g[o:o + 16] = [*col, 0, *L['pos'], 0, *(rot @ [L['width'] / 2, 0, 0]), 0, *(rot @ [0, L['height'] / 2, 0]), 0]
                    nr += 1
            g.view(np.uint32)[72:76] = [nl, nh, nr, 1 if has_env else 0]
            g[76:80] = [h, math.log2(e['h']) - 2, 1 / e['w'], 1 / e['h']] if e else [h, 0, 1, 1]
            # the things to draw
            if not nd:
                continue
            rows = insts[s * nd:(s + 1) * nd]
            W = world[s, self.draw_nodes]
            rows[:, 0:16] = W.transpose(0, 2, 1).reshape(nd, 16)
            nm = np.zeros((nd, 4, 4))
            det = np.abs(np.linalg.det(W[:, :3, :3])) > 1e-12
            nm[det, :3, :3] = np.linalg.inv(W[det, :3, :3]).transpose(0, 2, 1)
            rows[:, 16:32] = nm.transpose(0, 2, 1).reshape(nd, 16)
            iu = rows.view(np.uint32)
            later = []
            for k, nid in enumerate(self.draw_nodes):
                n = self.nodes[nid]
                if not seen[s, nid] or not det[k]:
                    continue
                m = J['mats'][n['mat']]
                iu[k, 32:36] = [n['mat'], 1 if n['receive'] else 0, 0, 0]
                sprite = n['kind'] == 'Sprite'
                geo = self.quad if sprite else n['geo']
                if n['cast'] and not sprite:
                    draws.append((s, 0, 1 if m['side'] == 2 else 0, geo, s * nd + k, 0))
                key = (1 if m['side'] == 2 else 0) | (0 if m['depthWrite'] else 2) | ((2 if m['additive'] else 1) << 2 if m['transparent'] else 0) | (16 if sprite else 0)
                row = (s, 1, key, geo, s * nd + k, n['mat'] + 1)       # its material's set of pictures
                if m['transparent']:
                    c = (ZFIX @ vp_gl) @ W[k, :, 3]
                    later.append((-(c[2] / c[3]), nid, row))
                else:
                    draws.append(row)
            draws += [r for _, _, r in sorted(later)]             # the see-through ones last, the farthest first
        vg = K['vignette']
        lk = [w, h, K['exposure'], 0.0 if flags.get('nobloom') else K['bloom'], *K['tint'], K['aces'], *vg, 0.0 if flags.get('nograin') else K['grain'],
              float(int(round(t * fps)) % 97), K.get('glow_from', 1.0), 1.0 if K.get('raw') else 0.0]
        return G, insts, mats, np.array(draws, np.uint32).reshape(-1, 6), lk

    def render(self, t, w, h, fps, **kw):
        G, insts, mats, draws, lk = self.frame(t, w, h, fps, **kw)
        return self.gpu().render(w, h, np.ascontiguousarray(G), np.ascontiguousarray(insts), np.ascontiguousarray(mats), np.ascontiguousarray(draws), lk, [0.0, 0.0, 0.0, 0.0])


@functools.lru_cache(8)
def load(path, clip, mtime):
    return Scene(path, clip)


def layer_scene(L):
    """the layer's scene: a file (read once per file) or the one written in the document"""
    if L.get('made') is not None:
        if L.get('_scene') is None:
            L['_scene'] = Scene(made=L['made'])
        return L['_scene']
    return load(L['src'], L.get('clip'), os.path.getmtime(L['src']))


def picture(L, t, fps, scale=1.0):
    """the layer's picture at scene time t → (h, w, 4) float32: colour 0..255, alpha 0..1"""
    w, h = (max(2, int(round(v * scale))) for v in L['psize'])
    return layer_scene(L).render(t, w, h, fps, samples=L.get('samples'), look=L.get('look'), over=L.get('over'), cam_over=L.get('cam_over'), cuts=L.get('cuts'),
                                 lit=L.get('lit'), mts=L.get('mts')).download()
