"""Models made elsewhere (kadr/scene3d.py: `shape: model`): a glTF 2.0 file — .glb, or .gltf with its files beside it —
as Blender, Sketchfab and most stock libraries give them.

Read: the tree of nodes with their places, turns and sizes; the meshes (triangles, with their normals and picture
coordinates — normals are made here when the file has none); materials — base colour, metal, roughness and glow, each
with its picture, relief (a normal map), how much surrounding light reaches a place (when it is the red of the
roughness picture), see-through, two-sided, lacquer; and the animations that move, turn and size nodes.
Not read: skeletons (a rigged character stands in its rest pose), shape keys, cameras and lights (the scene has its
own). What was left out is said once, when the document is read."""
import base64, io, json, math, os, struct

import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


class Track:
    """a value that moves through keys in time: straight between them (turns — the short way round a sphere), or held"""

    def __init__(self, times, values, turn=False, step=False, loop=False, speed=1.0, length=None):
        self.t, self.v, self.turn, self.step, self.loop, self.speed = np.asarray(times, float), np.asarray(values, float), turn, step, loop, speed
        self.length = float(length if length is not None else (self.t[-1] if len(self.t) else 0.0))
        self.animated = len(self.t) > 1

    def at(self, t):
        t = t * self.speed
        if self.loop and self.length > 0:
            t = t % self.length
        i = int(np.searchsorted(self.t, t, 'right')) - 1
        if i < 0:
            return self.v[0]
        if i >= len(self.t) - 1:
            return self.v[-1]
        a, b = self.v[i], self.v[i + 1]
        if self.step:
            return a
        u = (t - self.t[i]) / max(1e-12, self.t[i + 1] - self.t[i])
        if not self.turn:
            return a + (b - a) * u
        d = float(np.dot(a, b))
        if d < 0:
            b, d = -b, -d
        if d > .9995:
            q = a + (b - a) * u
        else:
            th = math.acos(d)
            q = (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)
        return q / np.linalg.norm(q)


def _split(m):
    """a 4×4 matrix (column-major, as glTF writes it) → place, turn (xyzw), size"""
    m = np.asarray(m, float).reshape(4, 4).T
    p = m[:3, 3]
    s = np.linalg.norm(m[:3, :3], axis=0)
    if np.linalg.det(m[:3, :3]) < 0:
        s[0] = -s[0]
    r = m[:3, :3] / np.where(np.abs(s) > 1e-12, s, 1)
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        w = math.sqrt(1 + tr) / 2
        q = [(r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        x = math.sqrt(max(0.0, 1 + r[i, i] - r[j, j] - r[k, k])) / 2
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / (4 * x)]
        q[i], q[j], q[k] = x, (r[j, i] + r[i, j]) / (4 * x), (r[k, i] + r[i, k]) / (4 * x)
    return p.tolist(), q, s.tolist()


def load(path, may_read=None):
    """the file → {nodes: [{name, parent, p, q, s, draws: [(mesh, material)]}], meshes: [(vertices (n, 8), indices)],
    mats: [...], texs: [{data (h, w, 4) uint8, repeat}], anims: [{name, length, tracks: [(node, what, Track args)]}],
    left_out: [what the file has that is not read]}"""
    raw = open(path, 'rb').read()
    here = os.path.dirname(os.path.abspath(path))

    def beside(uri):
        """a file the model names: beside the model, or deeper. A name that leads out of its folder (../, an absolute
        path) is read only if the film may read there (may_read) — otherwise any file of the disk could be a «buffer»"""
        p = os.path.join(here, str(uri).replace('%20', ' '))
        real, base = os.path.realpath(p), os.path.realpath(here)
        if not (real.startswith(base + os.sep) or (may_read is not None and may_read(p))):
            raise ValueError(f'модель ссылается на файл вне своей папки ({uri})')
        return p
    blob = None
    if raw[:4] == b'glTF':
        ver, total = struct.unpack_from('<II', raw, 4)
        o = 12
        J = None
        while o < total:
            n, kind = struct.unpack_from('<II', raw, o)
            body = raw[o + 8:o + 8 + n]
            if kind == 0x4E4F534A:
                J = json.loads(body.decode('utf-8'))
            elif kind == 0x004E4942 and blob is None:
                blob = body
            o += 8 + n + (-n % 4)
        if J is None:
            raise ValueError(f'{os.path.basename(path)}: в .glb нет описания сцены')
    else:
        try:
            J = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError(f'{os.path.basename(path)}: это не glTF (.glb или .gltf)')
    if str(J.get('asset', {}).get('version', '2')).split('.')[0] != '2':
        raise ValueError(f'{os.path.basename(path)}: читается glTF 2.0, а это версия {J.get("asset", {}).get("version")}')
    need = set(J.get('extensionsRequired') or [])
    for bad in need & {'KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'KHR_mesh_quantization'} - {'KHR_mesh_quantization'}:
        raise ValueError(f'{os.path.basename(path)}: модель сжата ({bad}) — сохраните её без сжатия (в Blender: Export glTF, снять Compression)')

    bufs = []
    for i, b in enumerate(J.get('buffers') or []):
        uri = b.get('uri')
        if uri is None:
            bufs.append(blob or b'')
        elif uri.startswith('data:'):
            bufs.append(base64.b64decode(uri.split(',', 1)[1]))
        else:
            bufs.append(open(beside(uri), 'rb').read())

    def view(i):
        v = J['bufferViews'][i]
        return bufs[v['buffer']], v.get('byteOffset', 0), v.get('byteLength', 0), v.get('byteStride', 0)

    def acc(i):
        a = J['accessors'][i]
        dt, w, n = COMP[a['componentType']], WIDTH[a['type']], a['count']
        if 'bufferView' not in a:
            out = np.zeros((n, w), dt)
        else:
            buf, off, _, stride = view(a['bufferView'])
            off += a.get('byteOffset', 0)
            size = np.dtype(dt).itemsize * w
            if stride and stride != size:
                rows = np.frombuffer(buf, np.uint8, count=stride * (n - 1) + size, offset=off)
                idx = (np.arange(n) * stride)[:, None] + np.arange(size)[None]
                out = rows[idx].copy().view(dt).reshape(n, w)
            else:
                out = np.frombuffer(buf, dt, n * w, off).reshape(n, w)
        out = out.astype(np.float64) if dt != np.uint32 else out
        if a.get('normalized') and dt != np.float32:
            out = out / float(np.iinfo(dt).max)
        return out

    left = []
    texs = []

    seen = {}

    def picture(ti, srgb=True):
        from PIL import Image
        if (ti, srgb) in seen:
            return seen[ti, srgb]
        t = J['textures'][ti]
        im = J['images'][t.get('source', 0)]
        if im.get('bufferView') is not None:
            buf, off, n, _ = view(im['bufferView'])
            data = buf[off:off + n]
        elif str(im.get('uri', '')).startswith('data:'):
            data = base64.b64decode(im['uri'].split(',', 1)[1])
        else:
            data = open(beside(im['uri']), 'rb').read()
        a = np.asarray(Image.open(io.BytesIO(data)).convert('RGBA'))
        wrap = (J.get('samplers') or [{}])[t['sampler']].get('wrapS', 10497) if t.get('sampler') is not None else 10497
        texs.append(dict(data=np.ascontiguousarray(a), repeat=wrap != 33071, srgb=srgb))
        seen[ti, srgb] = len(texs) - 1
        return len(texs) - 1

    mats = []
    for m in J.get('materials') or []:
        pb = m.get('pbrMetallicRoughness') or {}
        col = pb.get('baseColorFactor', [1, 1, 1, 1])
        tx = picture(pb['baseColorTexture']['index']) if pb.get('baseColorTexture') is not None else -1
        mr = picture(pb['metallicRoughnessTexture']['index'], False) if pb.get('metallicRoughnessTexture') is not None else -1
        nm = picture(m['normalTexture']['index'], False) if m.get('normalTexture') is not None else -1
        emt = picture(m['emissiveTexture']['index']) if m.get('emissiveTexture') is not None else -1
        # how much surrounding light reaches the surface: read when it is the red of the roughness / metal picture (as it
        # usually is — one «ORM» file)
        ao = m.get('occlusionTexture') is not None and pb.get('metallicRoughnessTexture') is not None and m['occlusionTexture']['index'] == pb['metallicRoughnessTexture']['index']
        if m.get('occlusionTexture') is not None and not ao and 'отдельные карты затенения' not in left:
            left.append('отдельные карты затенения')
        cc = (m.get('extensions') or {}).get('KHR_materials_clearcoat') or {}
        em = np.array(m.get('emissiveFactor', [0, 0, 0]), float) * ((m.get('extensions') or {}).get('KHR_materials_emissive_strength') or {}).get('emissiveStrength', 1.0)
        mats.append(dict(color=[float(c) for c in col[:3]], opacity=float(col[3]) if m.get('alphaMode') == 'BLEND' else 1.0, transparent=m.get('alphaMode') == 'BLEND',
                         metal=float(pb.get('metallicFactor', 1)), roughness=float(pb.get('roughnessFactor', 1)), emissive=em.tolist(),
                         two=bool(m.get('doubleSided')), map=tx, map_mr=mr, map_n=nm, map_e=emt, ao=bool(ao), clearcoat=float(cc.get('clearcoatFactor', 0)), clearcoat_roughness=float(cc.get('clearcoatRoughnessFactor', 0)),
                         unlit='KHR_materials_unlit' in (m.get('extensions') or {})))
    mats.append(dict(color=[1, 1, 1], opacity=1.0, transparent=False, metal=1.0, roughness=1.0, emissive=[0, 0, 0], two=False, map=-1, map_mr=-1, map_n=-1, map_e=-1, ao=False, clearcoat=0.0, clearcoat_roughness=0.0, unlit=False))

    meshes, prims = [], {}                                     # prims: mesh number → [(our mesh, material)]
    for mi, m in enumerate(J.get('meshes') or []):
        prims[mi] = []
        for pr in m.get('primitives') or []:
            if pr.get('mode', 4) != 4 or 'POSITION' not in pr.get('attributes', {}):
                if 'линии и точки' not in left:
                    left.append('линии и точки')
                continue
            if pr.get('targets') and 'ключи формы' not in left:
                left.append('ключи формы')
            at = pr['attributes']
            P = acc(at['POSITION'])[:, :3]
            ix = acc(pr['indices']).astype(np.int64).ravel() if pr.get('indices') is not None else np.arange(len(P))
            ix = ix[:len(ix) // 3 * 3]
            if len(ix) and (int(ix.max()) >= len(P) or int(ix.min()) < 0):
                raise ValueError('в модели номера вершин выходят за их число')
            if 'NORMAL' in at:
                N = acc(at['NORMAL'])[:, :3]
            else:                                              # none in the file: each triangle its own flat one
                P = P[ix]
                ix = np.arange(len(P))
                t = P.reshape(-1, 3, 3)
                fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
                N = np.repeat(fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12), 3, 0)
            UV = acc(at['TEXCOORD_0'])[:, :2] if 'TEXCOORD_0' in at else np.zeros((len(P), 2))
            if len(UV) != len(P):                              # normals were made per triangle: the picture coordinates follow
                UV = UV[acc(pr['indices']).astype(np.int64).ravel()[:len(P)]]
            ln = np.linalg.norm(N, axis=1, keepdims=True)
            N = np.where(ln > 1e-9, N / np.maximum(ln, 1e-12), [0, 1.0, 0])
            meshes.append((np.concatenate([P, N, UV], 1).astype(np.float32), ix.astype(np.uint32)))
            prims[mi].append((len(meshes) - 1, pr['material'] if pr.get('material') is not None else len(mats) - 1))

    nodes = []
    for i, n in enumerate(J.get('nodes') or []):
        if 'matrix' in n:
            p, q, s = _split(n['matrix'])
        else:
            p, q, s = n.get('translation', [0, 0, 0]), n.get('rotation', [0, 0, 0, 1]), n.get('scale', [1, 1, 1])
        nodes.append(dict(name=str(n.get('name', i)), parent=-1, p=[float(v) for v in p], q=[float(v) for v in q], s=[float(v) for v in s],
                          draws=prims.get(n['mesh'], []) if n.get('mesh') is not None else []))
        if n.get('skin') is not None and 'скелеты (модель стоит в исходной позе)' not in left:
            left.append('скелеты (модель стоит в исходной позе)')
    for i, n in enumerate(J.get('nodes') or []):
        for c in n.get('children') or []:
            nodes[c]['parent'] = i
    scene = (J.get('scenes') or [{}])[J.get('scene', 0)] if J.get('scenes') else {}
    roots = set(scene.get('nodes', [i for i, n in enumerate(nodes) if n['parent'] < 0]))
    keep = [False] * len(nodes)                                # only what hangs on the scene's own roots

    def mark(i):
        keep[i] = True
        for c in (J['nodes'][i].get('children') or []):
            mark(c)
    for r in roots:
        mark(r)

    anims = []
    for a in J.get('animations') or []:
        tracks, length = [], 0.0
        for ch in a.get('channels') or []:
            tg = ch.get('target') or {}
            what = tg.get('path')
            if tg.get('node') is None or what not in ('translation', 'rotation', 'scale'):
                if what == 'weights' and 'ключи формы' not in left:
                    left.append('ключи формы')
                continue
            sm = a['samplers'][ch['sampler']]
            times, vals = acc(sm['input']).ravel(), acc(sm['output'])
            kind = sm.get('interpolation', 'LINEAR')
            if kind == 'CUBICSPLINE':                          # in-tangent, value, out-tangent for every key: the values
                vals = vals.reshape(len(times), 3, -1)[:, 1]
            if what == 'rotation':
                vals = vals / np.maximum(np.linalg.norm(vals, axis=1, keepdims=True), 1e-12)
            tracks.append((tg['node'], what, times, vals, kind == 'STEP'))
            length = max(length, float(times[-1]) if len(times) else 0.0)
        anims.append(dict(name=str(a.get('name', len(anims))), length=length, tracks=tracks))
    return dict(nodes=nodes, keep=keep, meshes=meshes, mats=mats, texs=texs, anims=anims, left_out=left)


_cache = {}


def cached(path, may_read=None):
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _cache:
        if len(_cache) > 8:
            _cache.clear()
        _cache[key] = load(path, may_read)
    return _cache[key]


def bounds(M):
    """the model's box in its own space, at rest → (low corner, high corner)"""
    world = {}

    def mat(i):
        if i not in world:
            n = M['nodes'][i]
            q = np.asarray(n['q'], float)
            q = q / (np.linalg.norm(q) or 1.0)
            x, y, z, w = q
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            L = np.eye(4)
            L[:3, :3] = R * np.asarray(n['s'], float)[None]
            L[:3, 3] = n['p']
            world[i] = (mat(n['parent']) @ L) if n['parent'] >= 0 else L
        return world[i]
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for i, n in enumerate(M['nodes']):
        if not M['keep'][i]:
            continue
        for g, _ in n['draws']:
            P = M['meshes'][g][0][:, :3].astype(float) @ mat(i)[:3, :3].T + mat(i)[:3, 3]
            lo, hi = np.minimum(lo, P.min(0)), np.maximum(hi, P.max(0))
    if not np.isfinite(lo).all():
        raise ValueError('в модели нет ни одного треугольника')
    return lo, hi
