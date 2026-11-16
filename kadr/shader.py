"""Effects and layers written as GPU shaders (WGSL) — the escape hatch for whatever the built-in kernels do not do:
distortions, glitches, halftones, glows, generated light… An effect is one function:

    //! params: {amount: 20, center: [0.5, 0.5], tint: "#ff3b30"}
    //! grow: amount                 # layer px the picture may spread by (a number or a formula of the params)
    //! map: true                    # takes a second picture: `map: <layer id>` in the document
    fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {       // xy: this pixel's centre in the picture, px
        return src_at(xy + vec2<f32>(p.amount * fx_scale(), 0.0));
    }

It returns premultiplied colour 0..1 (as shaders usually do); the pictures it reads come the same way. The params
become the struct `FxP` (a number → f32, a list of 2–4 → vecN, "#rrggbb" → vec3 0..1, the word pivot → a vec2 point
in layer px that is the layer's own pivot unless the document gives one), animate like any property, and
sizes given in layer px are multiplied by fx_scale(). What a shader may call is in PRELUDE below.

Built-in effects are the files of kadr/shaders (their names are effect types); a film's own shader is
`effects: [{type: shader, src: wobble.wgsl, amount: 12}]` or `code: "…"` in the document. A shader has no numpy twin:
the CPU renderer runs it on the GPU too (kadr_core), so both renderers give the same picture, and a built-in one is
held by its known-answer tests (tests/ae)."""
import hashlib, math, os, re, struct
import numpy as np
import yaml

RESERVED = {'type', 'src', 'code', 'map'}
MAXF = 22                                                      # floats a shader's params may take (32 − 10 of the engine)

PRELUDE = r'''
// ---- what a shader may use (kadr/shader.py). Colours are premultiplied 0..1; xy is in this picture's px ----
fn fx_size() -> vec2<f32> { return vec2<f32>(f32(U(0u)), f32(U(1u))); }      // this picture, px
fn fx_time() -> f32 { return F(0u); }                                        // film seconds
fn fx_ltime() -> f32 { return F(1u); }                                       // seconds since the layer began
fn fx_scale() -> f32 { return F(2u); }                                       // screen px per layer px
fn fx_origin() -> vec2<f32> { return vec2<f32>(F(3u), F(4u)); }              // the frame px of this picture's corner
fn fx_frame() -> vec2<f32> { return vec2<f32>(F(5u), F(6u)); }               // the frame, px
fn fx_has_map() -> bool { return (U(7u) & 2u) != 0u; }
fn fx_data(i: u32) -> f32 { return aux[i]; }                                 // a table handed to the shader (a LUT)
// a layer drawn by a shader (type: shader, flare, …): its box in layer px, and this pixel's place in it 0..1
fn fx_box() -> vec2<f32> { return vec2<f32>(bitcast<f32>(U(2u)), bitcast<f32>(U(3u))); }
// the layer's own content px under the pixel xy (a texture made from it sticks to the layer as it moves)
fn fx_layer_xy(xy: vec2<f32>) -> vec2<f32> {
    let q = xy + fx_origin();
    let w = bitcast<f32>(U(14u)) * q.x + bitcast<f32>(U(15u)) * q.y + 1.0;
    return vec2<f32>((bitcast<f32>(U(8u)) * q.x + bitcast<f32>(U(9u)) * q.y + bitcast<f32>(U(10u))) / w,
                     (bitcast<f32>(U(11u)) * q.x + bitcast<f32>(U(12u)) * q.y + bitcast<f32>(U(13u))) / w);
}
// a move of d layer px as px of this picture at xy — through the layer's turn, scale and perspective
fn fx_px(d: vec2<f32>, xy: vec2<f32>) -> vec2<f32> {
    let o = fx_layer_xy(xy);
    let a = fx_layer_xy(xy + vec2<f32>(1.0, 0.0)) - o;
    let b = fx_layer_xy(xy + vec2<f32>(0.0, 1.0)) - o;
    let det = a.x * b.y - a.y * b.x;
    if (abs(det) < 1e-12) { return d; }
    return vec2<f32>(d.x * b.y - d.y * b.x, a.x * d.y - a.y * d.x) / det;
}
// the layer's picture: one pixel (nothing outside), a point between pixels, the same with the edge held
fn src_px(x: i32, y: i32) -> vec4<f32> {
    let sx = x - i32(U(6u)); let sy = y - i32(U(6u));
    if (sx < 0 || sy < 0 || sx >= i32(U(2u)) || sy >= i32(U(3u))) { return vec4<f32>(0.0); }
    let v = S(u32(sy) * U(2u) + u32(sx));
    return vec4<f32>(v.xyz * (v.w / 255.0), v.w);
}
fn src_at(p: vec2<f32>) -> vec4<f32> {
    let q = p - vec2<f32>(0.5);
    let i = vec2<i32>(floor(q)); let f = q - floor(q);
    return mix(mix(src_px(i.x, i.y), src_px(i.x + 1, i.y), f.x), mix(src_px(i.x, i.y + 1), src_px(i.x + 1, i.y + 1), f.x), f.y);
}
fn src_edge(p: vec2<f32>) -> vec4<f32> {
    let g = f32(U(6u));
    return src_at(clamp(p, vec2<f32>(g + 0.5), vec2<f32>(g + f32(U(2u)) - 0.5, g + f32(U(3u)) - 0.5)));
}
// the second picture (`map:` — another layer drawn into the frame): by frame px, or under this picture's pixel
fn map_px(x0: i32, y0: i32) -> vec4<f32> {
    let x = x0 - i32(F(7u)); let y = y0 - i32(F(8u));          // the map's own corner in the frame (0, 0 for a layer's)
    if (x < 0 || y < 0 || x >= i32(U(4u)) || y >= i32(U(5u))) { return vec4<f32>(0.0); }
    let i = (u32(y) * U(4u) + u32(x)) * 4u;
    return vec4<f32>(aux[i] / 255.0, aux[i + 1u] / 255.0, aux[i + 2u] / 255.0, aux[i + 3u]);
}
fn map_at(p: vec2<f32>) -> vec4<f32> {
    let q = p - vec2<f32>(0.5);
    let i = vec2<i32>(floor(q)); let f = q - floor(q);
    return mix(mix(map_px(i.x, i.y), map_px(i.x + 1, i.y), f.x), mix(map_px(i.x, i.y + 1), map_px(i.x + 1, i.y + 1), f.x), f.y);
}
fn map_here(xy: vec2<f32>) -> vec4<f32> { return map_at(xy + fx_origin()); }
// small tools
fn fx_luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn fx_hash(p: vec2<f32>) -> f32 {                                            // 0..1, the same for the same p
    var q = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    q = q + dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
fn fx_noise(p: vec2<f32>) -> f32 {                                           // smooth value noise 0..1
    let i = floor(p); let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(fx_hash(i), fx_hash(i + vec2<f32>(1.0, 0.0)), u.x),
               mix(fx_hash(i + vec2<f32>(0.0, 1.0)), fx_hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}
fn fx_fbm(p: vec2<f32>) -> f32 {                                             // 4 octaves of it, 0..1
    var v = 0.0; var a = 0.5; var q = p;
    for (var i = 0; i < 4; i = i + 1) { v = v + a * fx_noise(q); q = q * 2.03 + vec2<f32>(17.1, 9.2); a = a * 0.5; }
    return v / 0.9375;
}
fn fx_rot(v: vec2<f32>, a: f32) -> vec2<f32> { let c = cos(a); let s = sin(a); return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y); }
fn fx_uv(xy: vec2<f32>) -> vec2<f32> { return fx_layer_xy(xy) / max(fx_box(), vec2<f32>(1e-6)); }
// how much of this pixel lies inside the layer's box (a shader layer's edge, anti-aliased)
fn fx_box_cover(xy: vec2<f32>) -> f32 {
    let l = fx_layer_xy(xy);
    let a = fx_layer_xy(xy + vec2<f32>(1.0, 0.0)) - l;
    let b = fx_layer_xy(xy + vec2<f32>(0.0, 1.0)) - l;
    let unit = sqrt(max(abs(a.x * b.y - a.y * b.x), 1e-12));     // layer px per screen px
    let d = min(min(l.x, fx_box().x - l.x), min(l.y, fx_box().y - l.y));
    return clamp(d / unit + 0.5, 0.0, 1.0);
}
'''

MAIN = r'''
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let ow = U(0u);
    if (g.x >= ow || g.y >= U(1u)) { return; }
    let xy = vec2<f32>(f32(g.x) + 0.5, f32(g.y) + 0.5);
    var c = effect(xy, fxp());
    if ((U(7u) & 4u) != 0u) { c = c * fx_box_cover(xy); }       // a shader layer: nothing outside its box
    let a = clamp(c.w, 0.0, 1.0);
    D(g.y * ow + g.x, vec4<f32>(clamp(c.xyz, vec3<f32>(0.0), vec3<f32>(a)) / max(a, 1e-6) * 255.0, a));
}
'''

_EXPR = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
_EXPR.update(abs=abs, min=min, max=max)


class ShaderError(Exception):
    pass


def _kind(v):
    """a parameter's WGSL type from its default: ('f', 1) number, ('f', n) list of n, ('c', 3) "#rrggbb" colour"""
    if isinstance(v, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', v):
        return 'c', 3
    if v == 'pivot':                                           # a point in layer px; not given — the layer's own pivot
        return 'p', 2
    if isinstance(v, (int, float)) and not isinstance(v, bool) or isinstance(v, bool):
        return 'f', 1
    if isinstance(v, (list, tuple)) and 2 <= len(v) <= 4 and all(isinstance(x, (int, float)) for x in v):
        return 'f', len(v)
    raise ShaderError(f'параметр шейдера: число, список из 2–4 чисел или "#rrggbb", а не {v!r}')


class Shader:
    """one compiled-on-demand shader: the author's code, its parameters and how far it spreads"""

    def __init__(self, body, params=None, grow=0, takes_map=False, name='shader', doc='', path=None):
        self.name, self.doc, self.path, self.takes_map, self.grow = name, doc, path, bool(takes_map), grow
        self.params = dict(params or {})
        self.kinds = {k: _kind(v) for k, v in self.params.items()}
        n = sum(k[1] for k in self.kinds.values())
        if n > MAXF:
            raise ShaderError(f'шейдер {name}: параметры занимают {n} чисел, помещается {MAXF}')
        fields, load, at = [], [], 10
        for k, (_, m) in self.kinds.items():
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', k):
                raise ShaderError(f'шейдер {name}: имя параметра «{k}» — латиница, цифры и _')
            ty = 'f32' if m == 1 else f'vec{m}<f32>'
            fields.append(f'{k}: {ty}')
            load.append(f'F({at}u)' if m == 1 else f'vec{m}<f32>(' + ', '.join(f'F({at + j}u)' for j in range(m)) + ')')
            at += m
        if not fields:
            fields, load = ['unused: f32'], ['0.0']
        head = PRELUDE + 'struct FxP { ' + ', '.join(fields) + ' }\nfn fxp() -> FxP { return FxP(' + ', '.join(load) + '); }\n'
        self.head_lines = head.count('\n')
        self.code = head + body + '\n' + MAIN
        self.timed = bool(re.search(r'\bfx_l?time\b', body))    # its picture changes with time even when nothing else does
        self.key = name + '-' + hashlib.sha1(self.code.encode()).hexdigest()[:12]

    def defaults(self):
        """the params as a document gives them (a colour as (r, g, b) 0..255, like doc.param makes it)"""
        return {k: (tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) if self.kinds[k][0] == 'c' else None if self.kinds[k][0] == 'p' else v)
                for k, v in self.params.items()}

    def pack(self, pv, pivot=(0.0, 0.0)):
        out = []
        for k, (kind, m) in self.kinds.items():
            v = pv.get(k, self.defaults()[k])
            if kind == 'p' and v is None:
                v = pivot
            v = [float(v)] if m == 1 else [float(x) for x in v]
            if len(v) != m:
                raise ShaderError(f'шейдер {self.name}: параметру «{k}» нужно {m} чисел, а не {len(v)}')
            out += [x / 255 for x in v] if kind == 'c' else v
        return out

    def grow_px(self, pv, scale):
        """how many px the picture grows on every side (the `grow` of the header, in layer px, × scale)"""
        g = self.grow
        if isinstance(g, str):
            env = {k: (pv.get(k, d) if not isinstance(pv.get(k, d), (list, tuple)) else tuple(pv.get(k, d))) for k, d in self.defaults().items()
                   if self.kinds[k][0] != 'p'}
            try:
                from . import safe
                g = eval(safe.compile_formula(g, 'grow'), {'__builtins__': {}, **_EXPR, **env})
            except Exception as e:
                raise ShaderError(f'шейдер {self.name}: grow «{self.grow}» не считается: {e}')
        return max(0, int(math.ceil(float(g) * scale)))

    def compile_error(self):
        """None, or what the WGSL compiler says — with the line numbers of the author's own code"""
        import kadr_core as kc
        try:
            kc.shader(self.key, self.code, 1, 1, [0.0] * 32, [0] * 16, check=True)
            return None
        except RuntimeError as e:
            return self._own_lines(str(e))

    def _own_lines(self, msg):
        import kadr_core as kc
        off = kc.shader_prefix_lines() + self.head_lines

        def fix(m):
            n = int(m.group(1)) - off
            return f'строка {n}:{m.group(2)}' if n > 0 else f'обвязка, строка {m.group(1)}:{m.group(2)}'
        msg = re.sub(r'wgsl:(\d+):(\d+)', fix, msg)
        msg = re.sub(r'^\s*(\d+) │', lambda m: f'{max(0, int(m.group(1)) - off):4d} │', msg, flags=re.M)
        return '\n'.join(l for l in msg.splitlines() if l.strip() and 'Validation Error' not in l and 'Caused by' not in l
                         and 'In Device::' not in l)


def parse(text, name='shader', path=None, extra_params=None):
    """a .wgsl text with its `//!` header (YAML: params, grow, map, doc) → Shader. extra_params: the document's own
    values for a shader without a header (inline `code:`) — they become its params"""
    head, body = [], []
    for line in text.splitlines():
        (head if line.lstrip().startswith('//!') else body).append(line)
    meta = yaml.safe_load('\n'.join(l.lstrip()[3:] for l in head)) or {} if head else {}
    if not isinstance(meta, dict):
        raise ShaderError(f'шейдер {name}: заголовок //! — это YAML-поля (params, grow, map, doc)')
    params = dict(meta.get('params') or {})
    for k, v in (extra_params or {}).items():
        params.setdefault(k, v)
    if not re.search(r'\bfn\s+effect\s*\(', text):
        raise ShaderError(f'шейдер {name}: нет функции fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32>')
    return Shader('\n'.join(body), params, meta.get('grow', 0), meta.get('map', False), name, meta.get('doc', ''), path)


_files = {}


def load(path, name=None):
    """a shader file, read once (again when the file changes)"""
    key = os.path.abspath(path)
    mt = os.path.getmtime(key)
    hit = _files.get(key)
    if hit is None or hit[0] != mt:
        hit = _files[key] = (mt, parse(open(key, encoding='utf-8').read(), name or os.path.splitext(os.path.basename(key))[0], key))
    return hit[1]


_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shaders')


def _folder(d):
    return {os.path.splitext(f)[0]: load(os.path.join(d, f)) for f in sorted(os.listdir(d))
            if f.endswith('.wgsl') and not f.startswith('_')} if os.path.isdir(d) else {}


def builtin():
    """the effects of kadr/shaders: {name: Shader} (files starting with _ are passes of bigger effects)"""
    return _folder(_DIR)


def generators():
    """the layers of kadr/shaders/gen — pictures a shader makes from nothing: {name: Shader}"""
    return _folder(os.path.join(_DIR, 'gen'))


def internal(name):
    """a pass of a many-pass effect: kadr/shaders/_name.wgsl"""
    return load(os.path.join(_DIR, '_' + name + '.wgsl'), name)


def _bits(v):
    return struct.unpack('I', struct.pack('f', float(v)))[0]


def run(sh, im, pv, fx, size=None, box=None, data=None, grow=None, half=True):
    """one pass of the shader: over the layer's picture `im` (a kadr_core Img: rgb, a) grown by its `grow` — or, with
    im None, a new picture of `size` (w, h) px showing the layer box `box` (w, h in layer px: a shader layer).
    fx: render.Fx (scale, times, origin, inv; map — a canvas Img with map_origin, its corner in the frame).
    data: a table (an Img plane) the shader reads with fx_data() instead of a map. → (Img, grow px)"""
    import kadr_core as kc
    g = (sh.grow_px(pv, fx.scale) if grow is None else int(grow)) if im is not None else 0
    ow, oh = (im.w + 2 * g, im.h + 2 * g) if im is not None else (int(size[0]), int(size[1]))
    mp = getattr(fx, 'map', None) if data is None else None
    mo = getattr(fx, 'map_origin', None) or (0.0, 0.0)
    inv = getattr(fx, 'inv', None)                             # raster px → layer px; unknown: the scale alone
    m = np.diag([1 / max(fx.scale, 1e-9), 1 / max(fx.scale, 1e-9), 1.0]) if inv is None else np.asarray(inv, float)
    m = m / m[2, 2] if abs(m[2, 2]) > 1e-12 else np.eye(3)
    frame = getattr(fx, 'frame', None) or (ow, oh)
    sw, sh_ = (im.w, im.h) if im is not None else (_bits(box[0]), _bits(box[1])) if box is not None else (0, 0)
    u = [ow, oh, sw, sh_, mp.w if mp is not None else 0, mp.h if mp is not None else 0,
         g, (1 if im is not None else 0) | (2 if mp is not None else 0) | (4 if box is not None else 0)] + [_bits(v) for v in m.ravel()[:8]]
    f = [float(fx.t), float(fx.lt), float(fx.scale), float(fx.origin[0] - g), float(fx.origin[1] - g), float(frame[0]), float(frame[1]),
         float(mo[0]), float(mo[1]), 0.0]
    try:
        out = kc.shader(sh.key, sh.code, ow, oh, f + sh.pack(pv, getattr(fx, 'pivot', None) or (0.0, 0.0)), u, im,
                        mp if data is None else data, half)
    except RuntimeError as e:
        raise ShaderError(f'шейдер {sh.name}' + (f' ({sh.path})' if sh.path else '') + ':\n' + sh._own_lines(str(e)))
    return out, g


def effect_fn(sh):
    """the callable an effect entry holds for a shader (plugins.EFFECTS style): fn(im, pv, fx) → (Img, grow px)"""
    def fn(im, pv, fx):
        return run(sh, im, pv, fx)
    fn.kadr_gpu, fn.shader, fn.__name__ = True, sh, 'shader_' + sh.name
    fn.kadr_timed, fn.takes_map = sh.timed, sh.takes_map
    return fn


def layer_fn(sh):
    """the callable of a layer a shader draws (plugins.LAYER_TYPES, draw 'gpu'): fn(pv, fx, size, box) → Img"""
    def fn(pv, fx, size, box):
        return run(sh, None, pv, fx, size=size, box=box)[0]
    fn.shader, fn.__name__ = sh, 'shader_layer_' + sh.name
    return fn


def effect_def(sh):
    """(fn, params, grow) — the entry of plugins.EFFECTS for a shader"""
    return effect_fn(sh), sh.defaults(), (lambda p, s, sh=sh: sh.grow_px(p, s))


def apply_numpy(fn, rgb, a, pv, fx, map_px=None):
    """a GPU effect for the CPU renderer: the picture (rgb 0..255, alpha) goes up, the effect runs, the result comes
    back — the same kernels as the GPU renderer's. map_px: a premultiplied canvas (H, W, 4) → (rgb, a, grow px)"""
    import kadr_core as kc
    im = kc.upload(np.ascontiguousarray(np.concatenate([rgb, a], -1), np.float32))
    fx.map = kc.upload(np.ascontiguousarray(map_px, np.float32)) if map_px is not None else None
    got = fn(im, pv, fx)
    out, g = got if isinstance(got, tuple) else (got, 0)
    arr = out.download()
    return arr[..., :3], arr[..., 3:], g
