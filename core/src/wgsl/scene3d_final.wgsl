// The finished 3D frame into a Kadr picture (see scene3d_post.wgsl). a: w, h, exposure, glow; b: tint rgb, filmic gain;
// c: vignette from, to, depth, grain; d: grain seed, raw (1: the linear light itself, only the glow added), 0, 0
struct P { a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var acc: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var b1: texture_2d<f32>;
@group(0) @binding(4) var b2: texture_2d<f32>;
@group(0) @binding(5) var b3: texture_2d<f32>;
@group(0) @binding(6) var b4: texture_2d<f32>;
@group(0) @binding(7) var b5: texture_2d<f32>;
@group(0) @binding(8) var<storage, read_write> dst: array<vec4<f32>>;
fn aces(x: vec3<f32>) -> vec3<f32> { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0)); }
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = u32(p.a.x); let h = u32(p.a.y);
    if (id.x >= w || id.y >= h) { return; }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(p.a.x, p.a.y);
    let t = textureLoad(acc, vec2<i32>(id.xy), 0);
    var c = t.rgb;
    let bl = textureSampleLevel(b1, smp, uv, 0.0).rgb * 0.28 + textureSampleLevel(b2, smp, uv, 0.0).rgb * 0.26 + textureSampleLevel(b3, smp, uv, 0.0).rgb * 0.22
           + textureSampleLevel(b4, smp, uv, 0.0).rgb * 0.14 + textureSampleLevel(b5, smp, uv, 0.0).rgb * 0.10;
    c += bl * p.a.w;
    if (p.d.y > 0.5) { dst[id.y * w + id.x] = vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0, clamp(t.a, 0.0, 1.0)); return; }
    c *= p.a.z; c *= p.b.rgb; c = aces(c * p.b.w);
    let d = length((uv - 0.5) * vec2<f32>(0.86, 1.0));
    c *= 1.0 - smoothstep(p.c.x, p.c.y, d) * p.c.z;
    c = pow(c, vec3<f32>(1.0 / 2.2));
    let gl = vec2<f32>(uv.x, 1.0 - uv.y) * vec2<f32>(p.a.x, p.a.y) + vec2<f32>(p.d.x);       // the reference counts rows from the bottom
    c += (fract(sin(dot(gl, vec2<f32>(12.9898, 78.233))) * 43758.5453) - 0.5) * p.c.w;
    dst[id.y * w + id.x] = vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0, clamp(t.a, 0.0, 1.0));
}
