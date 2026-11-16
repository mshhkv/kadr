// Shared by every kernel. An image is a flat array of vec4<f32>: for a layer's own picture (rgb 0..255 not
// premultiplied, alpha 0..1) — the pair Kadr's Python renderer passes around; for a canvas (colour accumulated
// premultiplied 0..255, alpha 0..1). src8: a source picture as packed RGBA8, premultiplied (PIL «RGBa»).
struct P { f: array<vec4<f32>, 8>, u: array<vec4<u32>, 4> };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> aux: array<f32>;
@group(0) @binding(4) var<storage, read> src8: array<u32>;
// a picture in half precision (8 bytes a pixel: rg and ba as two packed f16 pairs) — bound here instead of src / dst
// when the buffer holds one (HS / HD, set per pipeline). Layer pictures are half; canvases and accumulators stay f32
@group(0) @binding(5) var<storage, read> src16: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> dst16: array<vec2<u32>>;
override HS: bool = false;
override HD: bool = false;
fn S(i: u32) -> vec4<f32> {
    if (HS) { let v = src16[i]; return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y)); }
    return src[i];
}
fn D(i: u32, v: vec4<f32>) {
    // a colour divided by a near-zero alpha can pass f16's 65504 — infinite there, and ∞ · 0 darkens the edge; clamped,
    // its share of a pixel (colour · alpha) changes by less than 1/100 level
    if (HD) { let c = clamp(v, vec4<f32>(-60000.0), vec4<f32>(60000.0));
              dst16[i] = vec2<u32>(pack2x16float(c.xy), pack2x16float(c.zw)); return; }
    dst[i] = v;
}
fn F(i: u32) -> f32 { return p.f[i / 4u][i % 4u]; }
fn U(i: u32) -> u32 { return p.u[i / 4u][i % 4u]; }
fn cov(x: f32, a0: f32, a1: f32) -> f32 { return clamp(min(x + 1.0, a1) - max(x, a0), 0.0, 1.0); }
fn rgba8(v: u32) -> vec4<f32> {
    return vec4<f32>(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u), f32(v >> 24u));
}
// a producer's output: its own picture (U(13) = 0); or straight into a picture U(14) × U(15) at (F(30), F(31)) —
// U(13) = 1 added premultiplied into a motion-blur accumulator (what `accumulate` does), U(13) = 2 laid over a canvas
// with opacity F(24) and, if U(12) = 1, the clip F(25..28) (what `composite` does for a plain layer), U(13) = 3 the
// same for a shadow (below) — without the picture in between
fn put(x: u32, y: u32, ow: u32, v0: vec4<f32>) {
    if (U(13u) == 0u) { D(y * ow + x, v0); return; }
    // the value exactly as a stored picture would hold it: an or with U(11) — 0 for every producer, unknown to the
    // compiler — stops fast math from folding the producer's last steps into the blend (a level now and then)
    let v = bitcast<vec4<f32>>(bitcast<vec4<u32>>(v0) | vec4<u32>(U(11u)));
    let ax = i32(x) + i32(F(30u)); let ay = i32(y) + i32(F(31u));
    if (ax < 0 || ay < 0 || ax >= i32(U(14u)) || ay >= i32(U(15u))) { return; }
    let i = u32(ay) * U(14u) + u32(ax);
    if (U(13u) == 1u) { dst[i] = dst[i] + vec4<f32>(v.xyz * v.w, v.w); return; }
    var a = v.w;
    var rgb = v.xyz;
    if (U(13u) == 3u) {
        // a shadow: v's alpha in the colour F(16..18); U(9) = 1 — kept only outside the layer itself (CSS box-shadow),
        // the layer's picture bound as src8, U(10) × U(8), at (F(19), F(20)) in this picture
        rgb = vec3<f32>(F(16u), F(17u), F(18u));
        if (U(9u) == 1u) {
            let lx = i32(x) - i32(F(19u)); let ly = i32(y) - i32(F(20u));
            if (lx >= 0 && ly >= 0 && lx < i32(U(10u)) && ly < i32(U(8u))) {
                a = a * (1.0 - bitcast<f32>(src8[(u32(ly) * U(10u) + u32(lx)) * 4u + 3u]));
            }
        }
    }
    var m = a * F(24u);
    if (U(12u) == 1u) { m = m * cov(f32(ay), F(26u), F(28u)) * cov(f32(ax), F(25u), F(27u)); }
    if (m <= 0.0) { return; }
    let d = dst[i];
    dst[i] = vec4<f32>(d.xyz * (1.0 - m) + rgb * m, d.w * (1.0 - m) + m);
}
