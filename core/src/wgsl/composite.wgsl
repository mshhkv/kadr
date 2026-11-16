// one layer onto a canvas: m = alpha · opacity · clip coverage · matte; colour through the blend mode, then
// canvas = canvas·(1 − m) + colour·m, alpha likewise (render.py Renderer._blend)
// u: 0 dst_w 1 dst_h 2 src_w 3 src_h 4 mode 5 has_clip 6 has_matte 7 use_color 8 matte_w 9 matte_h 10 matte stride (1 plane, 4 .x of an image)
// f: 0 x0 1 y0 2 opacity 3..6 clip x0 y0 x1 y1 (canvas px) 7..9 colour 10 matte_x0 11 matte_y0 12 > 0: the source is
//    premultiplied (a group's canvas, an accumulator): × F(12) and divided back here, as pm() did as a pass of its own
// the non-separable modes of W3C compositing (hue, saturation, color, luminosity) — render.py _lum, _sat, _set_sat, _set_lum
fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.3, 0.59, 0.11)); }
fn sat(c: vec3<f32>) -> f32 { return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b)); }
fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let lo = min(c.r, min(c.g, c.b)); let hi = max(c.r, max(c.g, c.b));
    if (hi <= lo) { return vec3<f32>(0.0); }
    return (c - vec3<f32>(lo)) * s / (hi - lo);
}
fn set_lum(c0: vec3<f32>, l: f32) -> vec3<f32> {
    var c = c0 + vec3<f32>(l - lum(c0));
    let L = lum(c); let n = min(c.r, min(c.g, c.b)); let x = max(c.r, max(c.g, c.b));
    if (n < 0.0) { c = vec3<f32>(L) + (c - vec3<f32>(L)) * L / max(L - n, 1e-12); }
    if (x > 1.0) { c = vec3<f32>(L) + (c - vec3<f32>(L)) * (1.0 - L) / max(x - L, 1e-12); }
    return c;
}
fn blend(mode: u32, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    if (mode == 1u) { return a * b; }
    if (mode == 2u) { return 1.0 - (1.0 - a) * (1.0 - b); }
    if (mode == 3u) { return select(1.0 - 2.0 * (1.0 - a) * (1.0 - b), 2.0 * a * b, a < vec3<f32>(0.5)); }
    if (mode == 4u) {
        let d = select(sqrt(a), ((16.0 * a - 12.0) * a + 4.0) * a, a <= vec3<f32>(0.25));
        return select(a + (2.0 * b - 1.0) * (d - a), a - (1.0 - 2.0 * b) * a * (1.0 - a), b <= vec3<f32>(0.5));
    }
    if (mode == 5u) { return min(vec3<f32>(1.0), a + b); }
    if (mode == 6u) { return min(a, b); }
    if (mode == 7u) { return max(a, b); }
    if (mode == 8u) { return select(select(min(vec3<f32>(1.0), a / max(vec3<f32>(1.0) - b, vec3<f32>(1e-6))), vec3<f32>(1.0), b >= vec3<f32>(1.0)), vec3<f32>(0.0), a <= vec3<f32>(0.0)); }
    if (mode == 9u) { return select(select(vec3<f32>(1.0) - min(vec3<f32>(1.0), (vec3<f32>(1.0) - a) / max(b, vec3<f32>(1e-6))), vec3<f32>(0.0), b <= vec3<f32>(0.0)), vec3<f32>(1.0), a >= vec3<f32>(1.0)); }
    if (mode == 10u) { return select(1.0 - 2.0 * (1.0 - a) * (1.0 - b), 2.0 * a * b, b <= vec3<f32>(0.5)); }
    if (mode == 11u) { return abs(a - b); }
    if (mode == 12u) { return a + b - 2.0 * a * b; }
    if (mode == 13u) { return max(vec3<f32>(0.0), a - b); }
    if (mode == 14u) { return set_lum(set_sat(b, sat(a)), lum(a)); }
    if (mode == 15u) { return set_lum(set_sat(a, sat(b)), lum(a)); }
    if (mode == 16u) { return set_lum(b, lum(a)); }
    if (mode == 17u) { return set_lum(a, lum(b)); }
    return b;
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= U(2u) || g.y >= U(3u)) { return; }
    let dx = i32(g.x) + i32(F(0u));
    let dy = i32(g.y) + i32(F(1u));
    if (dx < 0 || dy < 0 || dx >= i32(U(0u)) || dy >= i32(U(1u))) { return; }
    var s = S(g.y * U(2u) + g.x);
    if (F(12u) > 0.0) { let v = s * F(12u); s = vec4<f32>(v.xyz / max(v.w, 1e-6), v.w); }
    var m = s.w * F(2u);
    if (U(5u) == 1u) { m = m * cov(f32(dy), F(4u), F(6u)) * cov(f32(dx), F(3u), F(5u)); }
    if (U(6u) == 1u) {
        let mx = dx - i32(F(10u));
        let my = dy - i32(F(11u));
        if (mx < 0 || my < 0 || mx >= i32(U(8u)) || my >= i32(U(9u))) { m = 0.0; }
        else { m = m * aux[(u32(my) * U(8u) + u32(mx)) * U(10u)]; }
    }
    if (m <= 0.0) { return; }
    var rgb = s.xyz;
    if (U(7u) == 1u) { rgb = vec3<f32>(F(7u), F(8u), F(9u)); }
    let i = u32(dy) * U(0u) + u32(dx);
    let d = dst[i];
    if (U(4u) != 0u) { rgb = blend(U(4u), d.xyz / 255.0, rgb / 255.0) * 255.0; }
    dst[i] = vec4<f32>(d.xyz * (1.0 - m) + rgb * m, d.w * (1.0 - m) + m);
}
