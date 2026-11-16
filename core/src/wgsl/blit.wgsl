// src (w × h) into dst (W × H) at (x0, y0); mode 0 replaces, 1 adds premultiplied (rgb·a, a) — motion-blur samples,
// 2 keeps a shadow only outside the layer: dst.a ·= 1 − src.a (CSS box-shadow)
// u: 0 W 1 H 2 w 3 h 4 mode   f: 0 x0 1 y0
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= U(2u) || g.y >= U(3u)) { return; }
    let x = i32(g.x) + i32(F(0u));
    let y = i32(g.y) + i32(F(1u));
    if (x < 0 || y < 0 || x >= i32(U(0u)) || y >= i32(U(1u))) { return; }
    let s = S(g.y * U(2u) + g.x);
    let i = u32(y) * U(0u) + u32(x);
    if (U(4u) == 1u) { dst[i] = dst[i] + vec4<f32>(s.xyz * s.w, s.w); }
    else if (U(4u) == 4u) { let c = s.xyz / max(s.w, 1e-6); dst[i] = dst[i] + vec4<f32>(c * s.w, s.w); }
    else if (U(4u) == 2u) { let d = dst[i]; dst[i] = vec4<f32>(d.xyz, d.w * (1.0 - s.w)); }
    else { dst[i] = s; }
}
