//! doc: "Кривые: значения выхода на входах 0.25, 0.5, 0.75 (между black и white), общая кривая master и отдельно r, g, b. AE: Curves"
//! params: {master: [0.25, 0.5, 0.75], black: 0, white: 1, r: [0.25, 0.5, 0.75], g: [0.25, 0.5, 0.75], b: [0.25, 0.5, 0.75]}
//! grow: "0"
fn seg(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {      // Catmull-Rom between p1 and p2
    return 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t);
}
fn curve(x0: f32, lo: f32, k: vec3<f32>, hi: f32) -> f32 {
    let x = clamp(x0, 0.0, 1.0) * 4.0;
    let i = min(u32(x), 3u);
    let t = x - f32(i);
    var v = array<f32, 7>(2.0 * lo - k.x, lo, k.x, k.y, k.z, hi, 2.0 * hi - k.z);
    return clamp(seg(v[i], v[i + 1u], v[i + 2u], v[i + 3u], t), 0.0, 1.0);
}
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let s = src_at(xy);
    let c = s.rgb / max(s.a, 1e-6);
    let m = vec3<f32>(curve(c.r, p.black, p.master, p.white), curve(c.g, p.black, p.master, p.white), curve(c.b, p.black, p.master, p.white));
    return vec4<f32>(vec3<f32>(curve(m.r, 0.0, p.r, 1.0), curve(m.g, 0.0, p.g, 1.0), curve(m.b, 0.0, p.b, 1.0)) * s.a, s.a);
}
