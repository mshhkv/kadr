//! params: {threshold: 0.6, knee: 0.2, tint: "#ffffff"}
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let s = src_at(xy);
    let c = s.rgb / max(s.a, 1e-6);
    let k = smoothstep(p.threshold, p.threshold + max(p.knee, 1e-4), max(c.r, max(c.g, c.b)));
    return vec4<f32>(c * p.tint, 1.0) * (k * s.a);
}
