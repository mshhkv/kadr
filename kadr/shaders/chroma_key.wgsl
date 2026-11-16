//! doc: "Хромакей: всё цвета key становится прозрачным (tolerance — насколько похожий цвет убирать, softness — ширина плавного края), spill убирает цветной отсвет фона с того, что осталось. AE: Keylight, Color Key"
//! params: {key: "#00d000", tolerance: 0.18, softness: 0.12, spill: 1}
//! grow: "0"
fn chroma(c: vec3<f32>) -> vec2<f32> {                         // colour without its brightness (BT.709 Cb, Cr)
    let y = fx_luma(c);
    return vec2<f32>((c.b - y) / 1.8556, (c.r - y) / 1.5748);
}
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let s = src_at(xy);
    let c = s.rgb / max(s.a, 1e-6);
    let a = smoothstep(p.tolerance, p.tolerance + max(p.softness, 1e-4), distance(chroma(c), chroma(p.key)));
    let kd = p.key - vec3<f32>(fx_luma(p.key));                // the key's colour as a direction away from grey
    let dir = kd / max(length(kd), 1e-6);
    let spill = max(dot(c - vec3<f32>(fx_luma(c)), dir), 0.0);
    let cc = clamp(c - dir * spill * clamp(p.spill, 0.0, 1.0), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(cc, 1.0) * (s.a * a);
}
