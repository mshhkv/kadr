//! doc: "бегущие косые полосы поверх картинки"
//! params: {width: 12, speed: 30, tint: "#ffffff", strength: 0.6}
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let s = src_at(xy);
    let l = fx_layer_xy(xy);
    let k = step(0.5, fract((l.x + l.y - fx_time() * p.speed) / max(2.0 * p.width, 1e-3))) * p.strength;
    return vec4<f32>(mix(s.rgb, p.tint * s.a, k), s.a);
}
