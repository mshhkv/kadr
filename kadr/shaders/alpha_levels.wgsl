//! doc: "Порог по прозрачности: ниже low — пусто, выше high — плотно. После blur даёт жидкие слияния (gooey) и сжатие/расширение края. AE: Simple Choker, Matte Choker, Levels по альфе"
//! params: {low: 0.45, high: 0.55}
//! grow: "0"
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let s = src_at(xy);
    let a = smoothstep(p.low, max(p.high, p.low + 1e-4), s.a);
    return vec4<f32>(s.rgb / max(s.a, 1e-6) * a, a);
}
