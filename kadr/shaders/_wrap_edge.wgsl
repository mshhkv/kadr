//! map: true
// src: the layer blurred (only its alpha is read); map: the background blurred. The light that may wrap: where the
// blurred layer is already thin, the background's colour
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let thin = 1.0 - clamp(src_at(xy).a, 0.0, 1.0);
    let b = map_here(xy);
    return vec4<f32>(b.rgb * thin, b.a * thin);
}
