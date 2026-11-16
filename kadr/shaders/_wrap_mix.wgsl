//! params: {strength: 0.6}
//! map: true
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let s = src_at(xy);
    let m = map_here(xy);                                      // a picture, not a canvas: colour 0..1 and alpha apart
    let w = min(m.rgb * m.a * p.strength, vec3<f32>(1.0)) * s.a;
    return vec4<f32>(s.rgb + w * (vec3<f32>(s.a) - s.rgb), s.a);
}
