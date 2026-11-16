//! params: {strength: 1}
//! map: true
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let s = src_at(xy);
    let g = min(map_here(xy).rgb * (p.strength / 3.0), vec3<f32>(1.0));      // the three blurs summed, premultiplied
    let ga = max(g.r, max(g.g, g.b));
    return vec4<f32>(s.rgb + g * (vec3<f32>(1.0) - s.rgb), s.a + ga * (1.0 - s.a));    // screen: light adds, never darkens
}
