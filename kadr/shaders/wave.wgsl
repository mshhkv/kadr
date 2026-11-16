//! doc: "Волна бежит по слою: сдвиг поперёк направления angle. AE: Wave Warp; с малой длиной волны и скоростью — марево"
//! params: {amplitude: 12, wavelength: 120, angle: 0, speed: 1, phase: 0}
//! grow: "abs(amplitude)"
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let dir = fx_rot(vec2<f32>(1.0, 0.0), radians(p.angle));
    let l = fx_layer_xy(xy);
    let s = sin(6.2831853 * (dot(l, dir) / max(p.wavelength, 1e-3) - p.speed * fx_time() + p.phase));
    return src_at(xy - fx_px(vec2<f32>(-dir.y, dir.x) * p.amplitude * s, xy));
}
