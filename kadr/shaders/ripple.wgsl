//! doc: "Круги по воде от точки center (px слоя; не задана — точка опоры слоя): сдвиг вдоль радиуса, затухает к radius (0 — без затухания). AE: Ripple, CC Ripple Pulse"
//! params: {center: pivot, amplitude: 10, wavelength: 60, speed: 1, radius: 0, phase: 0}
//! grow: "abs(amplitude)"
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let v = fx_layer_xy(xy) - p.center;
    let r = length(v);
    var k = sin(6.2831853 * (r / max(p.wavelength, 1e-3) - p.speed * fx_time() + p.phase));
    if (p.radius > 0.0) { k = k * clamp(1.0 - r / p.radius, 0.0, 1.0); }
    return src_at(xy - fx_px(v / max(r, 1e-6) * p.amplitude * k, xy));
}
