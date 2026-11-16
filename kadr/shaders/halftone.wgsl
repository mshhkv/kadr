//! doc: "Полутоновый растр: точки с шагом cell px слоя под углом angle, тем крупнее, чем темнее место (invert 1 — чем светлее); ink — цвет точки, tint 1 — точки цвета самой картинки; size — множитель размера. AE: CC Ball Action, Halftone"
//! params: {cell: 10, angle: 45, ink: "#111111", tint: 0, size: 1, invert: 0, soft: 1}
//! grow: "0"
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let c = max(p.cell, 1e-3);
    let a = radians(p.angle);
    let l = fx_rot(fx_layer_xy(xy), -a);
    let cell = (floor(l / c) + vec2<f32>(0.5)) * c;
    let s = src_edge(xy + fx_px(fx_rot(cell - l, a), xy));     // the picture at this dot's middle
    let col = s.rgb / max(s.a, 1e-6);
    let lum = fx_luma(col);
    let dark = mix(1.0 - lum, lum, clamp(p.invert, 0.0, 1.0)) * s.a;
    let r = c * 0.7071 * sqrt(clamp(dark, 0.0, 1.0)) * p.size;  // 0.7071: a full dot covers its cell to the corners
    let e = max(p.soft, 1e-3) / max(fx_scale(), 1e-6);
    let k = clamp((r - length(l - cell)) / e + 0.5, 0.0, 1.0);
    return vec4<f32>(mix(p.ink, col, clamp(p.tint, 0.0, 1.0)) * k, k);
}
