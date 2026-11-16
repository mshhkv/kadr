//! doc: "Линза в точке center (px слоя; не задана — точка опоры слоя): strength > 0 выпучивает, < 0 втягивает, в круге radius. AE: Bulge, Optics Compensation, fisheye"
//! params: {center: pivot, radius: 200, strength: 0.5}
//! grow: "0"
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let l = fx_layer_xy(xy);
    let v = l - p.center;
    let r = length(v) / max(p.radius, 1e-3);
    if (r >= 1.0) { return src_at(xy); }
    let k = 1.0 - p.strength * (1.0 - r * r);                  // the picture seen at this point comes from nearer the centre
    return src_edge(xy + fx_px(v * (k - 1.0), xy));
}
