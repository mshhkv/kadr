//! doc: "Мозаика: квадраты cell px слоя, каждый — средний цвет своего места. AE: Mosaic, пикселизация"
//! params: {cell: 16}
//! grow: "0"
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let c = max(p.cell, 1e-3);
    let l = fx_layer_xy(xy);
    let o = (floor(l / c) + vec2<f32>(0.5)) * c - l;            // to the middle of this cell, layer px
    var acc = vec4<f32>(0.0);
    for (var j = 0; j < 4; j = j + 1) {
        for (var i = 0; i < 4; i = i + 1) {
            let q = (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5)) / 4.0 - vec2<f32>(0.5);
            acc = acc + src_edge(xy + fx_px(o + q * c, xy));
        }
    }
    return acc / 16.0;
}
