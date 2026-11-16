//! doc: "Сдвиг пикселей: по карте (map: слой — красный двигает по x, зелёный по y, серый 0.5 стоит на месте) или по встроенному шуму. AE: Displacement Map, Turbulent Displace, Roughen Edges"
//! params: {amount: [20, 20], size: 80, evolution: 0, speed: 0}
//! grow: "max(abs(amount[0]), abs(amount[1]))"
//! map: true
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    var d: vec2<f32>;
    if (fx_has_map()) {
        let m = map_here(xy);
        let c = m.rgb / max(m.a, 1e-6);
        d = (c.rg - vec2<f32>(0.5)) * 2.0 * m.a;               // where the map is empty nothing moves
    } else {
        let q = fx_layer_xy(xy) / max(p.size, 1e-3);
        let e = p.evolution + p.speed * fx_time();
        d = vec2<f32>(fx_fbm(q + vec2<f32>(e, 0.0)), fx_fbm(q + vec2<f32>(31.7, 12.3 - e))) * 2.0 - vec2<f32>(1.0);
    }
    return src_at(xy - fx_px(d * p.amount, xy));
}
