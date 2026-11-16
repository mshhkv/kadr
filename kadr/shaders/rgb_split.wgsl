//! doc: "Расхождение цветовых каналов: красный и синий сдвинуты в стороны на amount px слоя под углом angle; radial 1 — сдвиг растёт от центра картинки, как у объектива. AE: хроматическая аберрация, RGB Split"
//! params: {amount: 6, angle: 0, radial: 0}
//! grow: "abs(amount)"
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    var o = fx_rot(vec2<f32>(1.0, 0.0), radians(p.angle)) * p.amount;
    if (p.radial > 0.0) {
        let c = (xy - fx_size() * 0.5) / (0.5 * length(fx_size()));
        o = mix(o, c * p.amount * 2.0, clamp(p.radial, 0.0, 1.0));
    }
    let d = fx_px(o, xy);
    let a = src_at(xy - d); let m = src_at(xy); let b = src_at(xy + d);
    return vec4<f32>(a.r, m.g, b.b, max(m.a, max(a.a, b.a)));
}
