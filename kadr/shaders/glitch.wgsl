//! doc: "Сбой сигнала: полосы высотой block px слоя прыгают вбок на shift, каналы расходятся на split; amount — доля сбитых полос, rate — смен в секунду. AE: Glitch, Bad TV, VHS"
//! params: {amount: 0.3, block: 40, shift: 60, split: 6, rate: 12, lines: 0.15, seed: 0}
//! grow: "abs(shift) + abs(split)"
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let l = fx_layer_xy(xy);
    let tick = floor(fx_time() * p.rate) + p.seed * 17.0;
    let band = floor(l.y / max(p.block, 1.0));
    let h = fx_hash(vec2<f32>(band, tick));
    var dx = 0.0;
    if (h < p.amount) { dx = (fx_hash(vec2<f32>(band + 7.3, tick)) * 2.0 - 1.0) * p.shift; }
    let o = fx_px(vec2<f32>(dx, 0.0), xy);
    let s = fx_px(vec2<f32>(p.split * (0.5 + h), 0.0), xy) * select(0.0, 1.0, p.amount > 0.0 || p.split != 0.0);
    let a = src_at(xy - o - s); let m = src_at(xy - o); let b = src_at(xy - o + s);
    var c = vec4<f32>(a.r, m.g, b.b, max(m.a, max(a.a, b.a)));
    let ln = 1.0 - p.lines * step(0.5, fract(l.y * 0.5));      // every other layer px line dimmed
    return vec4<f32>(c.rgb * ln, c.a);
}
