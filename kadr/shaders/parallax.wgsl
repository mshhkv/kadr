//! doc: "Объём из плоской картинки по карте глубины (map: слой, белое — близко; карту делает `kadr depth фото`): близкое сдвигается на shift px слоя, далёкое — в обратную сторону, focus — глубина, которая стоит на месте. Анимировать shift — «камера» плывёт. AE: смещение по глубине, VoluMax"
//! params: {shift: [20, 0], focus: 0.5}
//! grow: "0"
//! map: true
fn depth_at(q: vec2<f32>, focus: f32) -> f32 {
    let m = map_here(q);
    return mix(focus, fx_luma(m.rgb / max(m.a, 1e-6)), m.a);   // where the map is empty: the depth that stands still
}
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    if (!fx_has_map()) { return src_at(xy); }
    let sh = fx_px(p.shift, xy);
    // what shows here came from q = xy − s·shift, where s = depth(q) − focus. Walk s from the nearest depth to the
    // farthest, a step per px of travel: the first place the surface is at least that near is the one seen (a near
    // thing covers a far one; fixed-point guessing tore seams where depth jumps), then the crossing between two steps
    let hi = 1.0 - p.focus; let lo = -p.focus;
    let n = clamp(i32(ceil(length(sh))), 8, 48);
    var s0 = hi;
    var e0 = depth_at(xy - s0 * sh, p.focus) - p.focus - s0;
    var s = lo;
    if (e0 >= 0.0) { s = hi; }
    else {
        for (var i = 1; i <= n; i = i + 1) {
            let s1 = hi + (lo - hi) * f32(i) / f32(n);
            let e1 = depth_at(xy - s1 * sh, p.focus) - p.focus - s1;
            if (e1 >= 0.0) { s = s0 + (s1 - s0) * (-e0) / max(e1 - e0, 1e-6); break; }
            s0 = s1; e0 = e1;
        }
    }
    return src_edge(xy - s * sh);
}
