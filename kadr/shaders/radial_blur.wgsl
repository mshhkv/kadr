//! doc: "Размытие вокруг точки center (px слоя; не задана — точка опоры слоя): zoom — наезд (доля расстояния), spin — закрутка в градусах; reach — на сколько px слоя картинка может вырасти. AE: CC Radial Blur, Radial Fast Blur; анимированный zoom — зум-переход"
//! params: {center: pivot, zoom: 0.2, spin: 0, reach: 0}
//! grow: "reach"
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let l = fx_layer_xy(xy);
    let v = l - p.center;
    if (p.zoom == 0.0 && p.spin == 0.0) { return src_at(xy); }
    var acc = vec4<f32>(0.0);
    // a sample per px of the smear at this pixel (at least 8, at most 192): fewer leaves steps on a strong zoom
    let smear = length(fx_px(v * p.zoom, xy)) + length(fx_px(v, xy)) * abs(radians(p.spin));
    let n = clamp(i32(ceil(smear)), 8, 192);
    for (var i = 0; i < n; i = i + 1) {
        let t = (f32(i) + 0.5) / f32(n) - 0.5;
        let q = fx_rot(v, radians(p.spin) * t) * (1.0 + p.zoom * t);
        acc = acc + src_at(xy + fx_px(q - v, xy));
    }
    return acc / f32(n);
}
