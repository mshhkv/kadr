// a prepared picture (rgb, a — already Lanczos-resized to about the size it shows at) placed through a 3×3 matrix:
// each output pixel's centre maps back to content px, samples the picture with Mitchell's bicubic (B = C = 1/3, on
// premultiplied colour, edges clamped) and gets the coverage of the content box's four edges — what Skia's
// drawImageRect with CubicResampler.Mitchell and an anti-aliased paint did (render.py _project / _rotated)
// u: 0 out_w 1 out_h 2 pic_w 3 pic_h   f: 0..8 inverse matrix (raster px → content px, row-major) 9 content w 10 content h
//    11 x0 12 y0 (output origin in raster px)
fn mitchell(x0: f32) -> f32 {
    let x = abs(x0);
    let B = 1.0 / 3.0; let C = 1.0 / 3.0;
    if (x < 1.0) { return ((12.0 - 9.0 * B - 6.0 * C) * x * x * x + (-18.0 + 12.0 * B + 6.0 * C) * x * x + (6.0 - 2.0 * B)) / 6.0; }
    if (x < 2.0) { return ((-B - 6.0 * C) * x * x * x + (6.0 * B + 30.0 * C) * x * x + (-12.0 * B - 48.0 * C) * x + (8.0 * B + 24.0 * C)) / 6.0; }
    return 0.0;
}
// the exact share of a pixel (a unit square around its centre) on the inner side of a straight edge: d — the centre's
// distance inside the edge, n — the edge normal; a box filter of a half-plane in closed form
fn halfplane(d: f32, n: vec2<f32>) -> f32 {
    let a = max(abs(n.x), abs(n.y));
    let b = min(abs(n.x), abs(n.y));
    let w1 = 0.5 * (a - b);
    let w2 = 0.5 * (a + b);
    if (d <= -w2) { return 0.0; }
    if (d >= w2) { return 1.0; }
    if (abs(d) <= w1) { return 0.5 + d / a; }
    let t = w2 - abs(d);
    let small = t * t / (2.0 * a * b);
    return select(small, 1.0 - small, d > 0.0);
}
// a raster px → content px through an inverse matrix (rows m0, m1, m2)
fn back3(x: f32, y: f32, m0: vec3<f32>, m1: vec3<f32>, m2: vec3<f32>) -> vec2<f32> {
    let w = m2.x * x + m2.y * y + m2.z;
    return vec2<f32>((m0.x * x + m0.y * y + m0.z) / w, (m1.x * x + m1.y * y + m1.z) / w);
}
fn tap(x: i32, y: i32) -> vec4<f32> {
    let pw = i32(U(2u)); let ph = i32(U(3u));
    let s = S(u32(clamp(y, 0, ph - 1)) * u32(pw) + u32(clamp(x, 0, pw - 1)));
    return vec4<f32>(s.xyz * s.w, s.w);
}
// the picture (U(2) × U(3)) at the raster point (X, Y) through the inverse matrix: (rgb, a) with the exact coverage of
// the content box 0..cw × 0..ch in a; zero outside. The picture holds the part `crop` (x0, y0, x1, y1 in content px) of
// the content — all of it, or only what can be seen of a picture zoomed far in
fn place_px(X: f32, Y: f32, m0: vec3<f32>, m1: vec3<f32>, m2: vec3<f32>, cw: f32, ch: f32, crop: vec4<f32>) -> vec4<f32> {
    let uv = back3(X, Y, m0, m1, m2);
    // the box's edges in output px: how fast u and v change across a pixel gives each edge's normal and the centre's
    // distance to it; the coverage is exact for a pixel one edge crosses (a corner: the product of two)
    let du = vec2<f32>(back3(X + 0.5, Y, m0, m1, m2).x - back3(X - 0.5, Y, m0, m1, m2).x, back3(X, Y + 0.5, m0, m1, m2).x - back3(X, Y - 0.5, m0, m1, m2).x);
    let dv = vec2<f32>(back3(X + 0.5, Y, m0, m1, m2).y - back3(X - 0.5, Y, m0, m1, m2).y, back3(X, Y + 0.5, m0, m1, m2).y - back3(X, Y - 0.5, m0, m1, m2).y);
    let gu = length(du); let gv = length(dv);
    let nu = du / max(gu, 1e-12); let nv = dv / max(gv, 1e-12);
    let cov = halfplane(uv.x / gu, nu) * halfplane((cw - uv.x) / gu, nu) * halfplane(uv.y / gv, nv) * halfplane((ch - uv.y) / gv, nv);
    if (cov <= 0.0) { return vec4<f32>(0.0); }
    let px = (uv.x - crop.x) * f32(U(2u)) / (crop.z - crop.x) - 0.5;
    let py = (uv.y - crop.y) * f32(U(3u)) / (crop.w - crop.y) - 0.5;
    let ix = i32(floor(px)); let iy = i32(floor(py));
    let fx = px - f32(ix); let fy = py - f32(iy);
    var acc = vec4<f32>(0.0);
    for (var j = -1; j <= 2; j = j + 1) {
        let wy = mitchell(f32(j) - fy);
        for (var i = -1; i <= 2; i = i + 1) {
            acc = acc + wy * mitchell(f32(i) - fx) * tap(ix + i, iy + j);
        }
    }
    let a = clamp(acc.w, 0.0, 1.0);
    let c = clamp(acc.xyz, vec3<f32>(0.0), vec3<f32>(255.0) * a);
    return vec4<f32>(c / max(a, 1e-6), a * cov);
}
