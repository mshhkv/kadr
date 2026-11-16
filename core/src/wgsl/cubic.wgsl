// Pillow's BICUBIC resize (a = −0.5) of a float picture, one axis — the way back up after the wide blur
// u: 0 in_w 1 in_h 2 out_w 3 out_h 4 axis (0 x, 1 y)   f: 0 scale (in px per out px)
fn cubic(x0: f32) -> f32 {
    let x = abs(x0); let a = -0.5;
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
    if (x < 2.0) { return (((x - 5.0) * x + 8.0) * x - 4.0) * a; }
    return 0.0;
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let ow = U(2u);
    if (g.x >= ow || g.y >= U(3u)) { return; }
    let axis = U(4u);
    let scale = F(0u);
    let fs = max(scale, 1.0);
    let support = 2.0 * fs;
    let o = select(g.x, g.y, axis == 1u);
    let n = i32(select(U(0u), U(1u), axis == 1u));
    let center = (f32(o) + 0.5) * scale;
    let lo = max(i32(floor(center - support + 0.5)), 0);
    let hi = min(i32(floor(center + support + 0.5)), n);
    var acc = vec4<f32>(0.0); var sum = 0.0;
    for (var i = lo; i < hi; i = i + 1) {
        let k = cubic((f32(i) - center + 0.5) / fs);
        sum = sum + k;
        var idx: u32;
        if (axis == 0u) { idx = g.y * U(0u) + u32(i); } else { idx = u32(i) * U(0u) + g.x; }
        acc = acc + k * S(idx);
    }
    if (sum != 0.0) { acc = acc / sum; }
    put(g.x, g.y, ow, acc);
}
