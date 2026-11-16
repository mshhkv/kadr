// CSS box-shadow spread on a picture's alpha (render.py spread_alpha, the same formula): out = max over the disc of
// radius r of a(x + o) · clamp(r + 0.5 − |o|, 0, 1) — a soft one-pixel rim, so an animated spread moves smoothly;
// shrinking (U(5) = 1) is the same on 1 − a. The output is U(4) px larger on every side; its colour is unused
// u: 0 out_w 1 out_h 2 in_w 3 in_h 4 pad 5 shrink   f: 0 r (px, ≥ 0)
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let ow = U(0u);
    if (g.x >= ow || g.y >= U(1u)) { return; }
    let r = F(0u);
    let n = i32(ceil(r + 0.5));
    let shrink = U(5u) == 1u;
    var m = 0.0;
    for (var oy = -n; oy <= n; oy = oy + 1) {
        for (var ox = -n; ox <= n; ox = ox + 1) {
            let w = clamp(r + 0.5 - length(vec2<f32>(f32(ox), f32(oy))), 0.0, 1.0);
            if (w <= 0.0) { continue; }
            let x = i32(g.x) - i32(U(4u)) + ox;
            let y = i32(g.y) - i32(U(4u)) + oy;
            var a = 0.0;
            if (x >= 0 && y >= 0 && x < i32(U(2u)) && y < i32(U(3u))) { a = S(u32(y) * U(2u) + u32(x)).w; }
            if (shrink) { a = 1.0 - a; }
            m = max(m, a * w);
        }
    }
    if (shrink) { m = 1.0 - m; }
    D(g.y * ow + g.x, vec4<f32>(0.0, 0.0, 0.0, m));
}
