// a gradient in its own box (render.py draw_gradient with `size` / gradient_field): linear along (dx, dy) or elliptical
// around a centre; stops in aux as [pos, r, g, b, a] × n; alpha also carries the box's edge coverage
// u: 0 out_w 1 out_h 2 kind (0 linear 1 radial) 3 n stops 4 premul (colour mixed as colour·alpha, CSS — render.stops_at)
// f: 0 X0 1 Y0 2 w 3 h (box, raster px relative to the output origin) 4 dx 5 dy 6 half 7 cx 8 cy 9 rx 10 ry
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let ow = U(0u);
    if (g.x >= ow || g.y >= U(1u)) { return; }
    let w = F(2u); let h = F(3u);
    let xs = (f32(g.x) + 0.5 - F(0u)) / w;
    let ys = (f32(g.y) + 0.5 - F(1u)) / h;
    var u: f32;
    if (U(2u) == 0u) { u = ((xs - 0.5) * w * F(4u) + (ys - 0.5) * h * F(5u)) / (2.0 * F(6u)) + 0.5; }
    else { u = sqrt(pow((xs - F(7u)) / F(9u), 2.0) + pow((ys - F(8u)) / F(10u), 2.0)); }
    let n = U(3u);
    var c = vec4<f32>(aux[1], aux[2], aux[3], aux[4]);                       // numpy.interp: clamp to the end stops
    if (u >= aux[(n - 1u) * 5u]) { let k = (n - 1u) * 5u; c = vec4<f32>(aux[k + 1u], aux[k + 2u], aux[k + 3u], aux[k + 4u]); }
    else if (u > aux[0]) {
        for (var i = 0u; i + 1u < n; i = i + 1u) {
            let p0 = aux[i * 5u]; let p1 = aux[(i + 1u) * 5u];
            if (u >= p0 && u <= p1) {
                let q = select((u - p0) / (p1 - p0), 0.0, p1 <= p0);
                let a = vec4<f32>(aux[i * 5u + 1u], aux[i * 5u + 2u], aux[i * 5u + 3u], aux[i * 5u + 4u]);
                let b = vec4<f32>(aux[i * 5u + 6u], aux[i * 5u + 7u], aux[i * 5u + 8u], aux[i * 5u + 9u]);
                c = a + (b - a) * q;
                if (U(4u) == 1u && c.w > 1e-6) {
                    c = vec4<f32>((a.xyz * a.w + (b.xyz * b.w - a.xyz * a.w) * q) / c.w, c.w);
                }
                break;
            }
        }
    }
    let m = cov(f32(g.y), F(1u), F(1u) + h) * cov(f32(g.x), F(0u), F(0u) + w);
    dst[g.y * ow + g.x] = vec4<f32>(c.xyz, c.w * m);
}
