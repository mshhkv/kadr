// a moving picture's motion blur in one pass: every sample's placement of the same picture (U(4) inverse matrices,
// 12 floats each in aux) at this pixel, summed premultiplied in sample order — what `put` mode 1 added into an
// accumulator sample by sample — then the average (× F(13) = 1 / samples drawn) back to (rgb, a), as pm mode 2.
// The accumulator is never made: one write per pixel instead of a read and a write per sample.
// u: 0 out_w 1 out_h 2 pic_w 3 pic_h 4 samples 5 one colour (the picture is a plain fill); f: 9 cw 10 ch 11 x0 12 y0 13 1 / samples drawn 14..17 the crop (project_fns)
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let ow = U(0u);
    if (g.x >= ow || g.y >= U(1u)) { return; }
    let X = f32(g.x) + F(11u) + 0.5;
    let Y = f32(g.y) + F(12u) + 0.5;
    var sum = vec4<f32>(0.0);
    let cw = F(9u); let ch = F(10u);
    let crop = vec4<f32>(F(14u), F(15u), F(16u), F(17u));
    for (var s = 0u; s < U(4u); s = s + 1u) {
        let b = s * 12u;
        let m0 = vec3<f32>(aux[b], aux[b + 1u], aux[b + 2u]);
        let m1 = vec3<f32>(aux[b + 3u], aux[b + 4u], aux[b + 5u]);
        let m2 = vec3<f32>(aux[b + 6u], aux[b + 7u], aux[b + 8u]);
        var v0: vec4<f32>;
        if (m2.x == 0.0 && m2.y == 0.0 && m2.z == 1.0) {
            // flat (no perspective): u, v change by the same amount across every pixel — the edge normals and
            // gradients are the matrix itself, nothing to differentiate per pixel
            let uv = vec2<f32>(m0.x * X + m0.y * Y + m0.z, m1.x * X + m1.y * Y + m1.z);
            let du = m0.xy; let dv = m1.xy;
            let gu = length(du); let gv = length(dv);
            let nu = du / max(gu, 1e-12); let nv = dv / max(gv, 1e-12);
            let cov = halfplane(uv.x / gu, nu) * halfplane((cw - uv.x) / gu, nu) * halfplane(uv.y / gv, nv) * halfplane((ch - uv.y) / gv, nv);
            if (cov <= 0.0) { continue; }
            if (U(5u) == 1u) {                                  // one colour (a rectangle): no samples to take
                let c = S(0);
                v0 = vec4<f32>(c.xyz, c.w * cov);
            } else {
                v0 = place_px(X, Y, m0, m1, m2, cw, ch, crop);
            }
        } else {
            v0 = place_px(X, Y, m0, m1, m2, cw, ch, crop);
        }
        let v = bitcast<vec4<f32>>(bitcast<vec4<u32>>(v0) | vec4<u32>(U(15u)));   // as a stored value (see put)
        sum = sum + vec4<f32>(v.xyz * v.w, v.w);
    }
    let v = sum * F(13u);
    D(g.y * ow + g.x, vec4<f32>(v.xyz / max(v.w, 1e-6), v.w));
}
