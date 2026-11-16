// one pass of a Gaussian along the unit vector (F(0), F(1)) — the blur of a squashed or turned layer is two of these
// along its axes (render.py blur_dir, the same sampling): out(x) = Σ_k w_k · src(x + k·u − s), bilinear, zero outside;
// s = (F(2), F(3)) moves the result (a shadow's offset). u: 0 w 1 h 2 R (taps −R…R, weights in aux)
fn texel_z(x: i32, y: i32) -> vec4<f32> {
    if (x < 0 || y < 0 || x >= i32(U(0u)) || y >= i32(U(1u))) { return vec4<f32>(0.0); }
    return S(u32(y) * U(0u) + u32(x));
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let w = U(0u);
    if (g.x >= w || g.y >= U(1u)) { return; }
    let R = i32(U(2u));
    var acc = vec4<f32>(0.0);
    for (var k = -R; k <= R; k = k + 1) {
        let dx = f32(k) * F(0u) - F(2u);
        let dy = f32(k) * F(1u) - F(3u);
        let fx0 = floor(dx); let fy0 = floor(dy);
        let fx = dx - fx0; let fy = dy - fy0;
        let x = i32(g.x) + i32(fx0); let y = i32(g.y) + i32(fy0);
        let v = texel_z(x, y) * ((1.0 - fx) * (1.0 - fy)) + texel_z(x + 1, y) * (fx * (1.0 - fy))
              + texel_z(x, y + 1) * ((1.0 - fx) * fy) + texel_z(x + 1, y + 1) * (fx * fy);
        acc = acc + aux[u32(k + R)] * v;
    }
    D(g.y * w + g.x, acc);
}
