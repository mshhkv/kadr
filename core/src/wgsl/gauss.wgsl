// one axis of a Gaussian with a fractional centre (render.py _gauss_direct): out[i] = Σ_k w_k · in[i − k], zero
// outside; the 2R+1 weights (already normalised) come in aux. u: 0 w 1 h 2 axis (0 x, 1 y) 3 R; the input is
// U(4) × U(5) placed at (U(6), U(7)) in the w × h output — zeros around it, a padding that is never made
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let w = U(0u); let h = U(1u);
    if (g.x >= w || g.y >= h) { return; }
    let R = i32(U(3u));
    var acc = vec4<f32>(0.0);
    for (var k = -R; k <= R; k = k + 1) {
        var x = i32(g.x); var y = i32(g.y);
        if (U(2u) == 0u) { x = x - k; } else { y = y - k; }
        x = x - i32(U(6u)); y = y - i32(U(7u));
        if (x >= 0 && y >= 0 && x < i32(U(4u)) && y < i32(U(5u))) {
            acc = acc + aux[u32(k + R)] * S(u32(y) * U(4u) + u32(x));
        }
    }
    put(g.x, g.y, w, acc);
}
