// the q×q mean of a picture padded with zeros to a multiple of q (render.py gauss_shift, the wide-blur shortcut)
// u: 0 in_w 1 in_h 2 out_w 3 out_h 4 q; the input is U(5) × U(6) placed at (U(7), U(8)) in the in_w × in_h
// picture — zeros around it, a padding that is never made
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let ow = U(2u); let q = U(4u);
    if (g.x >= ow || g.y >= U(3u)) { return; }
    var acc = vec4<f32>(0.0);
    for (var j = 0u; j < q; j = j + 1u) {
        let y = g.y * q + j;
        if (y >= U(1u)) { continue; }
        for (var i = 0u; i < q; i = i + 1u) {
            let x = g.x * q + i;
            let sx = i32(x) - i32(U(7u)); let sy = i32(y) - i32(U(8u));
            if (x < U(0u) && sx >= 0 && sy >= 0 && sx < i32(U(5u)) && sy < i32(U(6u))) { acc = acc + S(u32(sy) * U(5u) + u32(sx)); }
        }
    }
    dst[g.y * ow + g.x] = acc / f32(q * q);
}
