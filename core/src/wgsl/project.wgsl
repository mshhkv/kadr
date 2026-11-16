// one picture through a matrix into its own output (see project_fns.wgsl), or put() into an accumulator / canvas
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let ow = U(0u);
    if (g.x >= ow || g.y >= U(1u)) { return; }
    let v = place_px(f32(g.x) + F(11u) + 0.5, f32(g.y) + F(12u) + 0.5, vec3<f32>(F(0u), F(1u), F(2u)),
                     vec3<f32>(F(3u), F(4u), F(5u)), vec3<f32>(F(6u), F(7u), F(8u)), F(9u), F(10u),
                     vec4<f32>(F(13u), F(14u), F(15u), F(16u)));
    put(g.x, g.y, ow, v);
}
