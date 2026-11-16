// one colour with the alpha of a plane (a text's coverage): dst = (F0, F1, F2, clamp(plane, 0, 1))   u: 0 n
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    if (i >= U(0u)) { return; }
    dst[i] = vec4<f32>(F(0u), F(1u), F(2u), clamp(aux[i], 0.0, 1.0));
}
