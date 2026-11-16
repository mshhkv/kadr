// every pixel = (F0, F1, F2, F3)   u: 0 n
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    if (i >= U(0u)) { return; }
    dst[i] = vec4<f32>(F(0u), F(1u), F(2u), F(3u));
}
