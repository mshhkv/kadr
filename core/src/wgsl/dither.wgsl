// canvas → 8 bit with Kadr's fixed dither (render.py to_u8): clip(round(c + d)), ties to even like numpy; four pixels
// packed per vec4 (RGBA8 bit patterns), so the read-back is a quarter of the float canvas   u: 0 n
fn px(i: u32) -> u32 {
    if (i >= U(0u)) { return 0u; }
    let c = clamp(round(S(i).xyz + aux[i]), vec3<f32>(0.0), vec3<f32>(255.0));
    return u32(c.x) | (u32(c.y) << 8u) | (u32(c.z) << 16u) | (255u << 24u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let j = g.x + g.y * 65535u * 256u;
    if (j * 4u >= U(0u)) { return; }
    dst[j] = vec4<f32>(bitcast<f32>(px(4u * j)), bitcast<f32>(px(4u * j + 1u)), bitcast<f32>(px(4u * j + 2u)), bitcast<f32>(px(4u * j + 3u)));
}
