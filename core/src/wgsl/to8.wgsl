// a canvas (premultiplied colour 0..255, alpha 0..1) → a packed RGBA8 premultiplied source picture, four pixels per
// vec4 — render.py draw_comp: clip(round(clip(c, 0, 255), a · 255)), ties to even as numpy.  u: 0 n (pixels)
fn px(i: u32) -> u32 {
    if (i >= U(0u)) { return 0u; }
    let v = S(i);
    let c = clamp(round(clamp(v.xyz, vec3<f32>(0.0), vec3<f32>(255.0))), vec3<f32>(0.0), vec3<f32>(255.0));
    let a = clamp(round(v.w * 255.0), 0.0, 255.0);
    return u32(c.x) | (u32(c.y) << 8u) | (u32(c.z) << 16u) | (u32(a) << 24u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let j = g.x + g.y * 65535u * 256u;
    if (j * 4u >= U(0u)) { return; }
    dst[j] = vec4<f32>(bitcast<f32>(px(4u * j)), bitcast<f32>(px(4u * j + 1u)), bitcast<f32>(px(4u * j + 2u)), bitcast<f32>(px(4u * j + 3u)));
}
