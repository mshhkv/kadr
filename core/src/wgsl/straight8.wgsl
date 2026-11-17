// canvas (premultiplied colour 0..255, alpha 0..1) → 8-bit RGBA with straight (not premultiplied) colour, the export
// with transparency (cli._rgba): colour = clip(round(c / a)) where a > 1e-5 else 0, alpha = clip(round(a · 255)); ties
// to even like numpy. Four pixels packed per vec4, so the read-back is a quarter of the float canvas   u: 0 n
fn px(i: u32) -> u32 {
    if (i >= U(0u)) { return 0u; }
    let v = S(i);
    var c = vec3<f32>(0.0);
    if (v.w > 1e-5) {
        c = clamp(round(v.xyz / max(v.w, 1e-5)), vec3<f32>(0.0), vec3<f32>(255.0));
    }
    let a = clamp(round(v.w * 255.0), 0.0, 255.0);
    return u32(c.x) | (u32(c.y) << 8u) | (u32(c.z) << 16u) | (u32(a) << 24u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let j = g.x + g.y * 65535u * 256u;
    if (j * 4u >= U(0u)) { return; }
    dst[j] = vec4<f32>(bitcast<f32>(px(4u * j)), bitcast<f32>(px(4u * j + 1u)), bitcast<f32>(px(4u * j + 2u)), bitcast<f32>(px(4u * j + 3u)));
}
