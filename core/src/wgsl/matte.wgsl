// a track matte from the matte layer drawn on its own canvas (render.py Renderer.matte): its alpha or its brightness,
// or the inverse; written to .x   u: 0 n 1 luma 2 inverted
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    if (i >= U(0u)) { return; }
    let c = S(i);
    var v = c.w;
    if (U(1u) == 1u) { v = dot(c.xyz, vec3<f32>(0.2126, 0.7152, 0.0722)) / 255.0; }
    v = clamp(v, 0.0, 1.0);
    if (U(2u) == 1u) { v = 1.0 - v; }
    dst[i] = vec4<f32>(v, 0.0, 0.0, 0.0);
}
