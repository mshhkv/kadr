// a Skia surface's premultiplied RGBA8 pixels, ss× supersampled → a layer picture: the ss×ss box mean, then
// alpha = A/255 and colour = C / alpha (render.py Renderer._skia_out)   u: 0 w (output) 1 h 2 ss 3 src_w
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let w = U(0u); let ss = U(2u);
    if (g.x >= w || g.y >= U(1u)) { return; }
    var acc = vec4<f32>(0.0);
    for (var j = 0u; j < ss; j = j + 1u) {
        for (var i = 0u; i < ss; i = i + 1u) {
            acc = acc + rgba8(src8[(g.y * ss + j) * U(3u) + g.x * ss + i]);
        }
    }
    let v = acc / f32(ss * ss);
    let al = v.w / 255.0;
    D(g.y * w + g.x, vec4<f32>(v.xyz / max(al, 1e-6), al));
}
