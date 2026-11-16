// a layer's own picture through its CSS-like filter (render.py apply_grade_f, then clip 0..255), steps in order
// u: 0 n 1 count 2.. op codes (1 contrast 2 saturation 3 sepia 4 brightness 5 gamma)   f: values in the same order
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    if (i >= U(0u)) { return; }
    let s = S(i);
    var x = s.xyz;
    for (var k = 0u; k < U(1u); k = k + 1u) {
        let op = U(2u + k);
        let v = F(k);
        if (op == 1u) { x = (x - 127.5) * v + 127.5; }
        else if (op == 2u) { let l = dot(x, vec3<f32>(0.2126, 0.7152, 0.0722)); x = l + (x - l) * v; }
        else if (op == 3u) {
            let sp = vec3<f32>(dot(x, vec3<f32>(0.393, 0.769, 0.189)), dot(x, vec3<f32>(0.349, 0.686, 0.168)), dot(x, vec3<f32>(0.272, 0.534, 0.131)));
            x = x * (1.0 - v) + sp * v;
        }
        else if (op == 4u) { x = x * v; }
        else if (op == 5u) { x = 255.0 * pow(clamp(x / 255.0, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / v)); }
    }
    dst[i] = vec4<f32>(clamp(x, vec3<f32>(0.0), vec3<f32>(255.0)), s.w);
}
