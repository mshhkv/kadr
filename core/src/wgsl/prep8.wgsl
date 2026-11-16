// a picture file made ready as a source on the GPU (render.py load_image / load_plain): its 8-bit RGB or RGBA as
// decoded, graded and rounded to 8 bits (apply_grade), mirrored, edge-padded, premultiplied with Pillow's RGBA → RGBa
// rounding — packed RGBA8, four pixels per vec4, the bytes the CPU path made
// u: 0 w 1 h (decoded) 2 channels 3 or 4 3 flip 4 n (output pixels) 5 pad 7 grade steps 8..13 codes; f: 16..21 values
fn byte(k: u32) -> u32 { return (src8[k / 4u] >> (8u * (k % 4u))) & 255u; }
// a product rounded to f32 on its own before it is added — numpy rounds every step, while Metal fuses a·b + c into
// one rounding (fast math: even through a multiply by 1) and lands on the other side of a .5 now and then. A bit-or
// with U(15) — 0 in this kernel, but the compiler cannot know — leaves it nothing to fuse
fn keep(x: vec3<f32>) -> vec3<f32> { return bitcast<vec3<f32>>(bitcast<vec3<u32>>(x) | vec3<u32>(U(15u))); }
fn graded(c: vec3<f32>) -> vec3<f32> {
    if (U(7u) == 0u) { return c; }
    var x = c;
    for (var k = 0u; k < U(7u); k = k + 1u) {
        let op = U(8u + k);
        let v = F(16u + k);
        if (op == 1u) { x = keep((x - 127.5) * v) + 127.5; }
        else if (op == 2u) { let l = dot(x, vec3<f32>(0.2126, 0.7152, 0.0722)); x = l + keep((x - l) * v); }
        else if (op == 3u) {
            let sp = vec3<f32>(dot(x, vec3<f32>(0.393, 0.769, 0.189)), dot(x, vec3<f32>(0.349, 0.686, 0.168)), dot(x, vec3<f32>(0.272, 0.534, 0.131)));
            x = keep(x * (1.0 - v)) + keep(sp * v);
        }
        else if (op == 4u) { x = x * v; }
        else if (op == 5u) { x = 255.0 * pow(clamp(x / 255.0, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / v)); }
    }
    return clamp(round(x), vec3<f32>(0.0), vec3<f32>(255.0));
}
fn mul255(c: u32, a: u32) -> u32 { let t = c * a + 128u; return ((t >> 8u) + t) >> 8u; }
fn px(i: u32) -> u32 {
    if (i >= U(4u)) { return 0u; }
    let w = U(0u); let pad = U(5u);
    let ow = w + 2u * pad;
    let ox = i % ow; let oy = i / ow;
    var sx = u32(clamp(i32(ox) - i32(pad), 0, i32(w) - 1));
    let sy = u32(clamp(i32(oy) - i32(pad), 0, i32(U(1u)) - 1));
    if (U(3u) == 1u) { sx = w - 1u - sx; }
    let ch = U(2u);
    let k = (sy * w + sx) * ch;
    let c = vec3<u32>(graded(vec3<f32>(f32(byte(k)), f32(byte(k + 1u)), f32(byte(k + 2u)))));
    let a = select(255u, byte(k + 3u), ch == 4u);
    return mul255(c.x, a) | (mul255(c.y, a) << 8u) | (mul255(c.z, a) << 16u) | (a << 24u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let j = g.x + g.y * 65535u * 256u;
    if (j * 4u >= U(4u)) { return; }
    dst[j] = vec4<f32>(bitcast<f32>(px(4u * j)), bitcast<f32>(px(4u * j + 1u)), bitcast<f32>(px(4u * j + 2u)), bitcast<f32>(px(4u * j + 3u)));
}
