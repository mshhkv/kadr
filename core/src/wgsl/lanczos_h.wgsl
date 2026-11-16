// Pillow's Lanczos resample, horizontal pass, on a premultiplied RGBA8 source: the kernel widens by the scale when
// shrinking; each output is rounded to 8 bits as Pillow does between passes.
// u: 0 iw 1 ih 2 ow 3 row0 4 nrows 5 vpad (the source is read as if padded by its edge pixels, Kadr's PAD) 6 rgb24
//   (the source is packed RGB bytes, opaque — a footage frame as it comes) 7 grade steps 8..13 their codes (as grade.wgsl)
//   f: 0 box_x0 1 scale 2 filterscale 3 support 16..21 grade values — the footage grade applied to each source pixel and
//   rounded to 8 bits before filtering, exactly as render.py apply_grade did to the frame; U(14): the weights' stride
//   in vec4 (lzw.wgsl)
fn byte(k: u32) -> u32 { return (src8[k / 4u] >> (8u * (k % 4u))) & 255u; }
fn texel(x: i32, y: i32) -> vec4<f32> {
    let p = i32(U(5u));
    let sx = u32(clamp(x - p, 0, i32(U(0u)) - 2 * p - 1));
    let sy = u32(clamp(y - p, 0, i32(U(1u)) - 2 * p - 1));
    let w = U(0u) - 2u * U(5u);
    if (U(6u) == 1u) {
        let k = (sy * w + sx) * 3u;
        return vec4<f32>(graded(vec3<f32>(f32(byte(k)), f32(byte(k + 1u)), f32(byte(k + 2u)))), 255.0);
    }
    return rgba8(src8[sy * w + sx]);
}
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
fn sinc(x: f32) -> f32 { if (x == 0.0) { return 1.0; } let a = x * 3.14159265358979; return sin(a) / a; }
fn lanczos(x: f32) -> f32 { if (x > -3.0 && x < 3.0) { return sinc(x) * sinc(x / 3.0); } return 0.0; }
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= U(2u) || g.y >= U(4u)) { return; }
    let iw = i32(U(0u));
    let row = U(3u) + g.y;
    // this column's weights, made once by lzw.wgsl (bound as aux): first index, count, sum, weights
    let b = g.x * U(14u) * 4u;
    let xmin = bitcast<i32>(aux[b]);
    let n = bitcast<i32>(aux[b + 1u]);
    var acc = vec4<f32>(0.0); var sum = 0.0;
    for (var j = 0; j < n; j = j + 1) {
        let k = aux[b + 4u + u32(j)];
        sum = sum + k;
        acc = acc + k * texel(xmin + j, i32(row));
    }
    if (sum != 0.0) { acc = acc / sum; }
    dst[g.y * U(2u) + g.x] = clamp(floor(acc + 0.5), vec4<f32>(0.0), vec4<f32>(255.0));
}
