// vertical pass on the horizontal pass's rows, then Kadr's raster form: alpha = A/255 times the coverage of the layer's
// box (exact area at its fractional edges), colour = C / alpha (render.py _raster)
// u: 0 ow 1 oh 2 nrows 3 the weights' stride in vec4 (lzw.wgsl)   f: 0 box_y0 (relative to row0) 1 scale 2 filterscale 3 support 4..7 cover box x0 y0 x1 y1 (output px)
fn sinc(x: f32) -> f32 { if (x == 0.0) { return 1.0; } let a = x * 3.14159265358979; return sin(a) / a; }
fn lanczos(x: f32) -> f32 { if (x > -3.0 && x < 3.0) { return sinc(x) * sinc(x / 3.0); } return 0.0; }
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let ow = U(0u);
    if (g.x >= ow || g.y >= U(1u)) { return; }
    // this row's weights, made once by lzw.wgsl (bound as aux)
    let b = g.y * U(3u) * 4u;
    let ymin = bitcast<i32>(aux[b]);
    let n = bitcast<i32>(aux[b + 1u]);
    var acc = vec4<f32>(0.0); var sum = 0.0;
    for (var j = 0; j < n; j = j + 1) {
        let k = aux[b + 4u + u32(j)];
        sum = sum + k;
        acc = acc + k * S(u32(ymin + j) * ow + g.x);
    }
    if (sum != 0.0) { acc = acc / sum; }
    let v = clamp(floor(acc + 0.5), vec4<f32>(0.0), vec4<f32>(255.0));
    let al = v.w / 255.0;
    let m = cov(f32(g.y), F(5u), F(7u)) * cov(f32(g.x), F(4u), F(6u));
    put(g.x, g.y, ow, vec4<f32>(v.xyz / max(al, 1e-6), m * al));
}
