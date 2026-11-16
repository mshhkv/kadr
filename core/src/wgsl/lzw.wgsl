// Lanczos weights, once per output column (or row) instead of once per pixel — the very expressions lanczos_h /
// lanczos_v used inline, so the same numbers. Layout per output o, at o · U(3) vec4: [first source index, count, −, −]
// then the raw weights four to a vec4 (the sum is still taken by the resampling pass, in the same order as before).
// One thread per four weights of one output.  u: 0 n (outputs) 1 n_in (source length) 2 K (max taps) 3 stride in vec4
//   f: 0 box start 1 scale 2 filterscale 3 support
fn sinc(x: f32) -> f32 { if (x == 0.0) { return 1.0; } let a = x * 3.14159265358979; return sin(a) / a; }
fn lanczos(x: f32) -> f32 { if (x > -3.0 && x < 3.0) { return sinc(x) * sinc(x / 3.0); } return 0.0; }
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let o = g.y;
    let jq = g.x;
    if (o >= U(0u) || jq + 1u >= U(3u)) { return; }
    let center = F(0u) + (f32(o) + 0.5) * F(1u);
    let ss = 1.0 / F(2u);
    let lo = max(i32(floor(center - F(3u) + 0.5)), 0);
    let hi = min(i32(floor(center + F(3u) + 0.5)), i32(U(1u)));
    let n = min(max(hi - lo, 0), i32(U(2u)));
    let b = o * U(3u);
    if (jq == 0u) { dst[b] = vec4<f32>(bitcast<f32>(lo), bitcast<f32>(n), 0.0, 0.0); }
    var k = vec4<f32>(0.0);
    for (var c = 0; c < 4; c = c + 1) {
        let j = i32(jq) * 4 + c;
        if (j < n) { k[c] = lanczos((f32(lo + j) - center + 0.5) * ss); }
    }
    dst[b + 1u + jq] = k;
}
