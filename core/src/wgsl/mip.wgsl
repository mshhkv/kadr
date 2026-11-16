// a picture seen at many sizes at once (a plane to the horizon, an endless tiled texture) — kadr/mip.py `place`, step
// for step: its mip pyramid (src, levels one after another, level 0 U(2) × U(3), U(6) levels) through U(4) inverse
// matrices (12 floats each in aux) — per pixel the footprint's long and short axes in texels, up to MAXN samples along
// the long axis from the level matching the short one (trilinear; Mitchell on level 0 where magnified), the content
// box's edges by exact coverage; the samples' premultiplied mean. One matrix: through put() (a sink may take it).
// u: 0 out_w 1 out_h 2 w0 3 h0 4 matrices 5 wrap (0 clamp, 1 repeat, 2 mirror) 6 levels
// f: 9 cw 10 ch 11 x0 12 y0 13 1 / matrices 14..17 texel map (sx, sy, ox, oy): texel = (content − o) · s
const MAXN: u32 = 8u;
fn mitchell(x0: f32) -> f32 {
    let x = abs(x0);
    let B = 1.0 / 3.0; let C = 1.0 / 3.0;
    if (x < 1.0) { return ((12.0 - 9.0 * B - 6.0 * C) * x * x * x + (-18.0 + 12.0 * B + 6.0 * C) * x * x + (6.0 - 2.0 * B)) / 6.0; }
    if (x < 2.0) { return ((-B - 6.0 * C) * x * x * x + (6.0 * B + 30.0 * C) * x * x + (-12.0 * B - 48.0 * C) * x + (8.0 * B + 24.0 * C)) / 6.0; }
    return 0.0;
}
fn halfplane(d: f32, n: vec2<f32>) -> f32 {
    let a = max(abs(n.x), abs(n.y));
    let b = min(abs(n.x), abs(n.y));
    let w1 = 0.5 * (a - b);
    let w2 = 0.5 * (a + b);
    if (d <= -w2) { return 0.0; }
    if (d >= w2) { return 1.0; }
    if (abs(d) <= w1) { return 0.5 + d / a; }
    let t = w2 - abs(d);
    let small = t * t / (2.0 * a * b);
    return select(small, 1.0 - small, d > 0.0);
}
fn back3(x: f32, y: f32, m0: vec3<f32>, m1: vec3<f32>, m2: vec3<f32>) -> vec2<f32> {
    let w = m2.x * x + m2.y * y + m2.z;
    return vec2<f32>((m0.x * x + m0.y * y + m0.z) / w, (m1.x * x + m1.y * y + m1.z) / w);
}
fn wrapi(i: i32, n: i32) -> i32 {
    let mode = U(5u);
    if (mode == 1u) { return ((i % n) + n) % n; }
    if (mode == 2u) {
        let m = ((i % (2 * n)) + 2 * n) % (2 * n);
        return select(m, 2 * n - 1 - m, m >= n);
    }
    return clamp(i, 0, n - 1);
}
// level l's size and where it starts in src
fn lsize(l: u32) -> vec2<i32> {
    var w = i32(U(2u)); var h = i32(U(3u));
    for (var k = 0u; k < l; k = k + 1u) { w = max(1, (w + 1) / 2); h = max(1, (h + 1) / 2); }
    return vec2<i32>(w, h);
}
fn lstart(l: u32) -> u32 {
    var w = U(2u); var h = U(3u); var o = 0u;
    for (var k = 0u; k < l; k = k + 1u) { o = o + w * h; w = max(1u, (w + 1u) / 2u); h = max(1u, (h + 1u) / 2u); }
    return o;
}
fn texel(o: u32, sz: vec2<i32>, x: i32, y: i32) -> vec4<f32> {
    return S(o + u32(wrapi(y, sz.y)) * u32(sz.x) + u32(wrapi(x, sz.x)));
}
fn bicubic0(x: f32, y: f32) -> vec4<f32> {
    let sz = vec2<i32>(i32(U(2u)), i32(U(3u)));
    let px = x - 0.5; let py = y - 0.5;
    let fx0 = floor(px); let fy0 = floor(py);
    let fx = px - fx0; let fy = py - fy0;
    let ix = i32(fx0); let iy = i32(fy0);
    var acc = vec4<f32>(0.0);
    for (var j = -1; j <= 2; j = j + 1) {
        let wy = mitchell(f32(j) - fy);
        for (var i = -1; i <= 2; i = i + 1) {
            acc = acc + wy * mitchell(f32(i) - fx) * texel(0u, sz, ix + i, iy + j);
        }
    }
    return acc;
}
fn bilinear(l0: u32, x0: f32, y0: f32) -> vec4<f32> {
    let l = min(l0, U(6u) - 1u);
    let s = f32(1u << l);
    let sz = lsize(l); let o = lstart(l);
    let px = x0 / s - 0.5; let py = y0 / s - 0.5;
    let fx0 = floor(px); let fy0 = floor(py);
    let fx = px - fx0; let fy = py - fy0;
    let ix = i32(fx0); let iy = i32(fy0);
    let a = texel(o, sz, ix, iy) * (1.0 - fx) + texel(o, sz, ix + 1, iy) * fx;
    let b = texel(o, sz, ix, iy + 1) * (1.0 - fx) + texel(o, sz, ix + 1, iy + 1) * fx;
    return a * (1.0 - fy) + b * fy;
}
fn sample(x: f32, y: f32, lod: f32) -> vec4<f32> {
    if (lod <= 0.0) { return bicubic0(x, y); }
    if (lod < 1.0) {
        let b = bicubic0(x, y);
        let n = select(b, bilinear(1u, x, y), U(6u) > 1u);
        return b * (1.0 - lod) + n * lod;
    }
    let lo = floor(lod);
    let f = lod - lo;
    let l = u32(lo);
    return bilinear(l, x, y) * (1.0 - f) + bilinear(l + 1u, x, y) * f;
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let ow = U(0u);
    if (g.x >= ow || g.y >= U(1u)) { return; }
    let X = f32(g.x) + F(11u) + 0.5;
    let Y = f32(g.y) + F(12u) + 0.5;
    let cw = F(9u); let ch = F(10u);
    let sx = F(14u); let sy = F(15u); let ox = F(16u); let oy = F(17u);
    var acc = vec4<f32>(0.0);
    for (var s = 0u; s < U(4u); s = s + 1u) {
        let b = s * 12u;
        let m0 = vec3<f32>(aux[b], aux[b + 1u], aux[b + 2u]);
        let m1 = vec3<f32>(aux[b + 3u], aux[b + 4u], aux[b + 5u]);
        let m2 = vec3<f32>(aux[b + 6u], aux[b + 7u], aux[b + 8u]);
        if (m2.x * X + m2.y * Y + m2.z <= 0.0) { continue; }     // past the horizon: behind the eye
        let uv = back3(X, Y, m0, m1, m2);
        let xp = back3(X + 0.5, Y, m0, m1, m2); let xm = back3(X - 0.5, Y, m0, m1, m2);
        let yp = back3(X, Y + 0.5, m0, m1, m2); let ym = back3(X, Y - 0.5, m0, m1, m2);
        let du = vec2<f32>(xp.x - xm.x, yp.x - ym.x);
        let dv = vec2<f32>(xp.y - xm.y, yp.y - ym.y);
        let gu = length(du); let gv = length(dv);
        let nu = du / max(gu, 1e-12); let nv = dv / max(gv, 1e-12);
        let cov = halfplane(uv.x / gu, nu) * halfplane((cw - uv.x) / gu, nu) * halfplane(uv.y / gv, nv) * halfplane((ch - uv.y) / gv, nv);
        if (cov <= 0.0) { continue; }
        // the footprint: singular values of the Jacobian texel / output px, the long axis in texel space
        let J00 = du.x * sx; let J01 = du.y * sx; let J10 = dv.x * sy; let J11 = dv.y * sy;
        let p = J00 * J00 + J01 * J01;
        let q = J00 * J10 + J01 * J11;
        let r = J10 * J10 + J11 * J11;
        let half = 0.5 * (p + r);
        let disc = sqrt(max(0.0, 0.25 * (p - r) * (p - r) + q * q));
        let s1 = sqrt(max(half + disc, 0.0));
        let s2 = sqrt(max(half - disc, 0.0));
        var e = select(select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), p >= r), vec2<f32>(half + disc - r, q), abs(q) > 1e-12);
        e = e / max(length(e), 1e-30);
        let n = clamp(ceil(s1 / max(s2, 1e-6) - 1e-3), 1.0, f32(MAXN));
        let lod = log2(max(max(s1 / n, s2), 1e-6));
        let tx = (uv.x - ox) * sx; let ty = (uv.y - oy) * sy;
        var sum = vec4<f32>(0.0);
        for (var k = 0u; k < MAXN; k = k + 1u) {
            if (f32(k) >= n) { break; }
            let off = ((f32(k) + 0.5) / n - 0.5) * s1;
            sum = sum + sample(tx + off * e.x, ty + off * e.y, lod);
        }
        sum = sum / n;
        let a = clamp(sum.w, 0.0, 1.0);
        let c = clamp(sum.xyz, vec3<f32>(0.0), vec3<f32>(255.0) * a);
        acc = acc + vec4<f32>(c * cov, a * cov);
    }
    let v = acc * F(13u);
    put(g.x, g.y, ow, vec4<f32>(v.xyz / max(v.w, 1e-6), v.w));
}
