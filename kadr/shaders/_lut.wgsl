//! params: {size: 2, amount: 1}
fn lut3(x: u32, y: u32, z: u32, n: u32) -> vec3<f32> {       // .cube order: red runs fastest
    let k = ((z * n + y) * n + x) * 3u;
    return vec3<f32>(fx_data(k), fx_data(k + 1u), fx_data(k + 2u));
}
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let s = src_at(xy);
    let c = clamp(s.rgb / max(s.a, 1e-6), vec3<f32>(0.0), vec3<f32>(1.0));
    let n = u32(p.size);
    let q = c * f32(n - 1u);
    let i = vec3<u32>(floor(q)); let j = min(i + vec3<u32>(1u), vec3<u32>(n - 1u)); let f = q - floor(q);
    let x00 = mix(lut3(i.x, i.y, i.z, n), lut3(j.x, i.y, i.z, n), f.x); let x10 = mix(lut3(i.x, j.y, i.z, n), lut3(j.x, j.y, i.z, n), f.x);
    let x01 = mix(lut3(i.x, i.y, j.z, n), lut3(j.x, i.y, j.z, n), f.x); let x11 = mix(lut3(i.x, j.y, j.z, n), lut3(j.x, j.y, j.z, n), f.x);
    let o = mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
    return vec4<f32>(mix(c, clamp(o, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(p.amount, 0.0, 1.0)) * s.a, s.a);
}
