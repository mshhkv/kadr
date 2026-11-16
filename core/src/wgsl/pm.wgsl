// u: 0 n 1 dir — 0: (rgb, a) → (rgb·a, a); 1: (c, a) → (c / max(a, 1e-6), a); 2: × F(0) then like 1 (an average of n)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    if (i >= U(0u)) { return; }
    var v = S(i);
    if (U(1u) == 0u) { D(i, vec4<f32>(v.xyz * v.w, v.w)); return; }
    if (U(1u) == 2u) { v = v * F(0u); }
    D(i, vec4<f32>(v.xyz / max(v.w, 1e-6), v.w));
}
