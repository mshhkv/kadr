// After the samples of a 3D frame (scene3d.rs): one sample added to the frame's sum; the glow's chain of halvings;
// and the «film look» of the reference renderer — glow, exposure, tint, a filmic curve, vignette, gamma, grain —
// written out as a Kadr layer picture (colour 0..255, alpha 1).
struct P { a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var t0: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: V; o.pos = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0); o.uv = vec2<f32>(xy.x, 1.0 - xy.y); return o;
}
// a.x: the sample's weight
@fragment fn fs_add(i: V) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(t0, smp, i.uv, 0.0);
    return vec4<f32>(min(c.rgb, vec3<f32>(40.0)) * p.a.x, clamp(c.a, 0.0, 1.0) * p.a.x);
}
// a.xy: half a texel of the target; a.z: threshold (0: none)
@fragment fn fs_down(i: V) -> @location(0) vec4<f32> {
    var c = (textureSampleLevel(t0, smp, i.uv + p.a.xy * vec2<f32>(-1.0, -1.0), 0.0).rgb + textureSampleLevel(t0, smp, i.uv + p.a.xy * vec2<f32>(1.0, -1.0), 0.0).rgb
           + textureSampleLevel(t0, smp, i.uv + p.a.xy * vec2<f32>(-1.0, 1.0), 0.0).rgb + textureSampleLevel(t0, smp, i.uv + p.a.xy * vec2<f32>(1.0, 1.0), 0.0).rgb) * 0.25;
    if (p.a.z > 0.0) { let l = max(c.r, max(c.g, c.b)); c *= smoothstep(p.a.z, p.a.z * 2.2, l); }
    return vec4<f32>(c, 1.0);
}
