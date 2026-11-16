// The 3D scene (scene3d.rs): one sample of the frame — triangles through a camera, lit as three.js r185 lights a
// MeshStandard / MeshPhysical material (the reference the layer was checked against, line by line): GGX with
// multiple-scattering compensation, clearcoat, point / spot / directional / hemisphere / rect-area (LTC) lights, a PCF
// shadow of one spot light, image-based light from a CubeUV-packed prefiltered environment. World space throughout.
struct Light { pos: vec4<f32>, color: vec4<f32>, dir: vec4<f32>, cone: vec4<f32> };   // pos.w kind (0 point, 1 spot, 2 dir); color.w cutoff; dir.w decay; cone: cos, penumbra cos, shadow
struct Hemi { dir: vec4<f32>, sky: vec4<f32>, ground: vec4<f32> };
struct Rect { color: vec4<f32>, pos: vec4<f32>, hw: vec4<f32>, hh: vec4<f32> };
struct G {
    vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>, shadow_vp: mat4x4<f32>,
    cam: vec4<f32>,                 // the eye; w: environment intensity
    shadow: vec4<f32>,              // bias, normal bias, radius, map size
    counts: vec4<u32>,              // lights, hemispheres, rect lights, has environment
    misc: vec4<f32>,                // frame height, max mip of the environment, 1 / its width, 1 / its height
    lights: array<Light, 8>, hemis: array<Hemi, 4>, rects: array<Rect, 2>,
};
struct Inst { model: mat4x4<f32>, nrm: mat4x4<f32>, m: vec4<u32> };        // m: material, receives shadow, _, _
struct Mat { color: vec4<f32>, emissive: vec4<f32>, p: vec4<f32>, q: vec4<f32>, uv: vec4<f32> };   // p: roughness, metalness, clearcoat, its roughness; q: kind, double-sided, has map, opaque; uv: repeat x y, flip y, which pictures it has
@group(0) @binding(0) var<uniform> g: G;
@group(0) @binding(1) var<storage, read> insts: array<Inst>;
@group(0) @binding(2) var<storage, read> mats: array<Mat>;
@group(0) @binding(3) var shadow_map: texture_depth_2d;
@group(0) @binding(4) var shadow_smp: sampler_comparison;
@group(0) @binding(5) var env_map: texture_2d<f32>;
@group(0) @binding(6) var lin: sampler;
@group(0) @binding(7) var ltc_1: texture_2d<f32>;
@group(0) @binding(8) var ltc_2: texture_2d<f32>;
@group(0) @binding(9) var dfg: texture_2d<f32>;
@group(1) @binding(0) var map_tex: texture_2d<f32>;
@group(1) @binding(1) var map_smp: sampler;
@group(1) @binding(2) var mr_tex: texture_2d<f32>;                // green: roughness, blue: metal, red: how much surrounding light reaches it
@group(1) @binding(3) var nrm_tex: texture_2d<f32>;               // relief: a normal map
@group(1) @binding(4) var em_tex: texture_2d<f32>;                // glow

const PI = 3.141592653589793;
struct V { @builtin(position) clip: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, @location(3) @interpolate(flat) inst: u32 };

@vertex fn vs(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, @builtin(instance_index) ii: u32) -> V {
    let it = insts[ii];
    let w = it.model * vec4<f32>(pos, 1.0);
    var o: V;
    o.clip = g.vp * w; o.world = w.xyz; o.uv = uv; o.inst = ii;
    o.normal = normalize((it.nrm * vec4<f32>(normal, 0.0)).xyz);
    return o;
}
// a sprite: a quad that faces the eye, as wide and tall as the node's scale
@vertex fn vs_sprite(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, @builtin(instance_index) ii: u32) -> V {
    let it = insts[ii];
    var mv = g.view * (it.model * vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let sc = vec2<f32>(length(it.model[0].xyz), length(it.model[1].xyz));
    mv = vec4<f32>(mv.xy + pos.xy * sc, mv.zw);
    var o: V;
    o.clip = g.proj * mv; o.world = (it.model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz; o.uv = uv; o.inst = ii; o.normal = vec3<f32>(0.0, 0.0, 1.0);
    return o;
}
@vertex fn vs_shadow(@location(0) pos: vec3<f32>, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> {
    return g.shadow_vp * (insts[ii].model * vec4<f32>(pos, 1.0));
}

fn pow2(x: f32) -> f32 { return x * x; }
fn pow4(x: f32) -> f32 { let y = x * x; return y * y; }
fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn F_Schlick(f0: vec3<f32>, f90: f32, dotVH: f32) -> vec3<f32> {
    let fr = exp2((-5.55473 * dotVH - 6.98316) * dotVH);
    return f0 * (1.0 - fr) + vec3<f32>(f90 * fr);
}
fn V_GGX(alpha: f32, dotNL: f32, dotNV: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = dotNL * sqrt(a2 + (1.0 - a2) * dotNV * dotNV);
    let gl = dotNV * sqrt(a2 + (1.0 - a2) * dotNL * dotNL);
    return 0.5 / max(gv + gl, 1e-6);
}
fn D_GGX(alpha: f32, dotNH: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = dotNH * dotNH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
fn brdf_ggx(l: vec3<f32>, v: vec3<f32>, n: vec3<f32>, f0: vec3<f32>, rough: f32) -> vec3<f32> {
    let alpha = rough * rough;
    let h = normalize(l + v);
    let dotNL = sat(dot(n, l)); let dotNV = sat(dot(n, v)); let dotNH = sat(dot(n, h)); let dotVH = sat(dot(v, h));
    return F_Schlick(f0, 1.0, dotVH) * (V_GGX(alpha, dotNL, dotNV) * D_GGX(alpha, dotNH));
}
fn dfg_at(rough: f32, d: f32) -> vec2<f32> { return textureSampleLevel(dfg, lin, vec2<f32>(rough, d), 0.0).rg; }
fn brdf_ggx_ms(l: vec3<f32>, v: vec3<f32>, n: vec3<f32>, f0: vec3<f32>, rough: f32) -> vec3<f32> {
    let single = brdf_ggx(l, v, n, f0, rough);
    let dv = dfg_at(rough, sat(dot(n, v))); let dl = dfg_at(rough, sat(dot(n, l)));
    let fv = f0 * dv.x + vec3<f32>(dv.y); let fl = f0 * dl.x + vec3<f32>(dl.y);
    let ev = 1.0 - (dv.x + dv.y); let el = 1.0 - (dl.x + dl.y);
    let favg = f0 + (vec3<f32>(1.0) - f0) * 0.047619;
    let fms = fv * fl * favg / (vec3<f32>(1.0) - ev * el * favg + vec3<f32>(1e-6));
    return single + fms * (ev * el);
}
// ---- the environment: three.js PMREM, CubeUV layout
fn cube_face(d: vec3<f32>) -> f32 {
    let a = abs(d);
    if (a.x > a.z) { if (a.x > a.y) { return select(3.0, 0.0, d.x > 0.0); } return select(4.0, 1.0, d.y > 0.0); }
    if (a.z > a.y) { return select(5.0, 2.0, d.z > 0.0); }
    return select(4.0, 1.0, d.y > 0.0);
}
fn cube_uv(d: vec3<f32>, face: f32) -> vec2<f32> {
    var uv: vec2<f32>;
    if (face == 0.0) { uv = vec2<f32>(d.z, d.y) / abs(d.x); }
    else if (face == 1.0) { uv = vec2<f32>(-d.x, -d.z) / abs(d.y); }
    else if (face == 2.0) { uv = vec2<f32>(-d.x, d.y) / abs(d.z); }
    else if (face == 3.0) { uv = vec2<f32>(-d.z, d.y) / abs(d.x); }
    else if (face == 4.0) { uv = vec2<f32>(-d.x, d.z) / abs(d.y); }
    else { uv = vec2<f32>(d.x, d.y) / abs(d.z); }
    return 0.5 * (uv + vec2<f32>(1.0));
}
fn bilinear_cube(d: vec3<f32>, mip0: f32) -> vec3<f32> {
    var face = cube_face(d);
    let filter_int = max(4.0 - mip0, 0.0);
    let mip = max(mip0, 4.0);
    let fs = exp2(mip);
    var uv = cube_uv(d, face) * (fs - 2.0) + vec2<f32>(1.0);
    if (face > 2.0) { uv.y += fs; face -= 3.0; }
    uv.x += face * fs;
    uv.x += filter_int * 3.0 * 16.0;
    uv.y += 4.0 * (exp2(g.misc.y) - fs);
    return textureSampleLevel(env_map, lin, vec2<f32>(uv.x * g.misc.z, uv.y * g.misc.w), 0.0).rgb;
}
fn rough_to_mip(r: f32) -> f32 {
    if (r >= 0.8) { return (1.0 - r) * (-1.0 + 2.0) / (1.0 - 0.8) - 2.0; }
    if (r >= 0.4) { return (0.8 - r) * (2.0 + 1.0) / (0.8 - 0.4) - 1.0; }
    if (r >= 0.305) { return (0.4 - r) * (3.0 - 2.0) / (0.4 - 0.305) + 2.0; }
    if (r >= 0.21) { return (0.305 - r) * (4.0 - 3.0) / (0.305 - 0.21) + 3.0; }
    return -2.0 * log2(1.16 * r);
}
fn env_at(d: vec3<f32>, rough: f32) -> vec3<f32> {
    let mip = clamp(rough_to_mip(rough), -2.0, g.misc.y);
    let mf = fract(mip); let mi = floor(mip);
    let c0 = bilinear_cube(d, mi);
    if (mf == 0.0) { return c0; }
    return mix(c0, bilinear_cube(d, mi + 1.0), mf);
}
// ---- a rect-area light by linearly transformed cosines
fn ltc_uv(n: vec3<f32>, v: vec3<f32>, rough: f32) -> vec2<f32> {
    return vec2<f32>(rough, sqrt(1.0 - sat(dot(n, v)))) * (63.0 / 64.0) + vec2<f32>(0.5 / 64.0);
}
fn ltc_edge(v1: vec3<f32>, v2: vec3<f32>) -> vec3<f32> {
    let x = dot(v1, v2); let y = abs(x);
    let a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    let b = 3.4175940 + (4.1616724 + y) * y;
    let v = a / b;
    var ts = v;
    if (x <= 0.0) { ts = 0.5 * inverseSqrt(max(1.0 - x * x, 1e-7)) - v; }
    return cross(v1, v2) * ts;
}
fn ltc_eval(n: vec3<f32>, v: vec3<f32>, p: vec3<f32>, minv: mat3x3<f32>, rc: array<vec3<f32>, 4>) -> f32 {
    let v1 = rc[1] - rc[0]; let v2 = rc[3] - rc[0];
    if (dot(cross(v1, v2), p - rc[0]) < 0.0) { return 0.0; }
    let t1 = normalize(v - n * dot(v, n));
    let t2 = -cross(n, t1);
    let m = minv * transpose(mat3x3<f32>(t1, t2, n));
    let c0 = normalize(m * (rc[0] - p)); let c1 = normalize(m * (rc[1] - p)); let c2 = normalize(m * (rc[2] - p)); let c3 = normalize(m * (rc[3] - p));
    let f = ltc_edge(c0, c1) + ltc_edge(c1, c2) + ltc_edge(c2, c3) + ltc_edge(c3, c0);
    let l = length(f);
    return max((l * l + f.z) / (l + 1.0), 0.0);
}
fn atten(dist: f32, cutoff: f32, decay: f32) -> f32 {
    var f = 1.0 / max(pow(dist, decay), 0.01);
    if (cutoff > 0.0) { f *= pow2(sat(1.0 - pow4(dist / cutoff))); }
    return f;
}
fn vogel(i: f32, phi: f32) -> vec2<f32> {
    let r = sqrt((i + 0.5) / 5.0); let th = i * 2.399963229728653 + phi;
    return vec2<f32>(cos(th), sin(th)) * r;
}
fn shadow_at(world: vec3<f32>, vnormal: vec3<f32>, frag: vec2<f32>) -> f32 {
    let sc4 = g.shadow_vp * vec4<f32>(world + vnormal * g.shadow.y, 1.0);
    let sc = sc4.xyz / sc4.w;
    let uv = vec2<f32>(sc.x * 0.5 + 0.5, sc.y * 0.5 + 0.5);            // three's shadowCoord.xy (v up)
    let z = sc.z + g.shadow.x;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || z > 1.0) { return 1.0; }
    let radius = g.shadow.z / g.shadow.w;
    let phi = fract(52.9829189 * fract(dot(frag, vec2<f32>(0.06711056, 0.00583715)))) * 2.0 * PI;
    var s = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let o = uv + vogel(f32(i), phi) * radius;
        s += textureSampleCompareLevel(shadow_map, shadow_smp, vec2<f32>(o.x, 1.0 - o.y), z);
    }
    return s * 0.2;
}

struct Out { @location(0) color: vec4<f32> };
@fragment fn fs(i: V, @builtin(front_facing) front: bool) -> Out {
    let it = insts[i.inst];
    let m = mats[it.m.x];
    // what needs neighbouring pixels is asked before any branch: the texture (every material has one bound — a white
    // pixel when it has no map) and how fast the normal turns across the pixel (in view space, as three.js measures it)
    let tuv = vec2<f32>(i.uv.x * m.uv.x, mix(i.uv.y * m.uv.y, 1.0 - i.uv.y * m.uv.y, m.uv.z));
    let t = textureSample(map_tex, map_smp, tuv);
    let t_mr = textureSample(mr_tex, map_smp, tuv);
    let t_n = textureSample(nrm_tex, map_smp, tuv);
    let t_e = textureSample(em_tex, map_smp, tuv);
    // across the pixel and up it: the frame's rows run downwards here, the formula below was written for rows running up
    let q0 = dpdx(i.world); let q1 = -dpdy(i.world); let st0 = dpdx(tuv); let st1 = -dpdy(tuv);
    let has = u32(m.uv.w + 0.5);                                  // which pictures it has: 1 roughness / metal, 2 relief, 4 glow, 8 surrounding light
    let nv = (g.view * vec4<f32>(normalize(i.normal), 0.0)).xyz;
    let dxy = max(abs(dpdx(nv)), abs(dpdy(nv)));
    var diffuse = m.color;
    if (m.q.z > 0.5) { diffuse *= t; }
    var o: Out;
    if (m.q.x > 0.5) {                                            // basic and sprite: the colour itself, unlit
        o.color = vec4<f32>(diffuse.rgb, select(diffuse.a, 1.0, m.q.w > 0.5));
        return o;
    }
    var n = normalize(i.normal);
    if (m.q.y > 0.5 && !front) { n = -n; }
    if ((has & 2u) != 0u) {
        // relief: the map's normal stood on the surface. The surface's own two directions come from how the picture
        // lies across the pixel (three.js: perturbNormal2Arb); the map's green points up the picture, while v runs down it
        var mn = t_n.xyz * 2.0 - vec3<f32>(1.0);
        mn.y = mix(-mn.y, mn.y, m.uv.z);
        let q1p = cross(q1, n); let q0p = cross(n, q0);
        let tt = q1p * st0.x + q0p * st1.x;
        let bb = q1p * st0.y + q0p * st1.y;
        let det = max(dot(tt, tt), dot(bb, bb));
        if (det > 0.0) {
            let sc = inverseSqrt(det);
            n = normalize(tt * (mn.x * sc) + bb * (mn.y * sc) + n * mn.z);
        }
    }
    let v = normalize(g.cam.xyz - i.world);
    let metal = m.p.y * select(1.0, t_mr.b, (has & 1u) != 0u);
    let ao = select(1.0, t_mr.r, (has & 8u) != 0u);
    let diff_c = diffuse.rgb * (1.0 - metal);
    let f0 = mix(vec3<f32>(0.04), diffuse.rgb, metal);
    let geo_r = max(max(dxy.x, dxy.y), dxy.z);
    let rough = min(max(m.p.x * select(1.0, t_mr.g, (has & 1u) != 0u), 0.0525) + geo_r, 1.0);
    let cc = sat(m.p.z);
    let cc_r = min(max(m.p.w, 0.0525) + geo_r, 1.0);
    var dd = vec3<f32>(0.0); var ds = vec3<f32>(0.0); var cs = vec3<f32>(0.0);
    let frag = vec2<f32>(i.clip.x, g.misc.x - i.clip.y);          // gl_FragCoord: y up
    for (var k = 0u; k < g.counts.x; k = k + 1u) {
        let L = g.lights[k];
        var l: vec3<f32>; var col = L.color.rgb;
        if (L.pos.w > 1.5) { l = L.dir.xyz; }
        else {
            let lv = L.pos.xyz - i.world;
            let dist = length(lv);
            l = lv / dist;
            if (L.pos.w > 0.5) {
                let sa = smoothstep(L.cone.x, L.cone.y, dot(l, L.dir.xyz));
                if (sa <= 0.0) { continue; }
                col *= sa;
            }
            col *= atten(dist, L.color.w, L.dir.w);
        }
        if (L.cone.z > 0.5 && it.m.y == 1u && (col.x != 0.0 || col.y != 0.0 || col.z != 0.0)) { col *= shadow_at(i.world, normalize(i.normal), frag); }
        let dotNL = sat(dot(n, l));
        let irr = col * dotNL;
        if (cc > 0.0) { cs += irr * brdf_ggx(l, v, n, vec3<f32>(0.04), cc_r); }
        ds += irr * brdf_ggx_ms(l, v, n, f0, rough);
        dd += irr * diff_c / PI;
    }
    for (var k = 0u; k < g.counts.z; k = k + 1u) {
        let R = g.rects[k];
        let rc = array<vec3<f32>, 4>(R.pos.xyz + R.hw.xyz - R.hh.xyz, R.pos.xyz - R.hw.xyz - R.hh.xyz, R.pos.xyz - R.hw.xyz + R.hh.xyz, R.pos.xyz + R.hw.xyz + R.hh.xyz);
        let uv = ltc_uv(n, v, rough);
        let t1 = textureSampleLevel(ltc_1, lin, uv, 0.0); let t2 = textureSampleLevel(ltc_2, lin, uv, 0.0);
        let minv = mat3x3<f32>(vec3<f32>(t1.x, 0.0, t1.y), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(t1.z, 0.0, t1.w));
        let fr = f0 * t2.x + (vec3<f32>(1.0) - f0) * t2.y;
        ds += R.color.rgb * fr * ltc_eval(n, v, i.world, minv, rc);
        dd += R.color.rgb * diff_c * ltc_eval(n, v, i.world, mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)), rc);
        if (cc > 0.0) {
            let uc = ltc_uv(n, v, cc_r);
            let c1 = textureSampleLevel(ltc_1, lin, uc, 0.0); let c2 = textureSampleLevel(ltc_2, lin, uc, 0.0);
            let mc = mat3x3<f32>(vec3<f32>(c1.x, 0.0, c1.y), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(c1.z, 0.0, c1.w));
            cs += R.color.rgb * (0.04 * c2.x + 0.96 * c2.y) * ltc_eval(n, v, i.world, mc, rc);
        }
    }
    // indirect: hemispheres (diffuse only) and the environment
    var irr = vec3<f32>(0.0);
    for (var k = 0u; k < g.counts.y; k = k + 1u) {
        let H = g.hemis[k];
        irr += mix(H.ground.rgb, H.sky.rgb, 0.5 * dot(n, H.dir.xyz) + 0.5);
    }
    var ind_d = irr * diff_c / PI;
    var ind_s = vec3<f32>(0.0); var cs_i = vec3<f32>(0.0);
    if (g.counts.w == 1u) {
        let ei = g.cam.w * m.emissive.w;                                           // the surroundings' light, times this body's own share of it
        let ibl_irr = PI * env_at(n, 1.0) * ei;
        let rv = normalize(mix(reflect(-v, n), n, pow4(rough)));
        let radiance = env_at(rv, rough) * ei;
        let dotNV = sat(dot(n, v));
        let fab = dfg_at(rough, dotNV);
        let ess = fab.x + fab.y; let ems = 1.0 - ess;
        let fd = vec3<f32>(0.04) * fab.x + vec3<f32>(fab.y);                       // dielectric
        let favg_d = vec3<f32>(0.04) + vec3<f32>(0.96) * 0.047619;
        let ms_d = fd * favg_d / (vec3<f32>(1.0) - ems * favg_d) * ems;
        let fm = diffuse.rgb * fab.x + vec3<f32>(fab.y);                           // metallic
        let favg_m = diffuse.rgb + (vec3<f32>(1.0) - diffuse.rgb) * 0.047619;
        let ms_m = fm * favg_m / (vec3<f32>(1.0) - ems * favg_m) * ems;
        let single = mix(fd, fm, metal); let multi = mix(ms_d, ms_m, metal);
        let cw = ibl_irr / PI;
        ind_s = radiance * single + multi * cw;
        ind_d += diff_c * (vec3<f32>(1.0) - (fd + ms_d)) * cw;
        if (cc > 0.0) {
            let rc = normalize(mix(reflect(-v, n), n, pow4(cc_r)));
            let fc = dfg_at(cc_r, dotNV);
            cs_i = env_at(rc, cc_r) * ei * (0.04 * fc.x + fc.y);
        }
    }
    var outc = dd + (ind_d + ind_s) * ao + ds + m.emissive.rgb * select(vec3<f32>(1.0), t_e.rgb, (has & 4u) != 0u);
    if (cc > 0.0) {
        let fcc = F_Schlick(vec3<f32>(0.04), 1.0, sat(dot(n, v)));
        outc = outc * (vec3<f32>(1.0) - cc * fcc) + (cs + cs_i) * cc;
    }
    o.color = vec4<f32>(outc, select(diffuse.a, 1.0, m.q.w > 0.5));
    return o;
}
