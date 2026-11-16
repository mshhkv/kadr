//! doc: "Засветка плёнки: два мягких цветных пятна медленно плывут от краёв рамки слоя (speed — скорость, seed — другой рисунок). Ставить с blend: screen или add. AE: Light Leaks"
//! params: {color: "#ff6a2a", color2: "#ffd24a", speed: 0.15, strength: 0.8, seed: 0}
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let uv = fx_uv(xy);
    let t = fx_time() * p.speed + p.seed * 13.7;
    let n1 = fx_fbm(uv * 1.3 + vec2<f32>(t, -t * 0.7));
    let n2 = fx_fbm(uv * 0.9 + vec2<f32>(9.1 - t * 0.6, t));
    let k1 = smoothstep(0.50, 0.95, n1 + (1.0 - uv.x) * 0.45);
    let k2 = smoothstep(0.55, 1.0, n2 + uv.y * 0.35);
    let col = (p.color * k1 + p.color2 * k2 * 0.8) * p.strength;
    let a = clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0);
    return vec4<f32>(min(col, vec3<f32>(a)), a);
}
