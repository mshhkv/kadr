//! doc: "Блик объектива: горячее ядро, лучи, кольцо и «призраки» на линии от источника через середину рамки слоя. at — где источник (0..1 в рамке), radius — размер ядра в px слоя. Ставить с blend: screen или add. AE: Lens Flare, Optical Flares"
//! params: {at: [0.3, 0.3], color: "#ffd9a0", radius: 120, rays: 6, ghosts: 4, strength: 1, turn: 0}
fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {
    let l = fx_layer_xy(xy);
    let b = fx_box();
    let src = p.at * b;
    let v = l - src;
    let r = length(v) / max(p.radius, 1e-3);
    var k = exp(-r * r * 4.0) * 1.6 + 0.10 / (r * r * 6.0 + 0.25);          // the core and its soft halo
    let ang = atan2(v.y, v.x) + radians(p.turn);
    k = k + pow(abs(cos(ang * p.rays * 0.5)), 80.0) * exp(-r * 0.8) * 0.9;  // rays
    k = k + smoothstep(0.07, 0.0, abs(r - 1.7)) * 0.10;                     // a thin ring
    var col = p.color * k;
    let axis = b * 0.5 - src;                                               // ghosts: through the middle of the box
    for (var i = 1; i <= i32(p.ghosts); i = i + 1) {
        let fi = f32(i);
        let gp = src + axis * (0.55 + fi * 0.45);
        let gr = p.radius * (0.20 + 0.12 * fx_hash(vec2<f32>(fi, 3.1)) + 0.05 * fi);
        let d = length(l - gp) / gr;
        let tint = vec3<f32>(0.6 + 0.4 * fx_hash(vec2<f32>(fi, 1.0)), 0.6 + 0.4 * fx_hash(vec2<f32>(fi, 2.0)), 0.6 + 0.4 * fx_hash(vec2<f32>(fi, 5.0)));
        col = col + p.color * tint * (smoothstep(1.0, 0.85, d) * 0.10 + smoothstep(0.12, 0.0, abs(d - 0.95)) * 0.07);
    }
    col = col * p.strength;
    let a = clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0);
    return vec4<f32>(min(col, vec3<f32>(a)), a);
}
