// After Effects' Motion Tile with mirror edges (kadr/effects.py motion_tile) on a frame-sized picture: the tile rectangle
// repeats around itself, each neighbour mirrored   u: 0 w 1 h 2 mirror   f: 0 x0 1 y0 2 x1 3 y1 (tile, raster px, whole)
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let w = U(0u);
    if (g.x >= w || g.y >= U(1u)) { return; }
    let x0 = i32(F(0u)); let y0 = i32(F(1u)); let x1 = i32(F(2u)); let y1 = i32(F(3u));
    let tw = x1 - x0; let th = y1 - y0;
    let x = i32(g.x); let y = i32(g.y);
    // whole-number floor division: the GPU's float division is approximate (528 / 528 came out 0.99999994)
    let i = select((x - x0) / tw, (x - x0 - tw + 1) / tw, x - x0 < 0);
    let j = select((y - y0) / th, (y - y0 - th + 1) / th, y - y0 < 0);
    var out = vec4<f32>(0.0);
    if (i >= -1 && i <= 1 && j >= -1 && j <= 1) {
        var lx = x - x0 - i * tw; var ly = y - y0 - j * th;
        if (i != 0 && U(2u) == 1u) { lx = tw - 1 - lx; }
        if (j != 0 && U(2u) == 1u) { ly = th - 1 - ly; }
        let sx = x0 + lx; let sy = y0 + ly;
        if (sx >= 0 && sy >= 0 && sx < i32(w) && sy < i32(U(1u))) {
            let s = S(u32(sy) * w + u32(sx));
            out = s;
        }
    }
    dst[g.y * w + g.x] = out;
}
