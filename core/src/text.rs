//! Glyph coverage for a laid-out text — the loop of render.py `_glyph_raster`, line for line, in Rust.
//!
//! FreeType itself is the very library Python's freetype-py has loaded (2.13.2): Python hands over the addresses of
//! the four functions used and the byte offsets of the few struct fields read (from freetype-py's own ctypes
//! definitions), and the faces as the same FT_Face handles. So the glyphs are FreeType's exact-area coverage at the
//! same fractional positions as before, bit for bit — only the per-glyph Python, ctypes and numpy work is gone.
use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::os::raw::{c_int, c_long, c_uint, c_void};
use std::sync::OnceLock;

#[repr(C)]
struct FtMatrix {
    xx: c_long,
    xy: c_long,
    yx: c_long,
    yy: c_long,
}

#[repr(C)]
struct FtVector {
    x: c_long,
    y: c_long,
}

type SetCharSize = unsafe extern "C" fn(*mut c_void, c_long, c_long, c_uint, c_uint) -> c_int;
type SetTransform = unsafe extern "C" fn(*mut c_void, *const FtMatrix, *const FtVector);
type LoadGlyph = unsafe extern "C" fn(*mut c_void, c_uint, i32) -> c_int;
type RenderGlyph = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;

struct Ft {
    set_char_size: SetCharSize,
    set_transform: SetTransform,
    load_glyph: LoadGlyph,
    render_glyph: RenderGlyph,
    // byte offsets: FT_FaceRec.glyph; FT_GlyphSlotRec.bitmap, .bitmap_left, .bitmap_top; FT_Bitmap.rows, .width,
    // .pitch, .buffer
    off: [usize; 8],
}

static FT: OnceLock<Ft> = OnceLock::new();

/// FreeType's functions (FT_Set_Char_Size, FT_Set_Transform, FT_Load_Glyph, FT_Render_Glyph) as addresses in the
/// library Python has loaded, and the struct offsets above
#[pyfunction]
pub fn ft_setup(fns: [usize; 4], off: [usize; 8]) -> PyResult<()> {
    if fns.contains(&0) {
        return Err(PyRuntimeError::new_err("нет адреса функции FreeType"));
    }
    let _ = FT.set(unsafe {
        Ft {
            set_char_size: std::mem::transmute::<usize, SetCharSize>(fns[0]),
            set_transform: std::mem::transmute::<usize, SetTransform>(fns[1]),
            load_glyph: std::mem::transmute::<usize, LoadGlyph>(fns[2]),
            render_glyph: std::mem::transmute::<usize, RenderGlyph>(fns[3]),
            off,
        }
    });
    Ok(())
}

/// Python's round(): half to even
fn rnd(x: f64) -> c_long {
    x.round_ties_even() as c_long
}

unsafe fn rd<T: Copy>(p: *const u8, off: usize) -> T {
    unsafe { std::ptr::read_unaligned(p.add(off) as *const T) }
}

/// Coverage planes of a text, one per colour group that got a glyph — (group, (h, w) float32) in the order the groups
/// were first drawn — values min(1, Σ glyph coverage):
/// faces — FT_Face handles of the font chain; glyphs (id with the chain index in its top bits, x, baseline y, char
/// index); group — the colour group of each char index (−1: not drawn, a char outside `sel`); per — a text animator's
/// own move and scale of a char (dx, dy, scale, ux, uy). The rest is render.py `_glyph_raster`'s setup.
#[pyfunction]
#[pyo3(signature = (faces, size26, flags, glyphs, group, ngroups, per, m, base, lx, ly, cx, cy, x0, y0, w, h, fallback))]
#[allow(clippy::too_many_arguments)]
pub fn glyph_cover<'py>(py: Python<'py>, faces: Vec<usize>, size26: c_long, flags: i32, glyphs: Vec<(u32, f64, f64, usize)>,
                        group: Vec<i32>, ngroups: usize, per: HashMap<usize, (f64, f64, f64, f64, f64)>, m: [f64; 4],
                        base: [f64; 4], lx: f64, ly: f64, cx: f64, cy: f64, x0: i64, y0: i64, w: usize, h: usize,
                        fallback: u32) -> PyResult<Vec<(usize, Bound<'py, PyArray2<f32>>)>> {
    let ft = FT.get().ok_or_else(|| PyRuntimeError::new_err("ft_setup не вызван"))?;
    let [m00, m01, m10, m11] = m;
    let mut planes: Vec<Option<Vec<f32>>> = vec![None; ngroups];
    let mut order = Vec::new();                 // groups in the order their first glyph lands (render.py's dict order)
    let mask = (1u32 << fallback) - 1;
    let o = &ft.off;
    for &(g, gx, gy, ci) in &glyphs {
        let grp = *group.get(ci).unwrap_or(&-1);
        if grp < 0 {
            continue;
        }
        let (mut px, mut py) = (lx + gx, ly + gy);
        let (mxx, mxy, myx, myy);
        if let Some(&(dx, dy, gs, ux, uy)) = per.get(&ci) {
            px = lx + ux + gs * (gx - ux) + dx;
            py = ly + uy + gs * (gy - uy) + dy;
            (mxx, mxy, myx, myy) = (base[0] * gs, base[1] * gs, base[2] * gs, base[3] * gs);
        } else {
            (mxx, mxy, myx, myy) = (base[0], base[1], base[2], base[3]);
        }
        let xx = cx + m00 * px + m01 * py;
        let yy = cy + m10 * px + m11 * py;
        let (ix, iy) = (xx.floor(), yy.floor());
        let fi = (g >> fallback) as usize;
        let face = *faces.get(fi).ok_or_else(|| PyRuntimeError::new_err("нет шрифта цепочки"))? as *mut c_void;
        let gid = g & mask;
        let mat = FtMatrix { xx: rnd(mxx * 65536.0), xy: rnd(mxy * 65536.0), yx: rnd(myx * 65536.0), yy: rnd(myy * 65536.0) };
        let vec = FtVector { x: rnd((xx - ix) * 64.0), y: rnd(-(yy - iy) * 64.0) };
        unsafe {
            (ft.set_char_size)(face, size26, 0, 72, 72);
            (ft.set_transform)(face, &mat, &vec);
            if (ft.load_glyph)(face, gid, flags) != 0 {
                return Err(PyRuntimeError::new_err(format!("FreeType не загрузил глиф {gid}")));
            }
            let slot: *mut c_void = rd(face as *const u8, o[0]);
            if (ft.render_glyph)(slot, 0) != 0 {
                return Err(PyRuntimeError::new_err(format!("FreeType не нарисовал глиф {gid}")));
            }
            let sp = slot as *const u8;
            let bm = sp.add(o[1]);
            let rows: c_uint = rd(bm, o[4]);
            let width: c_uint = rd(bm, o[5]);
            let pitch: c_int = rd(bm, o[6]);
            let buf: *const u8 = rd(bm, o[7]);
            let left: c_int = rd(sp, o[2]);
            let top: c_int = rd(sp, o[3]);
            if rows == 0 || width == 0 || buf.is_null() {
                continue;
            }
            // the bitmap's top-left in the raster; FreeType rows go top-down with a positive pitch
            let bx = ix as i64 + left as i64 - x0;
            let by = iy as i64 - top as i64 - y0;
            let (xa, ya) = (bx.max(0), by.max(0));
            let (xb, yb) = ((bx + width as i64).min(w as i64), (by + rows as i64).min(h as i64));
            if xa >= xb || ya >= yb {
                continue;
            }
            if planes[grp as usize].is_none() {
                order.push(grp as usize);
            }
            let plane = planes[grp as usize].get_or_insert_with(|| vec![0f32; w * h]);
            let stride = pitch.unsigned_abs() as usize;
            for y in ya..yb {
                let src = buf.add((y - by) as usize * stride);
                let row = &mut plane[y as usize * w..(y as usize + 1) * w];
                for x in xa..xb {
                    let a = *src.add((x - bx) as usize) as f32 / 255.0;
                    let c = &mut row[x as usize];
                    *c = (*c + a).min(1.0);
                }
            }
        }
    }
    for &f in &faces {                          // as render.py leaves them: no transform
        let id = FtMatrix { xx: 65536, xy: 0, yx: 0, yy: 65536 };
        let z = FtVector { x: 0, y: 0 };
        unsafe { (ft.set_transform)(f as *mut c_void, &id, &z) };
    }
    let mut out = Vec::with_capacity(order.len());
    for k in order {
        let v = planes[k].take().unwrap_or_default();
        out.push((k, numpy::PyArray1::from_vec(py, v).reshape([h, w])?));
    }
    Ok(out)
}
