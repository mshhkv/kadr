"""Frame renderer. Every layer gets an exact float placement (centre, scale) per frame — nothing is rounded:
   text = FreeType outlines at fractional positions, rasterized by exact area coverage at 1x;
   images and footage = Lanczos with the kernel widened when shrinking, over a fractional source window;
   layer edges, wipes and shadows = exact (sub-pixel) coverage."""
import functools, glob, math, os, re
import numpy as np
import freetype as ft
from PIL import Image
from .doc import Prop, ease
from .media import probe, reader

OWN_FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fonts')      # Inter (SIL OFL 1.1): ships with Kadr
FONT_DIRS = ['/System/Library/Fonts', '/System/Library/Fonts/Supplemental', '/Library/Fonts', os.path.expanduser('~/Library/Fonts'),
             '/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.fonts'), 'C:/Windows/Fonts']
FT_FLAGS = ft.FT_LOAD_NO_HINTING | ft.FT_LOAD_NO_BITMAP
PAD = 16


_MEMO = {}


def memo(fn):
    """placement is a pure function of (layer, t): motion blur, group extents and mattes ask for the same one many
    times a frame — remember it (the documents are not changed while rendering)"""
    def wrapped(film, L, t):
        key = (fn.__name__, id(L), t)
        hit = _MEMO.get(key)
        if hit is None:                                    # the entry keeps L alive, so its id is not reused meanwhile
            if len(_MEMO) > 200_000:
                _MEMO.clear()
            hit = _MEMO[key] = (L, fn(film, L, t))
        return hit[1]
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped


# ---------------- fonts and text layout ----------------
@functools.lru_cache(None)
def font_file(name):
    if os.path.exists(name):
        return name
    want = name.lower().replace(' ', '')
    for d in [OWN_FONTS] + FONT_DIRS:
        for f in glob.glob(os.path.join(d, '**', '*.*'), recursive=True):
            base, ext = os.path.splitext(os.path.basename(f))
            if ext.lower() in ('.ttf', '.otf', '.ttc') and base.lower().replace(' ', '') == want:
                return f
    if want in ('arial', 'arialbold', 'arial-bold', 'inter', 'interbold'):      # the default where the system has no Arial (Linux; Windows names its files otherwise)
        return os.path.join(OWN_FONTS, 'Inter-Bold.ttf' if 'bold' in want else 'Inter-Regular.ttf')
    raise FileNotFoundError(f'шрифт «{name}» не найден (укажите путь к .ttf/.otf; с Kadr идут Inter-Regular и Inter-Bold)')


@functools.lru_cache(None)
def face(name):
    """«font#wght=700,opsz=24»: a variable font at those axis values (unnamed axes keep their defaults)"""
    path, _, axes = str(name).partition('#')
    f = ft.Face(font_file(path))
    if axes and f.has_multiple_masters:                        # a static font ignores axis values
        want = {k.strip(): float(v) for k, v in (a.split('=') for a in axes.split(','))}
        f.set_var_design_coords([want.get(ax.tag, ax.default) for ax in f.get_variation_info().axes])
    return f


FALLBACK = 24                       # a glyph from the n-th font of a fallback chain has id n << 24 | glyph


def glyph_face(fontname, g):
    """the FreeType face and glyph id of a laid-out glyph (fonts «A.ttf|B.ttf»: the first that has the character)"""
    return face(str(fontname).split('|')[g >> FALLBACK]), g & ((1 << FALLBACK) - 1)


@functools.lru_cache(None)
def hb_font(name):
    """the HarfBuzz font for a face name (same «path#axis=value» form), scaled to font units"""
    import uharfbuzz as hb
    path, _, axes = str(name).partition('#')
    hf = hb.Face(hb.Blob.from_file_path(font_file(path)))
    font = hb.Font(hf)
    font.scale = (hf.upem, hf.upem)
    if axes:
        font.set_variations({k.strip(): float(v) for k, v in (a.split('=') for a in axes.split(','))})
    return font


def feature_set(features, tracking):
    """«tnum,lnum,-liga» → HarfBuzz features; letter-spacing turns optional ligatures off, as Chrome does"""
    out = {'liga': False, 'clig': False, 'dlig': False, 'hlig': False} if tracking else {}
    for x in filter(None, (s.strip() for s in str(features or '').split(','))):
        out[x.lstrip('-+')] = not x.startswith('-')
    return out


@functools.lru_cache(1024)
def text_units(font, size, text, align, line_height, tracking, features, by):
    """a text animator's units — (char indices, x0, x1, baseline y): letters, or words — and the font's descent; they
    depend only on the layout, so they are found once, not every frame"""
    glyphs, _ = text_layout(font, size, text, align, line_height, tracking, features)
    f0 = face(str(font).split('|')[0])
    desc = -f0.descender / f0.units_per_EM * size
    units = []
    if by == 'letters':
        for j, (g, x, y, ci) in enumerate(glyphs):
            if text[ci].isspace():
                continue
            nx = next((x2 for _, x2, y2, _ in glyphs[j + 1:] if abs(y2 - y) < 1e-6), None)
            if nx is None:
                gf, gi = glyph_face(font, g)
                gf.set_char_size(int(round(size * 64)))
                gf.load_glyph(gi, FT_FLAGS)
                nx = x + gf.glyph.linearHoriAdvance / 65536
            units.append((frozenset({ci}), x, nx, y))
    else:
        cur = None
        for g, x, y, ci in glyphs:
            if text[ci].isspace():
                cur = None
                continue
            if cur is None or abs(cur[3] - y) > 1e-6:
                cur = [set(), x, x, y]
                units.append(cur)
            cur[0].add(ci)
            cur[2] = x + size * 0.6
        units = [(frozenset(u[0]), u[1], u[2], u[3]) for u in units]
    return desc, tuple(units)


def glyph_pixels(bm):
    """a FreeType bitmap as float coverage, read straight from FreeType's memory (freetype-py's .buffer builds a
    Python list of every byte — it was a quarter of the text time)"""
    import ctypes
    raw = bm._FT_Bitmap
    n = bm.rows * abs(bm.pitch)
    buf = np.ctypeslib.as_array(ctypes.cast(raw.buffer, ctypes.POINTER(ctypes.c_ubyte)), shape=(n,))
    return buf.reshape(bm.rows, abs(bm.pitch))[:, :bm.width].astype(np.float32) / 255


@functools.lru_cache(4096)
def text_layout(fontname, size, text, align, line_height, tracking=0.0, features=''):
    """glyphs (id, x, baseline_y, char_index) in layer px at scale 1 and the box (w, h).
       HarfBuzz shapes each line the way Chrome does: GPOS kerning, marks, OpenType features (`features`: «tnum,lnum,
       -liga»). Advances are design units, so the layout is exactly proportional at every scale.
       «A.ttf|B.ttf»: a character missing from A is taken from B (line metrics are A's)"""
    import uharfbuzz as hb
    names = str(fontname).split('|')
    chain = [face(n) for n in names]
    f = chain[0]
    upem = f.units_per_EM
    asc, desc = f.ascender / upem * size, -f.descender / upem * size
    feats = feature_set(features, tracking)
    lines = text.split('\n')
    runs, widths, ci = [], [], 0
    for line in lines:
        pen, run = 0.0, []
        pick = [next((i for i, x in enumerate(chain) if x.get_char_index(ch)), 0) for ch in line]
        s = 0
        while s < len(line):                                  # runs of characters drawn by the same font of the chain
            e = s
            while e < len(line) and pick[e] == pick[s]:
                e += 1
            fi = pick[s]
            k = size / chain[fi].units_per_EM
            buf = hb.Buffer()
            buf.add_codepoints([ord(c) for c in line[s:e]])
            buf.guess_segment_properties()
            hb.shape(hb_font(names[fi]), buf, feats)
            infos, pos = buf.glyph_infos, buf.glyph_positions
            for j, (gi, gp) in enumerate(zip(infos, pos)):
                run.append((fi << FALLBACK | gi.codepoint, pen + gp.x_offset * k, -gp.y_offset * k, ci + s + gi.cluster))
                pen += gp.x_advance * k
                nxt = infos[j + 1].cluster if j + 1 < len(infos) else e - s
                if nxt != gi.cluster:                         # CSS letter-spacing: after every character of the cluster
                    pen += tracking * abs(nxt - gi.cluster)
            s = e
        runs.append(run)
        widths.append(pen)
        ci += len(line) + 1                                   # the '\n'
    w = max(widths) if widths else 0.0
    lh = line_height * size
    h = asc + desc + lh * (len(lines) - 1)
    glyphs = []
    for i, (run, lw) in enumerate(zip(runs, widths)):
        x0 = {'left': 0.0, 'center': (w - lw) / 2, 'right': w - lw}[align]
        glyphs += [(g, x0 + px, asc + i * lh + dy, c) for g, px, dy, c in run]
    return tuple(glyphs), (w, h)


# ---------------- rasters ----------------
SEPIA = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], np.float32)


def apply_grade(rgb, grade):
    """CSS-filter-like grade on uint8 RGB, applied to the source before resampling (contrast, saturation, sepia, brightness).
    A big picture (a footage frame) looks every colour up in the grade's table of all 2²⁴ colours — the same bytes as
    computing it, ~8× faster; the table is built once per grade"""
    if not grade:
        return rgb
    if rgb.size < 3 * 300_000:
        return grade_direct(rgb, grade)
    idx = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    return grade_lut(grade)[idx]


@functools.lru_cache(3)
def grade_lut(grade):
    v = np.arange(1 << 24, dtype=np.uint32)
    out = np.empty((1 << 24, 3), np.uint8)
    for i in range(0, 1 << 24, 1 << 20):
        c = v[i:i + (1 << 20)]
        out[i:i + (1 << 20)] = grade_direct(np.stack([(c >> 16) & 255, (c >> 8) & 255, c & 255], -1).astype(np.uint8), grade)
    return out


def grade_direct(rgb, grade):
    x = rgb.astype(np.float32)
    for k, v in grade:
        if k == 'contrast':
            x = (x - 127.5) * v + 127.5
        elif k == 'saturation':
            lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
            x = lum + (x - lum) * v
        elif k == 'sepia':
            x = x * (1 - v) + (x @ SEPIA.T) * v
        elif k == 'brightness':
            x = x * v
        elif k == 'gamma':
            x = 255 * np.power(np.clip(x / 255, 0, 1), 1 / v)
        elif k == 'balance':                                   # ffmpeg colorbalance: shifts per channel in shadows / mids / highlights
            (sr, sg, sb), (mr, mg, mb), (hr, hg, hb) = v
            y = np.clip(x / 255, 0, 1)
            for c, (s_, m_, h_) in enumerate(((sr, mr, hr), (sg, mg, hg), (sb, mb, hb))):
                l = y[..., c]
                y[..., c] = l + 0.7 * (s_ * np.clip((0.333 - l) * 4 + 0.5, 0, 1)
                                       + m_ * np.clip((l - 0.333) * 4 + 0.5, 0, 1) * np.clip((1 - l - 0.333) * 4 + 0.5, 0, 1)
                                       + h_ * np.clip((l + 0.333 - 1) * 4 + 0.5, 0, 1))
            x = np.clip(y, 0, 1) * 255
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


@functools.lru_cache(6)                 # an archive scan is hundreds of MB in memory: keep only the few on screen now
def load_image(path, crop, grade=None, flip=False):
    """premultiplied RGBA padded by edge pixels (a Lanczos window at the edge never runs out of source), source size"""
    im = Image.open(path).convert('RGBA')
    if crop:
        x, y, w, h = crop
        im = im.crop((int(x), int(y), int(x + w), int(y + h)))
    a = np.asarray(im).copy()
    a[..., :3] = apply_grade(a[..., :3], grade)
    if flip:
        a = a[:, ::-1]
    a = np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='edge')
    return Image.fromarray(a, 'RGBA').convert('RGBa'), im.size


def apply_grade_f(x, grade):
    """the same grade on float colour (a layer filter: no rounding, values may animate)"""
    for k, v in grade:
        if k == 'contrast':
            x = (x - 127.5) * v + 127.5
        elif k == 'saturation':
            lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
            x = lum + (x - lum) * v
        elif k == 'sepia':
            x = x * (1 - v) + (x @ SEPIA.T) * v
        elif k == 'brightness':
            x = x * v
        elif k == 'gamma':
            x = 255 * np.power(np.clip(x / 255, 0, 1), 1 / v)
    return x


@functools.lru_cache(6)
def load_plain(path, crop, grade=None):
    """premultiplied RGBa without edge padding (a turned picture needs transparent surroundings)"""
    im = Image.open(path).convert('RGBA')
    if crop:
        x, y, w, h = crop
        im = im.crop((int(x), int(y), int(x + w), int(y + h)))
    a = np.asarray(im).copy()
    a[..., :3] = apply_grade(a[..., :3], grade)
    return Image.fromarray(a, 'RGBA').convert('RGBa')


def crop_frame(a, crop):
    if crop:
        x, y, w, h = map(int, crop)
        a = a[y:y + h, x:x + w]
    return a


def delogo(a, r):
    """paint a small rectangle out by interpolating its border (what ffmpeg delogo does): rows and columns blended by distance"""
    x, y, w, h = map(int, r)
    out = a.astype(np.float32).copy()
    top, bot, lef, rig = out[y - 1, x:x + w], out[y + h, x:x + w], out[y:y + h, x - 1], out[y:y + h, x + w]
    v = (np.arange(h, dtype=np.float32) + 1) / (h + 1)
    u = (np.arange(w, dtype=np.float32) + 1) / (w + 1)
    vert = top[None] * (1 - v)[:, None, None] + bot[None] * v[:, None, None]
    hori = lef[:, None] * (1 - u)[None, :, None] + rig[:, None] * u[None, :, None]
    wv = np.minimum(v, 1 - v)[:, None, None]
    wu = np.minimum(u, 1 - u)[None, :, None]
    out[y:y + h, x:x + w] = (vert * wu + hori * wv) / (wu + wv)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def frame_raster(a, crop, grade=None, flip=False):
    a = crop_frame(a, crop)
    if flip:
        a = a[:, ::-1]
    a = apply_grade(a, grade)
    return Image.fromarray(np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='edge'), 'RGB'), (a.shape[1], a.shape[0])


def img_src(L, t, fps):
    """the picture's file at t (an image sequence changes it every frame)"""
    if L.get('seq') is None:
        return L['src']
    path = L['seq'] % (int(round(t * fps)) + L['seq_offset'])
    return path if os.path.exists(path) else L['src']


@functools.lru_cache(4096)
def image_size(path, crop):
    """a picture's size (after crop) from its header — no decoding (load_image keeps only a few decoded pictures, and
    asking it for a size decoded the whole file again)"""
    with Image.open(path) as im:
        if not crop:
            return im.size
    x, y, w, h = crop
    return int(x + w) - int(x), int(y + h) - int(y)


def source_size(L):
    if L['type'] == 'image':
        return image_size(L['src'], L['crop'])
    p = probe(L['src'])
    return (int(L['crop'][2]), int(L['crop'][3])) if L['crop'] else (p['w'], p['h'])


def comp_time(L, t):
    """a comp layer's own time at film time t: remapped by keys / a formula, or running at `speed` from `time`"""
    tm = L['time']
    return tm.at(t) if tm.animated else tm.at(t) + (t - L['in']) * L['speed']


def video_frame(L, t):
    """the footage frame (h, w, 3) uint8 a video layer shows at t; frame_blend: the mix of the two source frames
    around that moment (a slow motion that does not step)"""
    r, st = reader(L['src']), video_time(L, t)
    if L.get('stab') is not None:                              # the camera's shake taken out: each source frame by its own number
        from . import track
        n = min(r.last, max(0, int(math.floor(st * r.fps + 1e-4))))

        def still(i):                                          # the last one is kept: motion blur asks for the same frame many times
            hit = getattr(r, '_steady', None)
            if hit is None or hit[0] != (i, id(L)):
                r._steady = hit = ((i, id(L)), track.steady(L, r.at((i + .5) / r.fps), i))
            return hit[1]
        w = min(1.0, max(0.0, st * r.fps - n))
        if not L.get('frame_blend') or n >= r.last or w < 1e-3:
            return still(n)
        a, b = track.steady(L, r.at((n + .5) / r.fps), n), track.steady(L, r.at((n + 1.5) / r.fps), n + 1)
        return np.round(a.astype(np.float32) * (1 - w) + b.astype(np.float32) * w).astype(np.uint8)
    if not L.get('frame_blend'):
        return r.at(st)
    a, b, w = r.pair(st)
    if b is None:
        return a
    return np.round(a.astype(np.float32) * (1 - w) + b.astype(np.float32) * w).astype(np.uint8)


def video_time(L, t):
    st = L['remap'].at(t) if L.get('remap') is not None else L['from'] + (t - L['in']) * L['speed']
    if L['loop']:
        st = st % probe(L['src'])['duration']
    if L.get('until') is not None:                             # the take is cut here: hold this frame
        st = min(st, L['until'])
    return st


# ---------------- geometry ----------------
def camera_at(film, t):
    c = film.camera
    if c is None:
        return None
    x, y, z = c['position'].at(t)
    return x, y, z, c['zoom'].at(t)


def phrase_at(L, t):
    for p in L['phrases']:
        if p['start'] <= t < p['end']:
            return p
    return None


def text_of(L, t):
    """the layer's text at t: fixed, or a formula's value (a number shows without «.0» when it is whole)"""
    if L.get('text_expr') is None:
        return L['text']
    v = L['text_expr'].at(t)
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


PLANES = ('particles', 'emitter', 'plexus', 'scene3d', 'spectrum', 'wave', 'svg', 'chart', 'puppet')                # drawn in their own plane (`size`), then placed like a picture


def content_size(film, L, t=0.0):
    """layer content size in layer px at scale 1"""
    ty = L['type']
    if ty == 'solid' or 'plugin' in L:
        return L['size'].at(t)
    if ty == 'text':
        return text_layout(L['font'], L['size'], text_of(L, t), L['align'], L['line_height'], L['tracking'], L.get('features', ''))[1]
    if ty == 'gradient' and L.get('box') is not None:     # a gradient in its own box
        return L['box'].at(t)
    if ty in PLANES:
        return L['psize']
    if ty in ('gradient', 'noise', 'dust'):
        return float(film.W), float(film.H)
    if ty == 'comp':
        return float(L['film'].W), float(L['film'].H)
    if ty == 'shape':
        b = shape_path(L, t).computeTightBounds()
        return float(b.width()), float(b.height())
    if ty == 'captions':
        p = phrase_at(L, t)
        return text_layout(L['font'], L['size'], p['text'], 'center', L['line_height'])[1] if p else (0.0, 0.0)
    if ty in ('image', 'video'):
        iw, ih = source_size(L)
        fit = L['fit']
        if fit == 'native':
            return float(iw), float(ih)
        if fit in ('cover', 'contain', 'height', 'width'):
            k = {'cover': max(film.W / iw, film.H / ih), 'contain': min(film.W / iw, film.H / ih),
                 'height': film.H / ih, 'width': film.W / iw}[fit]
            if L['z'] is not None and film.camera is not None:   # 3D: fit the frame at the layer's depth as seen at the start
                _, _, cz, zoom = camera_at(film, 0.0)
                k *= (L['z'].at(0.0) - cz) / zoom
            return iw * k, ih * k
        return tuple(map(float, fit))
    return 0.0, 0.0


@memo
def corners(film, L, t):
    """the four corners of the layer's content on screen (frame px) at t, or None when it is not there"""
    p = place(film, L, t)
    if p is None:
        return None
    if L['type'] == 'shape':
        b = shape_path(L, t).computeTightBounds()
        box = (b.left(), b.top(), b.right(), b.bottom())
    else:
        w, h = content_size(film, L, t)
        box = (0.0, 0.0, w, h)
    if p.mat is not None:
        out = []
        for x, y in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])):
            v = p.mat @ np.array([x, y, 1.0])
            out.append((v[0] / v[2], v[1] / v[2]))
        return out
    cx, cy, s, _ = p
    rot = p.rot if p.rot is not None else (L['rotation'].at(t) if L.get('rotation') else 0.0)
    th = math.radians(rot)
    if L['type'] == 'shape':
        ax, ay = L['anchor']
    else:
        fx, fy = anchor(L, (box[2], box[3]), t)
        ax, ay = fx * box[2], fy * box[3]
    return [(cx + s * ((x - ax) * math.cos(th) - (y - ay) * math.sin(th)), cy + s * ((x - ax) * math.sin(th) + (y - ay) * math.cos(th)))
            for x, y in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3]))]


class Place(tuple):
    """(cx, cy, s, depth) + rot (degrees, None = the layer's own rotation) + mat: a 3×3 matrix content px → frame px when
    the layer is not just moved, scaled evenly and turned (mirrored, stretched, skewed, turned in 3D)"""
    rot = None
    mat = None


def generic(L):
    """does the layer need the matrix path (parent, pivot, [sx, sy], skew, 3D turn) instead of centre + scale + turn?"""
    if L.get('parent_layer') is not None or L.get('pivot') is not None or L.get('skew') is not None or \
            L.get('rotation_x') is not None or L.get('rotation_y') is not None or L['type'] == 'group' or \
            L.get('corner_pin') is not None or L.get('tile'):
        return True
    sc = L['scale']
    return isinstance(sc.value, tuple) or any(isinstance(k[1], tuple) for k in (sc.keys or [])) or sc.expr is not None


def pivot_px(film, L, t):
    """the content point that sits at the layer position (AE anchor point), content px"""
    if L['type'] == 'shape':
        return L['anchor']
    if L.get('pivot') is not None:
        return L['pivot'].at(t)
    w, h = content_size(film, L, t)
    ax, ay = anchor(L)
    return ax * w, ay * h


@memo
def local_mat(film, L, t):
    """content px → parent space: T(position) · R · skew · S(sx, sy) · [3D turn] · T(−anchor); with corner_pin — the
    homography that puts the content box's corners (top left, top right, bottom left, bottom right) on those points"""
    if L.get('corner_pin') is not None:
        w, h = content_size(film, L, t)
        return homography(((0, 0), (w, 0), (0, h), (w, h)), np.asarray(L['corner_pin'].at(t), float).reshape(4, 2))
    px, py = L['position'].at(t)
    sc = L['scale'].at(t)
    sx, sy = sc if isinstance(sc, tuple) else (sc, sc)
    th = math.radians(L['rotation'].at(t)) if L.get('rotation') else 0.0
    kx = math.tan(math.radians(L['skew'].at(t))) if L.get('skew') else 0.0
    ax, ay = pivot_px(film, L, t)
    if L.get('reframe') is not None:                           # slid so that the person stands in the middle (kadr/reframe.py)
        from . import reframe
        px, py = reframe.shift(film, L, t, px, py, sx, sy)
    c, s_ = math.cos(th), math.sin(th)
    M = np.array([[c, -s_, px], [s_, c, py], [0, 0, 1]]) @ np.array([[1, kx, 0], [0, 1, 0], [0, 0, 1]]) @ np.diag([sx, sy, 1.0])
    if L.get('rotation_x') is not None or L.get('rotation_y') is not None:
        po = L['persp_origin'].at(t) if L.get('persp_origin') is not None else (ax, ay)
        M = M @ turn3d(L['rotation_x'].at(t) if L.get('rotation_x') else 0.0, L['rotation_y'].at(t) if L.get('rotation_y') else 0.0,
                       L['perspective'], (po[0] - ax, po[1] - ay))
    return M @ np.array([[1, 0, -ax], [0, 1, -ay], [0, 0, 1.0]])


def homography(src, dst):
    """the 3×3 projective matrix taking four points to four points (DLT, h33 = 1)"""
    A, b = [], []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y]); b.append(X)
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y]); b.append(Y)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def turn3d(rx, ry, d, eye=(0.0, 0.0)):
    """CSS `perspective(d) rotateX(rx) rotateY(ry)` around the anchor, as a 3×3 projective matrix of the plane; `eye` —
    the perspective origin relative to the anchor (CSS perspective-origin on the parent while the plane turns about its
    own transform-origin, e.g. a cover hinged at its spine and seen from the middle of the book)"""
    a, b = math.radians(rx), math.radians(ry)
    R = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]) @ \
        np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    # a plane point (x, y, 0) → R → (X, Y, Z); seen from distance d in front of `eye`: eye + (X − eye)·d / (d − Z)
    ex, ey = eye
    return np.array([[R[0, 0] - ex * R[2, 0] / d, R[0, 1] - ex * R[2, 1] / d, 0], [R[1, 0] - ey * R[2, 0] / d, R[1, 1] - ey * R[2, 1] / d, 0],
                     [-R[2, 0] / d, -R[2, 1] / d, 1.0]])


def world_mat(film, L, t):
    """content px → frame px through every parent; None when a parent is off screen"""
    M = local_mat(film, L, t)
    q = L.get('parent_layer')
    while q is not None:
        if not (q['in'] <= t < q['out']):
            return None
        M = local_mat(film, q, t) @ M
        q = q.get('parent_layer')
    return M


def collapsing(L):
    """a group that is drawn as one picture: its children are gathered into a frame-sized layer first, then the group's
    blur, filter, effects, shadow, matte, blend and opacity apply to that picture as a whole (a CSS div with a filter)"""
    return L['type'] == 'group' and bool(L.get('blur') or L.get('filter') or L.get('effects') or L.get('matte') or L.get('shadow') or
                                         L.get('blend', 'normal') != 'normal' or L['raw'].get('isolate'))


def collector(L):
    """the nearest collapsing group above the layer (it draws the layer), or None"""
    q = L.get('parent_layer')
    while q is not None:
        if collapsing(q):
            return q
        q = q.get('parent_layer')
    return None


def chain_opacity(L, t):
    """parents' opacity (and fades) multiply into the child's, as CSS opacity does for what a div holds; a collapsing
    group applies its own opacity to the gathered picture, so the chain stops there"""
    a, q = 1.0, L.get('parent_layer')
    if L.get('mask_of'):                                       # a layer's curve mask rides it but is not dimmed with it
        return 1.0
    while q is not None:
        if collapsing(q):
            break
        a *= min(1.0, q['opacity'].at(t))
        if q['fade_in'] > 0:
            a *= ease(min(1, max(0, (t - q['in']) / q['fade_in'])), 'inOut')
        if q['fade_out'] > 0:
            a *= ease(min(1, max(0, (q['out'] - t) / q['fade_out'])), 'inOut')
        q = q.get('parent_layer')
    return a


def facing(M):
    """is the plane's front toward the eye? (the linear part keeps its orientation)"""
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0] > 0


@memo
def place(film, L, t):
    """(cx, cy, s, depth) of the layer position in frame px at scale 1, or None if not visible at t"""
    if not (L['in'] <= t < L['out']):
        return None
    if generic(L) and not (L['z'] is not None and film.camera is not None):
        M = world_mat(film, L, t)
        if M is None:
            return None
        ax, ay = pivot_px(film, L, t)
        h = M @ np.array([ax, ay, 1.0])
        if h[2] <= 1e-6:
            return None                                       # turned behind the eye
        cx, cy = h[0] / h[2], h[1] / h[2]
        a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        det = a * d - b * c
        # the plane faces the eye when the projection keeps its orientation there: det of the Jacobian = det(M) / w³, w > 0
        if L['type'] != 'group' and L.get('backface') == 'hidden' and np.linalg.det(M) <= 0:
            return None
        affine = abs(M[2, 0]) < 1e-12 and abs(M[2, 1]) < 1e-12
        p = Place((cx, cy, math.sqrt(abs(det)) / (h[2] if affine else 1.0), 0.0))
        if affine and abs(a - d) < 1e-9 and abs(b + c) < 1e-9 and det > 0 and not L.get('tile'):
            p.rot = math.degrees(math.atan2(c, a))            # still just moved, scaled evenly and turned
        else:
            p.mat = M
        return p
    if L['type'] == 'captions' and phrase_at(L, t) is None:
        return None
    sc = L['scale'].at(t)
    px, py = L['position'].at(t)
    if L['z'] is not None and film.camera is not None:
        cx, cy, cz, zoom = camera_at(film, t)
        d = L['z'].at(t) - cz
        if d <= 1:
            return None                                       # behind the camera
        k = zoom / d
        return Place((film.W / 2 + k * (px - cx), film.H / 2 + k * (py - cy), k * sc, d))
    if L.get('reframe') is not None:                           # slid so that the person stands in the middle (kadr/reframe.py)
        from . import reframe
        px, py = reframe.shift(film, L, t, px, py, sc, sc)
    return Place((px, py, sc, 0.0))


def anchor(L, wh=None, t=0.0):
    """share of the content box that sits at the layer position"""
    if L.get('pivot') is not None and wh is not None:
        px, py = L['pivot'].at(t)
        return px / max(wh[0], 1e-9), py / max(wh[1], 1e-9)
    if L['type'] == 'text':
        return {'left': 0.0, 'right': 1.0}.get(L['align'], 0.5), 0.5
    if L['type'] == 'captions':
        return 0.5, 1.0                                       # captions grow upwards from their bottom position
    return 0.5, 0.5


def bbox(film, L, t):
    """screen box (x0, y0, x1, y1) at scale 1 or None"""
    p = place(film, L, t)
    if p is None:
        return None
    cx, cy, s, _ = p
    if p.mat is not None or L['type'] == 'group':             # through the matrix: the box's corners
        if L['type'] == 'group':
            return None
        if L['type'] == 'shape':
            b = shape_path(L, t).computeTightBounds()
            box = (b.left(), b.top(), b.right(), b.bottom())
        else:
            w, h = content_size(film, L, t)
            box = (0, 0, w, h)
        q = [p.mat @ np.array([x, y, 1.0]) for x in (box[0], box[2]) for y in (box[1], box[3])]
        q = [(v[0] / v[2], v[1] / v[2]) for v in q]
        return min(v[0] for v in q), min(v[1] for v in q), max(v[0] for v in q), max(v[1] for v in q)
    if L['type'] == 'shape':                                   # shapes live in their own coordinates around `anchor`
        b = shape_path(L, t).computeTightBounds()
        ax, ay = L['anchor']
        rot = math.radians(p.rot if p.rot is not None else (L['rotation'].at(t) if L.get('rotation') else 0.0))
        pts = [(cx + s * ((x - ax) * math.cos(rot) - (y - ay) * math.sin(rot)), cy + s * ((x - ax) * math.sin(rot) + (y - ay) * math.cos(rot)))
               for x in (b.left(), b.right()) for y in (b.top(), b.bottom())]
        return min(q[0] for q in pts), min(q[1] for q in pts), max(q[0] for q in pts), max(q[1] for q in pts)
    w, h = content_size(film, L, t)
    ax, ay = anchor(L, (w, h), t)
    rot = p.rot if p.rot is not None else (L['rotation'].at(t) if L.get('rotation') else 0.0)
    if not rot:
        return cx - s * w * ax, cy - s * h * ay, cx + s * w * (1 - ax), cy + s * h * (1 - ay)
    th = math.radians(rot)
    pts = [(cx + s * (x * math.cos(th) - y * math.sin(th)), cy + s * (x * math.sin(th) + y * math.cos(th)))
           for x in (-w * ax, w * (1 - ax)) for y in (-h * ay, h * (1 - ay))]
    return min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)


OPPOSITE = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}


def transition(film, L, t):
    """opacity factor and visible rectangle (frame px, scale 1) from fade_in/out and wipe_in/out"""
    a = 1.0
    if L['fade_in'] > 0:
        a *= ease(min(1, max(0, (t - L['in']) / L['fade_in'])), 'inOut')
    if L['fade_out'] > 0:
        a *= ease(min(1, max(0, (L['out'] - t) / L['fade_out'])), 'inOut')
    clip = None
    for k in ('wipe_in', 'wipe_out'):
        w = L.get(k)
        b = bbox(film, L, t) if w else None
        if b is None:
            continue
        x0, y0, x1, y1 = b
        u = ease(min(1, max(0, ((t - L['in']) if k == 'wipe_in' else (L['out'] - t)) / w['dur'])), 'inOut')
        side = w['side'] if k == 'wipe_in' else OPPOSITE[w['side']]   # wipe_out: the edge enters from `side` and hides the layer
        r = {'left': (x0, y0, x0 + u * (x1 - x0), y1), 'right': (x1 - u * (x1 - x0), y0, x1, y1),
             'top': (x0, y0, x1, y0 + u * (y1 - y0)), 'bottom': (x0, y1 - u * (y1 - y0), x1, y1)}[side]
        clip = r if clip is None else (max(clip[0], r[0]), max(clip[1], r[1]), min(clip[2], r[2]), min(clip[3], r[3]))
    m = L['mask'].at(t) if L.get('mask') else None
    if m:
        clip = m if clip is None else (max(clip[0], m[0]), max(clip[1], m[1]), min(clip[2], m[2]), min(clip[3], m[3]))
    return a, clip


def spread_alpha(a, r):
    """CSS box-shadow spread on an alpha picture: grown by r px (a disc with a soft one-pixel rim, so an animated
    spread moves smoothly; a box's corners come out round as in CSS), r < 0 shrinks it. Returns (alpha, g): the picture
    padded by g px on every side. kadr_core.spread is the same formula on the GPU"""
    if abs(r) < 1e-6:
        return a, 0
    g = int(math.ceil(abs(r))) + 1
    src = np.pad(a.astype(np.float32), g)
    if r < 0:
        src = 1.0 - src
    rr = abs(r)
    n = int(math.ceil(rr + 0.5))
    out = np.zeros_like(src)
    H, W = src.shape
    for oy in range(-n, n + 1):
        for ox in range(-n, n + 1):
            w = min(1.0, max(0.0, rr + 0.5 - math.hypot(ox, oy)))
            if w <= 0:
                continue
            sh = np.full_like(src, 1.0 if r < 0 else 0.0)          # outside the picture: empty (1 − 0 when shrinking)
            ys, yd = (slice(oy, H), slice(0, H - oy)) if oy >= 0 else (slice(0, H + oy), slice(-oy, H))
            xs, xd = (slice(ox, W), slice(0, W - ox)) if ox >= 0 else (slice(0, W + ox), slice(-ox, W))
            sh[yd, xd] = src[ys, xs]
            np.maximum(out, sh * w, out=out)
    return (1.0 - out if r < 0 else out), g


def gradient_field(L, xs, ys, W, H, t):
    """colour and alpha of a gradient over normalised box coordinates xs, ys (0..1 across a W×H box)"""
    if L['kind'] == 'linear':
        a = math.radians(L['angle'])
        dx, dy = math.sin(a), -math.cos(a)
        half = 0.5 * (abs(dx) * W + abs(dy) * H)
        u = ((xs - 0.5) * W * dx + (ys - 0.5) * H * dy) / (2 * half) + 0.5
    else:
        (gx, gy), (rx, ry) = L['center'].at(t), L['radius']
        u = np.sqrt(((xs - gx) / rx) ** 2 + ((ys - gy) / ry) ** 2)
    return stops_at(u, L['stops'], L.get('premul', True))


def stops_at(u, stops, premul=True):
    """colour and alpha of gradient stops [(pos, rgb, alpha)] at u (numpy.interp: clamped to the end stops). premul:
    colour mixed as colour·alpha (CSS), so a stop's colour carries into a transparent one instead of turning to its
    black; where alpha is 0 the colour is the straight mix (it shows nowhere)"""
    pos = [st[0] for st in stops]
    al = np.interp(u, pos, [st[2] for st in stops]).astype(np.float32)[..., None]
    col = np.stack([np.interp(u, pos, [st[1][c] for st in stops]) for c in range(3)], -1).astype(np.float32)
    if premul and len({st[2] for st in stops}) > 1:            # equal alphas: both ways are the same
        pm = np.stack([np.interp(u, pos, [st[1][c] * st[2] for st in stops]) for c in range(3)], -1).astype(np.float32)
        col = np.where(al > 1e-6, pm / np.maximum(al, 1e-6), col).astype(np.float32)
    return col, al


def coverage(a0, a1, n):
    """per-pixel overlap of the interval [a0, a1) with pixels 0..n-1 (exact area coverage of an edge)"""
    x = np.arange(n, dtype=np.float32)
    return np.clip(np.minimum(x + 1, a1) - np.maximum(x, a0), 0, 1)


def rect_mask(X0, Y0, X1, Y1, r):
    """coverage of rectangle r over the pixel region [X0, X1) × [Y0, Y1)"""
    return np.outer(coverage(r[1] - Y0, r[3] - Y0, Y1 - Y0), coverage(r[0] - X0, r[2] - X0, X1 - X0))


def gauss_shift(a, sigma, dx, dy):
    """blur by a Gaussian of `sigma` px centred at the fractional offset (dx, dy) — sub-pixel exact, so shadows don't step.
    A wide blur (σ > 6) is computed on a q× smaller copy and scaled back up: what is left after it has no detail to lose,
    and the work drops by q² (the way compositing programs do big blurs)."""
    if sigma > 6:
        q = min(8, int(sigma // 3))
        h, w = a.shape[:2]
        H, W = -(-h // q) * q, -(-w // q) * q
        pad = [(0, H - h), (0, W - w)] + [(0, 0)] * (a.ndim - 2)
        small = np.pad(a, pad).reshape(H // q, q, W // q, q, *a.shape[2:]).mean(axis=(1, 3))
        # a small pixel j averages big pixels [jq, (j+1)q): its centre sits at (j + ½)·q, so the box below maps centres exactly
        blurred = _gauss_direct(small, math.sqrt(max(sigma ** 2 - (q ** 2 - 1) / 12, 0.25)) / q, dx / q, dy / q)
        chans = blurred.reshape(blurred.shape[0], blurred.shape[1], -1)
        up = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(chans[..., c], np.float32), 'F').resize((W, H), Image.BICUBIC,
                       box=(0, 0, W / q, H / q))) for c in range(chans.shape[2])], -1)
        return up[:h, :w].reshape(a.shape).astype(np.float32)
    return _gauss_direct(a, sigma, dx, dy)


def blur_axis(a, sigma, axis):
    """a Gaussian along one axis only (a directional blur)"""
    if sigma <= 0.25:
        return a
    R = int(math.ceil(3 * sigma)) + 1
    ks = np.arange(-R, R + 1)
    w = np.exp(-0.5 * (ks / sigma) ** 2)
    w /= w.sum()
    src = np.pad(a, [(R, R) if ax == axis else (0, 0) for ax in range(a.ndim)])
    res, n = np.zeros_like(a), a.shape[axis]
    for k, wk in zip(ks, w):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(R - k, R - k + n)
        res += wk * src[tuple(sl)]
    return res


def shift_bilinear(a, dx, dy):
    """a(x + dx, y + dy) by bilinear interpolation, zero outside — the sampling of kadr_core's gdir pass"""
    ix, iy = math.floor(dx), math.floor(dy)
    fx, fy = dx - ix, dy - iy
    H, W = a.shape[:2]
    out = np.zeros_like(a)
    for ox, oy, w in ((ix, iy, (1 - fx) * (1 - fy)), (ix + 1, iy, fx * (1 - fy)), (ix, iy + 1, (1 - fx) * fy), (ix + 1, iy + 1, fx * fy)):
        if w <= 0 or abs(ox) >= W or abs(oy) >= H:
            continue
        ys, yd = (slice(oy, H), slice(0, H - oy)) if oy >= 0 else (slice(0, H + oy), slice(-oy, H))
        xs, xd = (slice(ox, W), slice(0, W - ox)) if ox >= 0 else (slice(0, W + ox), slice(-ox, W))
        out[yd, xd] += w * a[ys, xs]
    return out


def gdir_weights(sigma):
    """the taps of one directional pass: k = −R…R, Gaussian, normalised (as blur_axis)"""
    R = int(math.ceil(3 * sigma)) + 1
    ks = np.arange(-R, R + 1)
    w = np.exp(-0.5 * (ks / sigma) ** 2)
    return ks, (w / w.sum()).astype(np.float32)


def blur_dir(a, sigma, u, shift=(0.0, 0.0)):
    """a Gaussian of `sigma` px along the unit vector u, sampled bilinearly; `shift` (dx, dy) moves the result by that
    (a shadow's offset, sub-pixel). kadr_core.gauss_dir is the same pass on the GPU"""
    if sigma <= 0.25 and not any(shift):
        return a
    ks, w = gdir_weights(max(sigma, 1e-3)) if sigma > 0.25 else (np.array([0]), np.array([1.0], np.float32))
    out = np.zeros_like(a)
    for k, wk in zip(ks, w):
        out += wk * shift_bilinear(a, k * u[0] - shift[0], k * u[1] - shift[1])
    return out


def cov_axes(C):
    """a blur's covariance (2×2, px²) → ((σ1, u1), (σ2, u2)) along its axes, or None when it is axis-aligned (then
    σx = √C00, σy = √C11 do it exactly)"""
    a, b, d = float(C[0, 0]), float(C[0, 1]), float(C[1, 1])
    if abs(b) <= 1e-4 * max(a, d, 1e-9):
        return None
    th = 0.5 * math.atan2(2 * b, a - d)
    c, s = math.cos(th), math.sin(th)
    l1 = a * c * c + 2 * b * c * s + d * s * s
    l2 = a * s * s - 2 * b * c * s + d * c * c
    return (math.sqrt(max(l1, 0.0)), (c, s)), (math.sqrt(max(l2, 0.0)), (-s, c))


def shadow_cov(J, blur):
    """a layer-space shadow's blur covariance when the layer is squashed or skewed (σ = blur / 2 layer px through J),
    None when it stays round — then the shadow is drawn as before (σ · √det J)"""
    C = (blur / 2) ** 2 * (J @ J.T)
    a, b, d = C[0, 0], C[0, 1], C[1, 1]
    if abs(a - d) <= 1e-4 * max(a, d, 1e-9) and abs(b) <= 1e-4 * max(a, d, 1e-9):
        return None
    return C


def cov_reach(C):
    """px a blur of covariance C can spread along x and along y (the taps of both passes)"""
    ax = cov_axes(C)
    if ax is None:
        return 3 * math.sqrt(max(C[0, 0], 0)) + 1, 3 * math.sqrt(max(C[1, 1], 0)) + 1
    (s1, u1), (s2, u2) = ax
    return (3 * s1 + 1) * abs(u1[0]) + (3 * s2 + 1) * abs(u2[0]) + 2, (3 * s1 + 1) * abs(u1[1]) + (3 * s2 + 1) * abs(u2[1]) + 2


def blur_cov(a, C, shift=(0.0, 0.0)):
    """a Gaussian of covariance C (2×2, px²) — a blur in the layer's own px seen through its matrix (CSS: a filter
    on a scaled or skewed element is squashed with it): axis-aligned, or two passes along its tilted axes"""
    ax = cov_axes(C)
    if ax is None:
        (s1, u1), (s2, u2) = (math.sqrt(max(C[0, 0], 0)), (1.0, 0.0)), (math.sqrt(max(C[1, 1], 0)), (0.0, 1.0))
    else:
        (s1, u1), (s2, u2) = ax
    return blur_dir(blur_dir(a, s1, u1, shift), s2, u2)


def _gauss_direct(a, sigma, dx, dy):
    out = a
    for axis, d in ((1, dx), (0, dy)):
        R = int(math.ceil(3 * sigma + abs(d))) + 1
        ks = np.arange(-R, R + 1)
        w = np.exp(-0.5 * ((ks - d) / max(sigma, 1e-3)) ** 2)
        w /= w.sum()
        src = np.pad(out, [(R, R) if ax == axis else (0, 0) for ax in range(out.ndim)])
        res, n = np.zeros_like(out), out.shape[axis]
        for k, wk in zip(ks, w):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(R - k, R - k + n)
            res += wk * src[tuple(sl)]
        out = res
    return out


# ---------------- renderer ----------------
class Renderer:
    """k: resolution multiplier (8 = supersampled reference); view: (x, y, w, h) window of the frame to render, in frame px.
    Every layer is first drawn into its own raster (X0, Y0, colour, alpha); one compositor then applies its blur, drop
    shadow, transition clip, opacity and blend mode — the way a compositing program stacks a layer's effects."""

    def __init__(self, film, k=1, view=None):
        self.f, self.k = film, k
        vx, vy, vw, vh = view or (0, 0, film.W, film.H)
        self.ox, self.oy = vx * k, vy * k
        self.W, self.H = int(vw * k), int(vh * k)
        self._grad, self._tcache, self._plates, self._shadows, self._comps = {}, {}, {}, {}, {}
        self.alpha = None
        self._pad, self._matte = 0, None                   # margin drawn beyond the frame (a blur reaches in); track matte
        self._sampling = 0                                 # > 0 while a motion-blur sample of some layer is drawn
        self._stencil = False                              # True while a stencil layer's own picture is drawn
        self._held, self._echoing, self._dim = set(), set(), 1.0   # layers on stepped time / drawing their echoes; an echo's faintness
        self._flat_fx = set()                              # layers whose layer-space effects the flat drawing applied
        self._ids = {L['id']: L for L in film.layers}
        stencils = {L['matte']['src'] for L in film.layers if L.get('matte')}

        def shown(L):                                          # `visible: false` hides a layer and what is parented to it
            while L is not None:
                if not L.get('visible', True):
                    return False
                L = L.get('parent_layer')
            return True
        live = [L for L in film.layers if L['type'] != 'camera' and L['id'] not in stencils and shown(L)]   # a matte's source is not drawn itself            # a time index: only layers on screen are looked at
        self._coll = {L['id']: collector(L) for L in live}
        self._live = live
        self._ins = np.array([L['in'] for L in live], np.float64)
        self._outs = np.array([L['out'] for L in live], np.float64)

    def order(self, t):
        """document order; each consecutive run of 3D layers is sorted far → near"""
        out, run = [], []
        for j in np.nonzero((self._ins <= t) & (self._outs > t))[0]:
            L = self._live[j]
            if (L['type'] == 'group' and not collapsing(L)) or self._coll[L['id']] is not None:
                continue                                       # nothing to draw / drawn inside its collapsing group
            p = place(self.f, L, t)
            if p is None:
                continue
            if L['z'] is not None and self.f.camera is not None:
                run.append((p[3], L, p))
            else:
                out += [(L, q) for _, L, q in sorted(run, key=lambda r: -r[0])]
                run = []
                out.append((L, p))
        out += [(L, q) for _, L, q in sorted(run, key=lambda r: -r[0])]
        return out

    def static(self, L, t):
        """a layer that looks the same on every frame it is on: nothing animated, no fade/wipe in progress, not footage"""
        if L['type'] in ('video', 'captions', 'comp', 'dust') + PLANES or L.get('matte') or 'plugin' in L or self._moving_matrix(L) or \
                L.get('text_expr') is not None or L.get('animate') or L.get('seq') is not None or \
                (L['z'] is not None and self.f.camera is not None):
            return False
        if any(isinstance(v, Prop) and v.animated for fx in (L.get('effects') or []) for v in fx['p'].values()):
            return False
        if any(fx.get('map') or getattr(fx['def'][0], 'kadr_timed', False) for fx in (L.get('effects') or [])):
            return False                                       # an effect that reads the time or another layer
        if L['type'] == 'shape':
            from . import paths
            if paths.animated(L):                              # a morph, an animated offset or repeater, a text running along
                return False
        if any(isinstance(L.get(k), Prop) and L[k].animated for k in ('position', 'scale', 'opacity', 'size', 'rotation', 'blur', 'center', 'trim', 'mask', 'box', 'radius')):
            return False
        if isinstance((L.get('stroke') or {}).get('dash_offset'), Prop) and L['stroke']['dash_offset'].animated:
            return False
        if any(v.animated for s_ in (L.get('shadow') or []) for v in s_.values() if isinstance(v, Prop)) or \
                any(v.animated for _, v in (L.get('filter') or [])):
            return False
        a, clip = transition(self.f, L, t)
        return a == 1.0 and clip is None

    @staticmethod
    def _moving_matrix(L):
        """placed through a matrix that may change: a parent (it moves the child), a group, or an animated pivot, skew,
        3D turn, corner pin (a pixel anchor alone, a fixed skew or turn keep the matrix the same every frame)"""
        if not generic(L):
            return False
        if L.get('parent_layer') is not None or L['type'] == 'group':
            return True
        return any(isinstance(L.get(k), Prop) and L[k].animated for k in ('pivot', 'skew', 'rotation_x', 'rotation_y', 'corner_pin', 'persp_origin'))

    def frame_rgba(self, t):
        """premultiplied colour and alpha over a transparent ground — how a comp layer is rendered"""
        img = np.zeros((self.H, self.W, 3), np.float32)
        self.alpha = np.zeros((self.H, self.W, 1), np.float32)
        if self.f.background is not None:
            img[:] = self.f.background
            self.alpha[:] = 1
        for L, p in self.order(t):
            self.layer(img, L, p, t)
        a, self.alpha = self.alpha, None
        return img, a

    def _covers(self, L, t):
        """the layer hides all that lies under it at t: an opaque solid, or a comp with an opaque ground, as large as
        the frame, at full opacity, plainly placed. What is under it need not be drawn (a 3D scene or footage behind a
        full-screen insert costs a frame's time for nothing — found on the first real film, 16.11.2026)"""
        if L['type'] not in ('solid', 'comp') or L.get('blend', 'normal') != 'normal' or not L.get('visible', True) or L['z'] is not None:
            return False
        if any(L.get(k) for k in ('matte', 'mask', 'filter', 'effects', 'echo', 'posterize', 'corner_pin', 'blur')) or L.get('parent_layer') is not None or 'pivot' in L:
            return False
        if any(L.get(k) is not None and (L[k].animated or L[k].value) for k in ('rotation', 'skew', 'rotation_x', 'rotation_y')):
            return False

        def plain(X, W, H, tt):                                # full size, in the middle, scale 1, fully opaque
            if X['opacity'].at(tt) < 1 or X['scale'].at(tt) != 1:
                return False
            px = X['position'].at(tt)
            return abs(px[0] - W / 2) < .01 and abs(px[1] - H / 2) < .01
        fade, clip = transition(self.f, L, t)
        if fade < 1 or clip is not None or not plain(L, self.f.W, self.f.H, t):
            return False
        w, h = content_size(self.f, L, t)
        if w < self.f.W or h < self.f.H:
            return False
        if L['type'] == 'solid':
            return True
        F = L['film']
        if F.background is not None:
            return True
        if any(str(X.get('blend', '')).startswith(('stencil', 'silhouette')) for X in F.layers):
            return False                                       # a stencil inside cuts the comp's ground through
        tc = comp_time(L, t)
        B = F.layers[0] if F.layers else None                  # the comp's own ground: an opaque solid under everything in it
        if B is None or B['type'] != 'solid' or not (B['in'] <= tc < B['out']) or B.get('blend', 'normal') != 'normal' or B['z'] is not None:
            return False
        if any(B.get(k) for k in ('matte', 'mask', 'filter', 'effects', 'echo', 'posterize', 'corner_pin', 'blur')) or B.get('parent_layer') is not None or 'pivot' in B:
            return False
        if any(B.get(k) is not None and (B[k].animated or B[k].value) for k in ('rotation', 'skew', 'rotation_x', 'rotation_y')) or B['fade_in'] or B['fade_out'] or B.get('wipe_in') or B.get('wipe_out'):
            return False
        bw, bh = B['size'].at(tc)
        return plain(B, F.W, F.H, tc) and bw >= F.W and bh >= F.H

    def seen(self, t):
        """the layers of the frame at t, bottom to top, from the topmost one that covers the whole frame"""
        layers = self.order(t)
        for j in range(len(layers) - 1, 0, -1):
            if self._covers(layers[j][0], t):
                return layers[j:]
        return layers

    def frame(self, t):
        layers = self.seen(t)
        n = 0                                                  # the bottom run of static layers is composited once and reused
        while n < len(layers) and self.static(layers[n][0], t):
            n += 1
        key = tuple(L['id'] for L, _ in layers[:n])
        cached = bool(n) and key in self._plates
        img = self._plates[key].copy() if cached else np.full((self.H, self.W, 3), self.f.background, np.float32)
        for j, (L, p) in enumerate(layers):
            if j < n and cached:
                continue
            if j == n and n and not cached:
                if len(self._plates) > 8:
                    self._plates.clear()
                self._plates[key] = img.copy()
            self.layer(img, L, p, t)
        if n == len(layers) and n and not cached:
            self._plates[key] = img.copy()
        return img

    def layer(self, img, L, p, t):
        if L.get('posterize') and L['id'] not in self._held:   # Posterize Time: the layer as it was at its last step
            fps_ = L['posterize']
            tq = L['in'] + math.floor((t - L['in']) * fps_ + 1e-6) / fps_
            self._held.add(L['id'])
            try:
                q = place(self.f, L, tq)
                if q is not None:
                    self.layer(img, L, q, tq)
            finally:
                self._held.discard(L['id'])
            return
        if L.get('echo') and L['id'] not in self._echoing:     # Echo: what it was a moment ago, fainter, under itself
            e = L['echo']
            self._echoing.add(L['id'])
            try:
                for k_ in range(int(round(e['count'].at(t))), 0, -1):
                    tk = t - k_ * e['step'].at(t)
                    q = place(self.f, L, tk) if tk >= L['in'] else None
                    if q is not None:
                        self._dim = e['decay'].at(t) ** k_
                        self.layer(img, L, q, tk)
            finally:
                self._dim = 1.0
                self._echoing.discard(L['id'])
        fade, clip = transition(self.f, L, t)
        op = min(1.0, L['opacity'].at(t)) * fade * self._dim
        if L.get('parent_layer') is not None:
            op *= chain_opacity(L, t)
        if op <= 0:
            return
        if L.get('blend', 'normal') in STENCILS and not self._stencil:
            # After Effects' stencil / silhouette: what is drawn so far (in this frame, group or comp) is kept only
            # where this layer is — or is not; the layer itself is not seen
            kind, by = L['blend'].split('-')
            self._stencil = True
            try:
                A = self.matte({'src': L['id'], 'mode': by + ('_inverted' if kind == 'silhouette' else '')}, t)
            finally:
                self._stencil = False
            if A is None:
                A = np.zeros((self.H, self.W), np.float32)
            img *= A[..., None]
            if self.alpha is not None:
                self.alpha *= A[..., None]
            return
        k, ox, oy = self.k, self.ox, self.oy
        cx, cy, s, _ = p
        clip = (clip[0] * k - ox, clip[1] * k - oy, clip[2] * k - ox, clip[3] * k - oy) if clip else None
        rot = p.rot if getattr(p, 'rot', None) is not None else (L['rotation'].at(t) if L.get('rotation') else 0.0)
        sk = s * k
        bv = L['blur'].at(t) if L.get('blur') else 0.0
        bx, by = (bv[0] * sk, bv[1] * sk) if isinstance(bv, tuple) else (bv * sk, bv * sk)   # [x, y]: a directional blur
        J = self._lin(L, p, t)                                 # raster px per layer px: what layer-space sizes go through
        jmax = float(np.linalg.norm(J, 2))
        bcov = None
        if L.get('blur_space') == 'layer' and L.get('blur'):  # CSS: the blur squashed and turned with the layer
            bvx, bvy = bv if isinstance(bv, tuple) else (bv, bv)
            bcov = J @ np.diag([bvx * bvx, bvy * bvy]) @ J.T
            bx, by = (v / 3 for v in cov_reach(bcov))
        fxs = [(fx, {n: (v.at(t) if isinstance(v, Prop) else v) for n, v in fx['p'].items()}) for fx in (L.get('effects') or [])]
        reach = 3 * max(bx, by) + sum(fx['def'][2](pv, jmax if pv.get('space') == 'layer' else sk) for fx, pv in fxs if fx['def'][2])
        reach += max([3 * sh['blur'].at(t) * (jmax if sh.get('space') == 'layer' else sk) / 2 +
                      (max(map(abs, sh['offset'].at(t))) + max(0.0, sh['spread'].at(t))) * max(sk, jmax) for sh in (L.get('shadow') or [])], default=0)
        M = None
        if L.get('matte'):
            M = self.matte(L['matte'], t)
            if M is None:
                return
        maps = [self.fx_map(fx['map'], t) if fx.get('map') else None for fx, _ in fxs]
        self._pad = int(math.ceil(reach))                      # what lies just outside the frame still blurs/glows into it
        try:
            n, h = self.mb_samples(L, t)
            ras = self._draw_at(L, t, p) if n <= 1 else self._motion_blur(L, t, n, h)
        finally:
            self._pad = 0
        if ras is None:
            return
        X0, Y0, rgb, a, key = ras
        if bcov is not None and max(bx, by) > 0.25:            # in the layer's own px: an ellipse through its matrix
            Rx, Ry = int(math.ceil(3 * bx)) + 1, int(math.ceil(3 * by)) + 1
            pm = blur_cov(np.pad(np.concatenate([rgb * a, a], -1), ((Ry, Ry), (Rx, Rx), (0, 0))), bcov)
            a = pm[..., 3:]
            rgb = pm[..., :3] / np.maximum(a, 1e-6)
            X0, Y0, key = X0 - Rx, Y0 - Ry, None
        elif max(bx, by) > 0.25:                               # CSS blur(r): a Gaussian of σ = r
            Rx, Ry = int(math.ceil(3 * bx)) + 1, int(math.ceil(3 * by)) + 1
            pm = np.pad(np.concatenate([rgb * a, a], -1), ((Ry, Ry), (Rx, Rx), (0, 0)))
            if abs(bx - by) < 1e-6:
                pm = gauss_shift(pm, bx, 0, 0)
            else:
                pm = blur_axis(blur_axis(pm, bx, 1), by, 0)
            a = pm[..., 3:]
            rgb = pm[..., :3] / np.maximum(a, 1e-6)
            X0, Y0, key = X0 - Rx, Y0 - Ry, None
        if L.get('filter'):                                    # the layer's own picture through a CSS-like filter
            rgb = np.clip(apply_grade_f(rgb, [(k, v.at(t)) for k, v in L['filter']]), 0, 255)
            key = None
        flat_done = L['id'] in self._flat_fx                   # drawn flat: its layer-space effects are on already
        self._flat_fx.discard(L['id'])
        for (fx, pv), mp in zip(fxs, maps):                    # effects from plugins (kadr/effects.py and plugin folders)
            if flat_done and getattr(fx['def'][0], 'kadr_flat', False):
                continue
            fxo = Fx(sk, t, t - L['in'], self.k, (X0, Y0), self._raster_inv(L, p, t), J)
            fxo.frame = (self.W, self.H)
            if getattr(fx['def'][0], 'kadr_gpu', False):       # a shader: the same kernel the GPU renderer runs
                fxo.pivot = self._fx_pivot(fxo.inv, p)
                from . import shader
                rgb, a, gx = shader.apply_numpy(fx['def'][0], rgb, a, pv, fxo, mp)
                X0, Y0, key = X0 - gx, Y0 - gx, None
                continue
            got = fx['def'][0](np.ascontiguousarray(rgb, np.float32), np.ascontiguousarray(a, np.float32), pv, fxo)
            rgb, a = got[0], got[1]
            if len(got) > 2:                                   # the effect grew the picture: by n px or by (dx, dy)
                gx, gy = got[2] if isinstance(got[2], tuple) else (got[2], got[2])
                X0, Y0 = X0 - gx, Y0 - gy
            key = None
        self._matte = M
        for si, sh in enumerate(L.get('shadow') or []):
            sb, so, sop = sh['blur'].at(t), sh['offset'].at(t), sh['opacity'].at(t)
            scov = None
            if sh.get('space') == 'layer':                    # CSS drop-shadow / AE Drop Shadow: offset and blur in the layer's own px
                dx, dy = J[0, 0] * so[0] + J[0, 1] * so[1], J[1, 0] * so[0] + J[1, 1] * so[1]
                bs = math.sqrt(abs(np.linalg.det(J)))
                scov = shadow_cov(J, sb)                       # squashed or skewed: the blur an ellipse too
            else:                                              # offset in frame px (times the layer's scale)
                dx, dy, bs = so[0] * sk, so[1] * sk, sk
            sp = sh['spread'].at(t) * bs                       # px, the way the offset scales
            Hm = self._shadow_h(L, p, t, so) if sh.get('space') == 'layer' else None
            skey = (L['id'], si, key, round(sk, 6), sb, so, sh['outside'], round(dx, 5), round(dy, 5), round(sp, 5),
                    None if scov is None else tuple(np.round(scov.ravel(), 5))) if key is not None and Hm is None else None
            if skey is not None and skey in self._shadows:
                SX, SY, sa = self._shadows[skey]
            else:
                sig = sb * bs / 2
                WX0, WY0, src = X0, Y0, a[..., 0]
                if Hm is not None:                             # a 3D layer: the offset through its perspective, point by point
                    got = self._warp_alpha(src, X0, Y0, Hm)
                    if got is None:
                        continue
                    WX0, WY0, src = got
                    dx = dy = 0.0
                base, g = spread_alpha(src, sp)                # the alpha grown (or shrunk) by the spread, g px more each side
                if scov is not None:
                    R = int(math.ceil(max(cov_reach(scov)) + max(abs(dx), abs(dy)))) + 1
                    sa = blur_cov(np.pad(base, R), scov, (dx, dy))[..., None]
                else:
                    R = int(math.ceil(3 * sig + max(abs(dx), abs(dy)))) + 1
                    sa = gauss_shift(np.pad(base, R), sig, dx, dy)[..., None]
                SX, SY = WX0 - R - g, WY0 - R - g
                if sh['outside']:                              # CSS box-shadow: none under the box itself
                    sa = sa.copy()
                    cx_, cy_ = X0 - SX, Y0 - SY
                    sa[cy_:cy_ + a.shape[0], cx_:cx_ + a.shape[1]] *= 1 - a
                if skey is not None:
                    if len(self._shadows) > 64:
                        self._shadows.clear()
                    self._shadows[skey] = (SX, SY, sa)
            mode, self.mode = getattr(self, 'mode', 'normal'), 'normal'
            self._put(img, SX, SY, sa * (sop * op), np.array(sh['color'], np.float32), clip)
            self.mode = mode
        self.mode = L.get('blend', 'normal')
        self._put(img, X0, Y0, a * op, rgb, clip)
        self.mode, self._matte = 'normal', None

    def _warp_alpha(self, a, X0, Y0, Hm):
        """an alpha raster at (X0, Y0) moved through the raster homography Hm (Skia, Mitchell, as _project places a
        picture) → (UX0, UY0, alpha) over its box joined with the original one"""
        import skia
        h, w = a.shape
        Mc = Hm @ np.array([[1, 0, X0], [0, 1, Y0], [0, 0, 1.0]])
        r = self._warp_box(Mc, w, h, X0, Y0)
        if r is None:
            return None
        a8 = np.clip(np.round(a * 255), 0, 255).astype(np.uint8)
        im = skia.Image.fromarray(np.ascontiguousarray(np.repeat(a8[..., None], 4, -1)), colorType=skia.kRGBA_8888_ColorType,
                                  alphaType=skia.kPremul_AlphaType)
        surf = skia.Surface(r[2] - r[0], r[3] - r[1])
        c = surf.getCanvas()
        c.clear(skia.Color4f(0, 0, 0, 0))
        c.translate(-r[0], -r[1])
        c.concat(skia_matrix(Mc))
        c.drawImageRect(im, skia.Rect.MakeWH(w, h), skia.SamplingOptions(skia.CubicResampler.Mitchell()), skia.Paint(AntiAlias=True))
        out = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
        return r[0], r[1], out[..., 3].astype(np.float32) / 255

    def _shadow_h(self, L, p, t, so):
        """raster px → raster px: where a point's shadow lands for the offset `so` in the layer's own plane, through the
        layer's perspective (a far edge's shadow moves less than a near one's). None without a perspective: there the
        offset is the same everywhere, J @ so (_lin)"""
        M = getattr(p, 'mat', None)
        if M is None or abs(M[2, 0]) + abs(M[2, 1]) < 1e-12:
            return None
        inv = self._raster_inv(L, p, t)
        if inv is None:
            return None
        return np.linalg.inv(inv) @ np.array([[1, 0, so[0]], [0, 1, so[1]], [0, 0, 1.0]]) @ inv

    def _warp_box(self, Mc, w, h, X0, Y0):
        """the raster box of a w × h picture through Mc, joined with the picture's own box at (X0, Y0) (a cut needs it)"""
        r = self._quad_bounds(Mc, (0, 0, w, h))
        if r is None:
            return None
        return (min(r[0], X0), min(r[1], Y0), max(r[2], X0 + w), max(r[3], Y0 + h))

    def _raster_inv(self, L, p, t):
        """raster px → the layer's content px at t (3×3), or None (a matrix that folds to nothing)"""
        k, ox, oy = self.k, self.ox, self.oy
        if getattr(p, 'mat', None) is not None:
            M = np.array([[k, 0, -ox], [0, k, -oy], [0, 0, 1.0]]) @ p.mat
        else:
            cx, cy, s, _ = p
            rot = p.rot if getattr(p, 'rot', None) is not None else (L['rotation'].at(t) if L.get('rotation') else 0.0)
            if L['type'] == 'shape':
                ax, ay = L['anchor']
            else:
                w, h = content_size(self.f, L, t)
                fx_, fy_ = anchor(L, (w, h), t)
                ax, ay = fx_ * w, fy_ * h
            th, sc = math.radians(rot), s * k
            c, si = math.cos(th), math.sin(th)
            M = np.array([[sc * c, -sc * si, cx * k - ox], [sc * si, sc * c, cy * k - oy], [0, 0, 1.0]]) @ \
                np.array([[1, 0, -ax], [0, 1, -ay], [0, 0, 1.0]])
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return None

    def _lin(self, L, p, t):
        """raster px per layer px around the layer's pivot (2×2): its turn, scale, mirror, skew, 3D turn and k"""
        if getattr(p, 'mat', None) is not None:
            M = p.mat
            ax, ay = pivot_px(self.f, L, t)
            v = M @ np.array([ax, ay, 1.0])
            return (M[:2, :2] * v[2] - np.outer(v[:2], M[2, :2])) / v[2] ** 2 * self.k
        rot = p.rot if getattr(p, 'rot', None) is not None else (L['rotation'].at(t) if L.get('rotation') else 0.0)
        th, s = math.radians(rot), p[2] * self.k
        return np.array([[s * math.cos(th), -s * math.sin(th)], [s * math.sin(th), s * math.cos(th)]])

    def _skia_out(self, px, X0, Y0, ss):
        """a Skia surface's pixels (premultiplied RGBA8, ss× supersampled) → the layer raster (X0, Y0, rgb, a)"""
        a = px.astype(np.float32)
        if ss > 1:
            a = a.reshape(a.shape[0] // ss, ss, a.shape[1] // ss, ss, 4).mean(axis=(1, 3))
        al = a[..., 3:] / 255
        return X0, Y0, a[..., :3] / np.maximum(al, 1e-6), al, None

    def _draw_at(self, L, t, p, tc=None):
        """the layer's raster for placement p (at time t); tc: the time its content is taken at (footage under motion blur
        keeps the frame of the whole frame, only its placement moves)"""
        k, ox, oy = self.k, self.ox, self.oy
        cx, cy, s, _ = p
        tc = t if tc is None else tc
        self._now = tc                                         # the time a drawer without t (text glyphs) needs
        if L['type'] == 'group':
            return self.draw_group(L, t)
        if 'plugin' in L and L['plugin'][1] == 'gpu':          # a layer a shader draws, through whatever matrix it has
            return self.draw_shader(L, tc, p)
        if getattr(p, 'mat', None) is not None:                # mirrored, stretched, skewed or turned in 3D: the full matrix
            return self.draw_matrix(L, tc, np.array([[k, 0, -ox], [0, k, -oy], [0, 0, 1.0]]) @ p.mat, cx * k - ox, cy * k - oy)
        rot = p.rot if getattr(p, 'rot', None) is not None else (L['rotation'].at(t) if L.get('rotation') else 0.0)
        return getattr(self, 'draw_' + ('plugin' if 'plugin' in L else L['type']))(L, tc, cx * k - ox, cy * k - oy, s * k, rot)

    NO_MB = ('group', 'captions', 'noise', 'gradient', 'dust', 'camera')

    def mb_samples(self, L, t):
        """motion blur as in After Effects (shutter 180° by default, centred on the frame): how many sub-frame samples
        this layer needs — its corners' travel across the shutter, measured from its own keys (no optical flow), divided
        by the allowed step between copies; a layer standing still is drawn once"""
        mb = self.f.motion_blur
        if not mb or L['type'] in self.NO_MB or (L['type'] == 'gradient' and L.get('box') is None) or L.get('posterize'):
            return 1, 0.0                                      # (a layer on stepped time holds still between its steps)
        if self._sampling:                                     # inside another layer's sample (a group being blurred):
            return 1, 0.0                                      # that sample is one moment — blurring again = twice the shutter
        h = mb['shutter'] / 360 / self.f.fps / 2
        c0, c1 = corners(self.f, L, t - h), corners(self.f, L, t + h)
        if c0 is None or c1 is None:
            return 1, h
        # the path across the shutter in 8 pieces: a spin returns to where it began, and a jump cut (a hold key, a
        # punch-in made with a step in a formula) puts nearly all the travel into one piece — a cut is never blurred
        cs = [c0] + [corners(self.f, L, t - h + 2 * h * j / 8) for j in range(1, 8)] + [c1]
        if any(c is None for c in cs):
            return 1, h
        segs = [max(math.dist(a, b) for a, b in zip(p_, q_)) for p_, q_ in zip(cs, cs[1:])]
        d = sum(segs)
        if d * self.k > 2 * mb['step'] and max(segs) > 0.6 * d:
            return 1, h
        n = int(math.ceil(d * self.k / mb['step']))
        return (1 if n <= 1 else min(mb['max'], n)), h

    def _motion_blur(self, L, t, n, h):
        """n placements spread evenly over the shutter, each drawn and averaged (premultiplied) into one raster"""
        acc, box = None, None
        got = 0
        for i in range(n):
            ts = t - h + (i + 0.5) * 2 * h / n
            q = place(self.f, L, ts)
            if q is None:
                continue
            self._sampling += 1
            try:
                r = self._draw_at(L, ts, q, tc=t if (L['type'] == 'video' or L.get('seq')) else ts)
            finally:
                self._sampling -= 1
            got += 1
            if r is None:
                continue
            X0, Y0, rgb, a, _ = r
            pm = np.concatenate([np.asarray(rgb, np.float32) * a, np.broadcast_to(a, a.shape[:2] + (1,))], -1)
            if acc is None:
                acc, box = pm.copy(), [X0, Y0, X0 + pm.shape[1], Y0 + pm.shape[0]]
                continue
            nx0, ny0 = min(box[0], X0), min(box[1], Y0)
            nx1, ny1 = max(box[2], X0 + pm.shape[1]), max(box[3], Y0 + pm.shape[0])
            if (nx0, ny0, nx1, ny1) != tuple(box):
                big = np.zeros((ny1 - ny0, nx1 - nx0, 4), np.float32)
                big[box[1] - ny0:box[3] - ny0, box[0] - nx0:box[2] - nx0] = acc
                acc, box = big, [nx0, ny0, nx1, ny1]
            acc[Y0 - box[1]:Y0 - box[1] + pm.shape[0], X0 - box[0]:X0 - box[0] + pm.shape[1]] += pm
        if acc is None or not got:
            return None
        acc /= got
        a = acc[..., 3:]
        return box[0], box[1], acc[..., :3] / np.maximum(a, 1e-6), a, None

    def draw_shader(self, L, t, p):
        """a layer drawn by a shader (kadr/shader.py: `type: shader`, flare, light_leak…): the shader runs over the
        part of the frame the layer's box covers and asks, pixel by pixel, where in the box it is — so the layer may
        be moved, turned, scaled and tilted in 3D like any other, with no picture resampled"""
        inv = self._raster_inv(L, p, t)
        w, h = L['size'].at(t)
        if inv is None or w <= 0 or h <= 0 or abs(np.linalg.det(inv)) < 1e-12:
            return None
        r = self._quad_bounds(np.linalg.inv(inv), (0, 0, w, h))
        if r is None:
            return None
        J = self._lin(L, p, t)
        fx = Fx(math.sqrt(abs(np.linalg.det(J))), t, t - L['in'], self.k, (r[0], r[1]), inv, J)
        fx.frame, fx.pivot = (self.W, self.H), self._fx_pivot(inv, p)
        pv = {n: (v.at(t) if isinstance(v, Prop) else v) for n, v in L['params'].items()}
        return self._gpu_ras(r[0], r[1], L['plugin'][0](pv, fx, (r[2] - r[0], r[3] - r[1]), (w, h)))

    @staticmethod
    def _gpu_ras(X0, Y0, im):
        """a kadr_core picture as this renderer's raster: here it comes down to numpy"""
        a = im.download()
        return X0, Y0, a[..., :3], a[..., 3:], None

    def _fx_pivot(self, inv, p):
        """the layer's point that sits at its position, in the content px a shader sees (fx_layer_xy) — where an
        effect is centred unless the document says otherwise; through the same matrix, so it holds for any layer"""
        if inv is None:
            return None
        v = inv @ np.array([p[0] * self.k - self.ox, p[1] * self.k - self.oy, 1.0])
        return (float(v[0] / v[2]), float(v[1] / v[2])) if abs(v[2]) > 1e-12 else None

    def fx_map(self, src, t):
        """an effect's second picture (a displacement map, a depth map…): the layer `src` drawn alone into the frame —
        a premultiplied canvas (H, W, 4): colour 0..255 × alpha, alpha 0..1"""
        S = self._ids[src]
        p = place(self.f, S, t) if S['type'] != 'camera' else None
        img, alpha = np.zeros((self.H, self.W, 3), np.float32), self.alpha
        self.alpha = np.zeros((self.H, self.W, 1), np.float32)
        try:
            if p is not None:
                self.layer(img, S, p, t)
            return np.concatenate([img, self.alpha], -1)
        finally:
            self.alpha = alpha

    def matte(self, mt, t):
        """a track matte (After Effects): another layer, drawn into its own frame, decides where this one shows —
        by its alpha or its brightness, or the inverse. None: nothing shows."""
        S = self._ids[mt['src']]
        p = place(self.f, S, t) if S['type'] != 'camera' else None
        inv = mt['mode'].endswith('_inverted')
        if p is None:
            return np.ones((self.H, self.W), np.float32) if inv else None
        img, alpha = np.zeros((self.H, self.W, 3), np.float32), self.alpha
        self.alpha = np.zeros((self.H, self.W, 1), np.float32)
        try:
            self.layer(img, S, p, t)
            v = (img @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / 255 if mt['mode'].startswith('luma') else self.alpha[..., 0]
        finally:
            self.alpha = alpha
        return 1 - np.clip(v, 0, 1) if inv else np.clip(v, 0, 1)

    def _put(self, img, X0, Y0, m, rgb, clip):
        """blend a raster that may stick out of the frame: crop it to the frame first"""
        h, w = m.shape[:2]
        x0, y0, x1, y1 = max(0, X0), max(0, Y0), min(self.W, X0 + w), min(self.H, Y0 + h)
        if x0 >= x1 or y0 >= y1:
            return
        m = m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
            rgb = rgb[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        self._blend(img, x0, y0, m, rgb, clip)

    def _blend(self, img, X0, Y0, m, rgb, clip):
        """img[region] = img·(1 − m) + rgb·m"""
        Y1, X1 = Y0 + m.shape[0], X0 + m.shape[1]
        if clip:
            m = m * rect_mask(X0, Y0, X1, Y1, clip)[..., None]
        if self._matte is not None:
            m = m * self._matte[Y0:Y1, X0:X1, None]
        reg = img[Y0:Y1, X0:X1]
        if getattr(self, 'mode', 'normal') != 'normal':
            rgb = blend_mode(self.mode, reg / 255, np.broadcast_to(np.asarray(rgb, np.float32) / 255, reg.shape)) * 255
        reg *= 1 - m
        reg += m * rgb
        if self.alpha is not None:
            ar = self.alpha[Y0:Y1, X0:X1]
            ar *= 1 - m
            ar += m

    def _box(self, L, t, cx, cy, s):
        w, h = content_size(self.f, L, t)
        ax, ay = anchor(L, (w, h), t)
        return cx - s * w * ax, cy - s * h * ay, s * w, s * h

    # ---------- layer kinds: each returns (X0, Y0, colour (h, w, 3), alpha (h, w, 1), cache key or None) ----------
    def _rotated(self, L, t, cx, cy, s, rot, im, size, still=False):
        """a raster turned by `rot` degrees around the layer position: Lanczos to the display size (exact downscaling),
        then Skia places it with the rotation, a cubic filter and anti-aliased edges. A still picture is prepared once,
        15 % larger than it shows, and only gently rescaled afterwards: re-preparing it every frame at a whole-pixel size
        made its detail pulse as that size stepped (measured on a slowly pushing print)."""
        import skia
        w, h = content_size(self.f, L, t)
        ax, ay = anchor(L, (w, h), t)
        key = ('rot', L['id'])
        hit = self._tcache.get(key) if still else None
        if hit is None or not (0.6 * hit[1] <= s * w <= hit[1]):
            dw = max(1, int(math.ceil(s * w * (1.15 if still else 1.0))))
            dh = max(1, int(round(dw * h / w)))
            a = np.asarray(im.resize((dw, dh), Image.LANCZOS), np.uint8)
            if a.shape[2] == 3:
                a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
            sk_img = skia.Image.fromarray(np.ascontiguousarray(a), colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
            if still:
                self._tcache[key] = (sk_img, dw)
        else:
            sk_img = hit[0]
        th = math.radians(rot)
        corners = [(-ax * s * w, -ay * s * h), ((1 - ax) * s * w, -ay * s * h), (-ax * s * w, (1 - ay) * s * h), ((1 - ax) * s * w, (1 - ay) * s * h)]
        pts = [(cx + x * math.cos(th) - y * math.sin(th), cy + x * math.sin(th) + y * math.cos(th)) for x, y in corners]
        X0, Y0 = math.floor(min(p[0] for p in pts)) - 1, math.floor(min(p[1] for p in pts)) - 1
        X1, Y1 = math.ceil(max(p[0] for p in pts)) + 1, math.ceil(max(p[1] for p in pts)) + 1
        X0, Y0, X1, Y1 = max(X0, -64), max(Y0, -64), min(X1, self.W + 64), min(Y1, self.H + 64)
        if X0 >= X1 or Y0 >= Y1:
            return None
        surf = skia.Surface(X1 - X0, Y1 - Y0)
        c = surf.getCanvas()
        c.clear(skia.Color4f(0, 0, 0, 0))
        c.translate(cx - X0, cy - Y0)
        c.rotate(rot)
        c.drawImageRect(sk_img, skia.Rect.MakeXYWH(-ax * s * w, -ay * s * h, s * w, s * h), skia.SamplingOptions(skia.CubicResampler.Mitchell()),
                        skia.Paint(AntiAlias=True))
        return self._skia_out(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType), X0, Y0, 1)

    FULL = ('gradient', 'noise', 'dust', 'captions', 'comp')

    def _members(self, G, t):
        """the layers a collapsing group draws at t, with their placements"""
        out = []
        for C in self._live:
            if self._coll[C['id']] is not G or not (C['in'] <= t < C['out']):
                continue
            if C['type'] == 'group' and not collapsing(C):
                continue
            q = place(self.f, C, t)
            if q is not None:
                out.append((C, q))
        return out

    def _extent(self, G, t, members):
        """the raster box (x0, y0, x1, y1) the group's children can paint into — their boxes over the shutter, grown by
        their blur, shadows and effects — or None when it must be the whole frame (a full-frame layer, an effect)"""
        if G.get('effects'):
            return None                                        # motion_tile and the like read the whole frame
        mb = self.f.motion_blur
        h = mb['shutter'] / 360 / self.f.fps / 2 if mb else 0.0
        box = None
        for C, q in members:
            if C['type'] in self.FULL and not (C['type'] == 'gradient' and C.get('box') is not None) or C.get('effects'):
                return None
            if C['type'] == 'group':
                sub = self._extent(C, t, self._members(C, t))
                if sub is None:
                    return None
                b = ((sub[0] + self.ox) / self.k, (sub[1] + self.oy) / self.k, (sub[2] + self.ox) / self.k, (sub[3] + self.oy) / self.k)
                bs = [b]
            else:
                bs = [bbox(self.f, C, tt) for tt in ((t - h, t, t + h) if h else (t,))]
                if any(b is None for b in bs):
                    return None
            sc = q[2]
            grow = 6.0
            st = C.get('stroke') if C['type'] in ('shape', 'text') else None
            if st:                                             # a stroke (and its arrowheads) reaches past the outline
                grow += (st.get('width', 2) * (3 if C.get('arrow') else 1)) * sc
            bv = C['blur'].at(t) if C.get('blur') else 0.0
            grow += 3 * (max(bv) if isinstance(bv, tuple) else bv) * sc
            for sh in C.get('shadow') or []:
                grow += 1.5 * sh['blur'].at(t) * sc + (max(map(abs, sh['offset'].at(t))) + max(0.0, sh['spread'].at(t))) * sc
            for b in bs:
                g = (b[0] - grow, b[1] - grow, b[2] + grow, b[3] + grow)
                box = g if box is None else (min(box[0], g[0]), min(box[1], g[1]), max(box[2], g[2]), max(box[3], g[3]))
        if box is None:
            return (0, 0, 0, 0)
        e = self._pad                                          # the group's own blur/shadow reach past the frame
        x0 = max(-e, math.floor(box[0] * self.k - self.ox))
        y0 = max(-e, math.floor(box[1] * self.k - self.oy))
        x1 = min(self.W + e, math.ceil(box[2] * self.k - self.ox))
        y1 = min(self.H + e, math.ceil(box[3] * self.k - self.oy))
        return (x0, y0, max(x0, x1), max(y0, y1))

    def draw_group(self, L, t):
        """a collapsing group: its children (already placed through it) are drawn into a transparent picture — only as
        big as what they cover (the whole frame when a child or effect needs it) — which then goes through the group's own
        blur, filter, effects, shadow, matte, opacity and blend"""
        members = self._members(L, t)
        if not members:
            return None
        ext = self._extent(L, t, members)
        x0, y0, x1, y1 = (0, 0, self.W, self.H) if ext is None else ext
        if x1 <= x0 or y1 <= y0:
            return None
        view = (self.ox, self.oy, self.W, self.H)
        self.ox, self.oy, self.W, self.H = self.ox + x0, self.oy + y0, x1 - x0, y1 - y0   # draw the children in that window
        img = np.zeros((self.H, self.W, 3), np.float32)
        alpha, self.alpha = self.alpha, np.zeros((self.H, self.W, 1), np.float32)
        mode, pad = getattr(self, 'mode', 'normal'), self._pad
        try:
            for C, q in members:
                self.mode = 'normal'
                self.layer(img, C, q, t)
            a = self.alpha
        finally:
            self.alpha, self.mode, self._pad = alpha, mode, pad
            self.ox, self.oy, self.W, self.H = view
        if not a.any():
            return None
        return x0, y0, img / np.maximum(a, 1e-6), a, None

    # ---------- the matrix path: mirrored, stretched, skewed, parented or turned in 3D ----------
    def draw_matrix(self, L, t, M, cx, cy):
        """draw the layer through a 3×3 matrix (content px → raster px). Text keeps FreeType's exact coverage with the
        matrix's linear part; vector shapes and plugin drawings go through Skia's canvas; pictures are first resampled by
        Lanczos to the size they show at and then placed by Skia with a cubic filter and anti-aliased edges."""
        import skia
        ty = L['type']
        affine = abs(M[2, 0]) < 1e-12 and abs(M[2, 1]) < 1e-12
        if ty in ('text', 'captions') and affine:
            lin = (M[0, 0], M[0, 1], M[1, 0], M[1, 1])
            if ty == 'text':
                return self.draw_text(L, t, cx, cy, 1.0, 0.0, lin)
            p = phrase_at(L, t)
            return self._text(L, p['text'], 'center', cx, cy, 1.0, 0.0, lambda ci: L['color'], None, lin) if p else None
        if ty == 'text':                                       # through perspective: drawn flat, then placed like a picture
            return self._flat_text(L, t, M)
        if ty in PLANES:
            return self._particles(L, t, M)
        if ty == 'shape' or ('plugin' in L and L['plugin'][1] != 'pixels'):
            if ty == 'shape':
                b = shape_path(L, t).computeTightBounds()
                g = (L['stroke']['width'] * 3 + 4) if L['stroke'] else 2
                box = (b.left() - g, b.top() - g, b.right() + g, b.bottom() + g)
            else:
                w, h = L['size'].at(t)
                g = float(L['params'].get('_bleed', 0) if not isinstance(L['params'].get('_bleed'), Prop) else L['params']['_bleed'].at(t)) + 2
                box = (-g, -g, w + g, h + g)
            r = self._quad_bounds(M, box)
            if r is None:
                return None
            X0, Y0, X1, Y1 = r
            ss = 2
            surf = skia.Surface((X1 - X0) * ss, (Y1 - Y0) * ss)
            c = surf.getCanvas()
            c.clear(skia.Color4f(0, 0, 0, 0))
            c.scale(ss, ss)
            c.translate(-X0, -Y0)
            c.concat(skia_matrix(M))
            if ty == 'shape':
                self._shape_paint(c, L, t)
            else:
                fn, _ = L['plugin']
                pv = {n: (v.at(t) if isinstance(v, Prop) else v) for n, v in L['params'].items()}
                fn(c, pv, w, h, Fx(math.sqrt(abs(np.linalg.det(M[:2, :2]))), t, t - L['in'], self.k))
            return self._skia_out(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType), X0, Y0, ss)
        if ty == 'image' and L.get('tile'):
            return self._tile(L, t, M)
        w, h = content_size(self.f, L, t)
        if w <= 0 or h <= 0:
            return None
        still = False
        if ty == 'image':
            im, still = load_plain(img_src(L, t, self.f.fps), L['crop'], L['grade']), L.get('seq') is None
            if L.get('flip'):
                M = M @ np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1.0]])
        elif ty == 'video':
            fr = video_frame(L, t)
            if L.get('delogo'):
                fr = delogo(fr, L['delogo'])
            a = apply_grade(crop_frame(fr, L['crop']), L['grade'])
            im = Image.fromarray(a[:, ::-1] if L.get('flip') else a, 'RGB')
        elif ty == 'solid':
            im = Image.new('RGBa', (8, 8), tuple(int(round(c)) for c in L['color']) + (255,))
        elif ty == 'comp':
            r = self._comps.get(L['id'])
            if r is None:
                r = self._comps[L['id']] = Renderer(L['film'], self.k)
            rgb, a = r.frame_rgba(comp_time(L, t))
            im = Image.fromarray(np.clip(np.round(np.concatenate([np.clip(rgb, 0, 255), a * 255], -1)), 0, 255).astype(np.uint8), 'RGBa')
        elif ty == 'gradient' and L.get('box') is not None:
            n = max(2, int(math.ceil(max(w, h) * self._mscale(M, (0, 0, w, h)))))
            gw, gh = max(2, int(round(n * w / max(w, h)))), max(2, int(round(n * h / max(w, h))))
            ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float32)
            col, al = gradient_field(L, (xs + 0.5) / gw, (ys + 0.5) / gh, w, h, t)
            im = Image.fromarray(np.clip(np.round(np.concatenate([col * al, al * 255], -1)), 0, 255).astype(np.uint8), 'RGBa')
        elif 'plugin' in L:
            pv = {n: (v.at(t) if isinstance(v, Prop) else v) for n, v in L['params'].items()}
            sc = self._mscale(M, (0, 0, w, h))
            px = np.asarray(L['plugin'][0](pv, w, h, Fx(sc, t, t - L['in'], self.k)), np.float32)
            px = px * (255.0 if px.max() <= 1.0 else 1.0)
            im = Image.fromarray(np.clip(np.round(np.concatenate([px[..., :3] * px[..., 3:] / 255, px[..., 3:]], -1)), 0, 255).astype(np.uint8), 'RGBa')
        else:
            raise ValueError(f'{L["id"]}: слой {ty} пока нельзя отражать, растягивать или поворачивать в 3D — заверните его в comp')
        return self._project(L, im, w, h, M, still)

    def _flat_text(self, L, t, M):
        """text turned in 3D (or pinned by its corners): its glyphs drawn flat in the layer's own plane at the largest
        size it shows at (exact FreeType coverage, as any text), with a margin for what reaches past the layout box,
        then that picture placed through M like an image — the far edge Lanczos-shrunk, a deep tilt from mips"""
        w, h = content_size(self.f, L, t)
        if w <= 0 or h <= 0:
            return None
        mrg = 0.35 * L['size'] + 4                             # descenders, italics, strokes past the layout box
        bw, bh = w + 2 * mrg, h + 2 * mrg
        sc = max(0.05, min(4.0, self._mscale(M, (0, 0, w, h)) * 1.15))
        sc = 2 ** (math.ceil(math.log2(sc) * 8) / 8)           # eighth-octave steps: the flat picture is kept between them
        flat = [(fx, {n: (v.at(t) if isinstance(v, Prop) else v) for n, v in fx['p'].items()}) for fx in (L.get('effects') or [])
                if getattr(fx['def'][0], 'kadr_flat', False)]
        ckey = ('flat', L['id'], text_of(L, t), sc, repr([(fx['def'][0].__name__, sorted(pv.items())) for fx, pv in flat]))
        hit = self._tcache.get(ckey)
        if hit is not None:
            im, bw, bh, ext = hit
            if flat:
                self._flat_fx.add(L['id'])
            return self._project(L, im, bw, bh, M @ np.array([[1, 0, -ext], [0, 1, -ext], [0, 0, 1.0]]), True)
        RW, RH = max(1, int(math.ceil(bw * sc))), max(1, int(math.ceil(bh * sc)))
        if RW * RH > 36e6:
            return None
        scx, scy = RW / bw, RH / bh
        pax, pay = pivot_px(self.f, L, t)
        view = (self.W, self.H, self.ox, self.oy, self._pad)
        self.W, self.H, self.ox, self.oy, self._pad = RW, RH, 0, 0, 0
        try:
            r = self.draw_text(L, t, (pax + mrg) * scx, (pay + mrg) * scy, 1.0, 0.0, (scx, 0.0, 0.0, scy))
        finally:
            self.W, self.H, self.ox, self.oy, self._pad = view
        if r is None:
            return None
        X0, Y0, rgb, a = r[:4]
        if not isinstance(rgb, np.ndarray) or not isinstance(a, np.ndarray):
            got = self._as_numpy(r)
            if got is None:
                return None
            X0, Y0, rgb, a = got
        pm = np.zeros((RH, RW, 4), np.float32)
        a = np.asarray(a, np.float32)
        rgb = np.broadcast_to(np.asarray(rgb, np.float32), a.shape[:2] + (3,))
        hh, ww = a.shape[:2]
        x0, y0 = max(0, X0), max(0, Y0)
        x1, y1 = min(RW, X0 + ww), min(RH, Y0 + hh)
        if x0 >= x1 or y0 >= y1:
            return None
        sa = a[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        pm[y0:y1, x0:x1, :3] = rgb[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] * sa
        pm[y0:y1, x0:x1, 3:] = sa * 255
        ext = mrg
        if flat:                                               # layer-space effects (print) on the flat picture, once
            al = pm[..., 3:] / 255
            col = pm[..., :3] / np.maximum(al, 1e-6)
            inv = np.array([[1 / scx, 0, -mrg], [0, 1 / scy, -mrg], [0, 0, 1.0]])
            for fx, pv in flat:
                got = fx['def'][0](np.ascontiguousarray(col, np.float32), np.ascontiguousarray(al, np.float32), pv,
                                   Fx(sc, t, t - L['in'], self.k, (0, 0), inv))
                col, al = got[0], got[1]
                if len(got) > 2:
                    gx = got[2][0] if isinstance(got[2], tuple) else got[2]
                    inv = inv @ np.array([[1, 0, -gx], [0, 1, -gx], [0, 0, 1.0]])
            g = (al.shape[1] - RW) / 2                         # the effects grew the picture by g px each side
            ext = mrg + g / scx
            bw, bh = bw + 2 * g / scx, bh + 2 * g / scy
            pm = np.concatenate([col * al, al * 255], -1)
            self._flat_fx.add(L['id'])
        im = Image.fromarray(np.clip(np.round(pm), 0, 255).astype(np.uint8), 'RGBa')
        if len([k for k in self._tcache if k[0] == 'flat']) > 32:
            for k in [k for k in self._tcache if k[0] == 'flat']:
                del self._tcache[k]
        self._tcache[ckey] = (im, bw, bh, ext)
        return self._project(L, im, bw, bh, M @ np.array([[1, 0, -ext], [0, 1, -ext], [0, 0, 1.0]]), True)

    @staticmethod
    def _as_numpy(r):
        return None

    # ---------- a picture breaking into grains / gathered from them ----------
    def draw_particles(self, L, t, cx, cy, s, rot):
        w, h = L['psize']
        fx_, fy_ = anchor(L, (w, h), t)
        th = math.radians(rot)
        c, si = math.cos(th), math.sin(th)
        M = np.array([[s * c, -s * si, cx], [s * si, s * c, cy], [0, 0, 1.0]]) @ np.array([[1, 0, -fx_ * w], [0, 1, -fy_ * h], [0, 0, 1.0]])
        return self._particles(L, t, M)

    draw_emitter = draw_plexus = draw_scene3d = draw_spectrum = draw_wave = draw_svg = draw_chart = draw_puppet = draw_particles   # particle systems and 3D scenes: the same placing

    def _particles(self, L, t, M):
        """the picture cut into cells of `cell` content px; a front crosses it (from `start`, `sweep` s, travelling
        `direction` degrees) and releases each cell as a grain of its mean colour that flies off with the wind, a spread
        of its own, turbulence and gravity, fading over `life` s (mode blow) — or the same flight backwards, the grains
        landing into the picture (mode gather). The cells not yet released are the picture itself. Grains are squares
        with exact edge coverage, drawn several times along their path within the shutter when they move fast. All of
        it in the layer's own plane, then placed through M like a picture"""
        from . import particles
        sc = max(0.05, min(3.0, self._mscale(M, (0, 0, *L['psize'])) * 1.05))
        sc = 2 ** (math.ceil(math.log2(sc) * 8) / 8)           # in eighth-octave steps: the picture is resized once a step
        # the grains already blur along their own paths within the shutter, so the samples a moving camera takes
        # across one frame all reuse one drawing of the frame's own moment (drawn for a view grown by 6 %, so the
        # camera's travel within the shutter stays inside it): v1 drew it again for each of up to 32 samples — 30-150 s
        # and 6 GB a frame on a reel's zoom
        tf = round(t * self.f.fps) / self.f.fps
        view = self._particle_view(M, L)
        key = (L['id'], tf, sc)
        hit = getattr(self, '_pcache', None)
        if not (hit and hit[0] == key and (view is None) == (hit[1] is None)
                and (view is None or (view[0] >= hit[1][0] and view[1] >= hit[1][1] and view[2] <= hit[1][2] and view[3] <= hit[1][3]))):
            if view is not None:
                gx, gy = (view[2] - view[0]) * 0.06, (view[3] - view[1]) * 0.06
                view = (view[0] - gx, view[1] - gy, view[2] + gx, view[3] + gy)
            if L['type'] == 'particles':
                got = particles.render(L, tf, sc, self.f.motion_blur, self.f.fps, view=view)
            elif L['type'] == 'scene3d':                       # drawn whole, at the size it shows at (up to twice its own)
                from . import scene3d
                k3 = self._mscale(M, (0, 0, *L['psize']))        # its own size unless the layer is shown larger (then up to twice)
                k3 = 1.0 if k3 < 1.02 else min(2.0, k3)
                a = scene3d.picture(L, L['time'].at(tf) if L.get('time') is not None else tf, self.f.fps, k3)
                # the core's colour is already times its coverage (samples added up over nothing): multiplying it by the
                # alpha once more darkened every edge that stood against the film behind — a white letter's edge pixel of
                # coverage 0.36 came out 0.17 (found by test_extrude.py against the same letter drawn flat)
                pm = np.concatenate([a[..., :3], a[..., 3:] * 255], -1)
                got = (0.0, 0.0, L['psize'][0], L['psize'][1], Image.fromarray(np.clip(np.round(pm), 0, 255).astype(np.uint8), 'RGBa'))
            elif L['type'] == 'svg':                           # a vector picture, drawn at the size it shows at
                from . import svg
                got = svg.render(L, sc)
            elif L['type'] == 'puppet':                        # a picture bent by pins
                from . import puppet
                got = puppet.render(self.f, L, tf, sc)
            elif L['type'] == 'chart':                         # bars, a line, a pie from numbers
                from . import chart
                got = chart.render(self.f, L, tf, sc)
            elif L['type'] in ('spectrum', 'wave'):            # sound as a picture: its frequencies, its curve
                from . import sound
                got = sound.render(self.f, L, tf, sc)
            else:
                from . import psys
                got = psys.render(L, tf, sc, self.f.motion_blur, self.f.fps, view, self.__dict__.setdefault('_psims', {}))
            self._pcache = hit = (key, view, got)
        got = hit[2]
        if got is None:
            return None
        bx0, by0, bw, bh, im = got
        return self._project(L, im, bw, bh, M @ np.array([[1, 0, bx0], [0, 1, by0], [0, 0, 1.0]]), False)

    def _particle_view(self, M, L):
        """the part of the layer's plane the frame can see (content px, padded by a few cells), so a picture zoomed far
        past the frame draws only what shows — or None when a frame corner lies behind the eye"""
        try:
            inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return None
        e = 8 + self._pad
        pts = np.array([[-e, -e, 1], [self.W + e, -e, 1], [-e, self.H + e, 1], [self.W + e, self.H + e, 1]], float) @ inv.T
        if (pts[:, 2] <= 1e-9).any():
            return None
        x, y = pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]
        m = 3 * max(1.0, L['cell'])
        return float(x.min()) - m, float(y.min()) - m, float(x.max()) + m, float(y.max()) + m

    def _tile_src(self, L, t):
        """the tiled picture's pyramid (kadr/mip.py), its texel map and wrap"""
        from . import mip
        path = img_src(L, t, self.f.fps)
        lv, pk, size, size0 = mip.tile_pyramid(path, os.path.getmtime(path), L['crop'], L['grade'])
        return lv, pk, size0, mip.tile_map(L, size, size0), mip.MIRROR if L['tile'] == 'mirror' else mip.REPEAT

    def _tile_bounds(self, M, w, h):
        """the output box of a plane: its corners' bounds, or the whole view when a corner lies behind the eye (the
        plane runs past the horizon; pixels there are left out per pixel)"""
        r = self._quad_bounds(M, (0, 0, w, h))
        if r is None:
            if abs(np.linalg.det(M)) < 1e-12:
                return None
            e = 2 + self._pad
            r = (-e, -e, self.W + e, self.H + e)
        return r

    def _tile(self, L, t, M):
        """an image tiled over its whole box (tile: repeat | mirror) through M — the mip sampler, so a floor running to
        the horizon neither shimmers in the distance nor blurs up close"""
        from . import mip
        w, h = content_size(self.f, L, t)
        if L.get('flip'):
            M = M @ np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1.0]])
        r = self._tile_bounds(M, w, h)
        if r is None or w <= 0 or h <= 0:
            return None
        lv, _, _, tm, wrap = self._tile_src(L, t)
        rgb, a = mip.place(lv, w, h, [list(np.linalg.inv(M).ravel())], r, tm, wrap)
        return r[0], r[1], rgb, a, None

    def _deep(self, M, w, h):
        """does the picture shrink more than twice from its near part to its far part within the view (a plane running
        into the distance)? Then one prepared size cannot serve it: it is drawn from a mip pyramid (kadr/mip.py)"""
        if abs(M[2, 0]) < 1e-12 and abs(M[2, 1]) < 1e-12:
            return False
        g = np.linspace(0, 1, 7)
        x, y = np.meshgrid(g * w, g * h)
        X = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        Y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        Z = M[2, 0] * x + M[2, 1] * y + M[2, 2]
        e = 64
        ok = Z > 1e-6
        Zs = np.where(ok, Z, 1.0)
        X, Y = X / Zs, Y / Zs
        ok &= (X >= -e) & (X <= self.W + e) & (Y >= -e) & (Y <= self.H + e)
        if ok.sum() < 2:
            return False
        # Jacobian of the projection at each point: (A − p·mᵀ) / z, A the linear part, m the perspective row
        a = (M[0, 0] - X * M[2, 0]) / Zs; b = (M[0, 1] - X * M[2, 1]) / Zs
        c = (M[1, 0] - Y * M[2, 0]) / Zs; d = (M[1, 1] - Y * M[2, 1]) / Zs
        p, q, r = a * a + b * b, a * c + b * d, c * c + d * d
        n = np.sqrt(0.5 * (p + r) + np.sqrt(np.maximum(0.0, 0.25 * (p - r) ** 2 + q * q)))[ok]
        return bool(n.max() > 0 and n.min() < 0.5 * n.max())

    def _mscale(self, M, box):
        """the largest screen px per content px over the box (a 3D turn shrinks the far edge)"""
        x0, y0, x1, y1 = box
        out = 0.0
        for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)):
            v = M @ np.array([x, y, 1.0])
            J = (M[:2, :2] * v[2] - np.outer(v[:2], M[2, :2])) / v[2] ** 2     # Jacobian of the projection at the point
            out = max(out, float(np.linalg.norm(J, 2)))
        return out

    def _quad_bounds(self, M, box):
        x0, y0, x1, y1 = box
        q = []
        for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            v = M @ np.array([x, y, 1.0])
            if v[2] <= 1e-6:
                return None
            q.append((v[0] / v[2], v[1] / v[2]))
        e = 2 + self._pad
        X0, Y0 = max(-e, math.floor(min(v[0] for v in q)) - 1), max(-e, math.floor(min(v[1] for v in q)) - 1)
        X1, Y1 = min(self.W + e, math.ceil(max(v[0] for v in q)) + 1), min(self.H + e, math.ceil(max(v[1] for v in q)) + 1)
        return (X0, Y0, X1, Y1) if X0 < X1 and Y0 < Y1 else None

    def _project(self, L, im, w, h, M, still):
        """a picture of content size w × h placed by matrix M: Lanczos to the size it shows at (a still one prepared once,
        15 % larger, then only gently rescaled — see _rotated), Skia places it with a cubic filter and smooth edges"""
        import skia
        r = self._quad_bounds(M, (0, 0, w, h))
        if r is None:
            return None
        X0, Y0, X1, Y1 = r
        need = max(1.0, w * self._mscale(M, (0, 0, w, h)))
        key = ('mat', L['id'])
        hit = self._tcache.get(key) if still else None
        if hit is None or not (0.6 * hit[1] <= need <= hit[1]) or hit[2] is not im:
            dw = max(1, int(math.ceil(need * (1.15 if still else 1.0))))
            dw = min(dw, max(im.size[0], int(math.ceil(need))))        # never blow a picture up beyond what is needed
            dh = max(1, int(round(dw * h / w)))
            a = np.asarray(im.resize((dw, dh), Image.LANCZOS), np.uint8)
            if a.shape[2] == 3:
                a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
            sk_img = skia.Image.fromarray(np.ascontiguousarray(a), colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
            if still:
                self._tcache[key] = (sk_img, dw, im)
        else:
            sk_img = hit[0]
        if self._deep(M, w, h):                                # into the distance: the mip sampler on the prepared picture
            from . import mip
            hit = self._tcache.get(('pyr', L['id']))
            if hit is None or hit[0] is not sk_img:
                px = sk_img.toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType).astype(np.float32)
                hit = self._tcache[('pyr', L['id'])] = (sk_img, mip.pyramid(np.concatenate([px[..., :3], px[..., 3:] / 255], -1)))
            lv = hit[1]
            rgb, a = mip.place(lv, w, h, [list(np.linalg.inv(M).ravel())], r, (lv[0].shape[1] / w, lv[0].shape[0] / h, 0.0, 0.0), mip.CLAMP)
            return X0, Y0, rgb, a, None
        surf = skia.Surface(X1 - X0, Y1 - Y0)
        c = surf.getCanvas()
        c.clear(skia.Color4f(0, 0, 0, 0))
        c.translate(-X0, -Y0)
        c.concat(skia_matrix(M))
        c.drawImageRect(sk_img, skia.Rect.MakeWH(w, h), skia.SamplingOptions(skia.CubicResampler.Mitchell()), skia.Paint(AntiAlias=True))
        return self._skia_out(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType), X0, Y0, 1)

    def draw_solid(self, L, t, cx, cy, s, rot):
        if rot:
            w, h = content_size(self.f, L, t)
            sw, sh = max(1, int(math.ceil(w))), max(1, int(math.ceil(h)))
            im = Image.new('RGBa', (sw, sh), tuple(int(round(c)) for c in L['color']) + (255,))
            return self._rotated(L, t, cx, cy, s, rot, im, (sw, sh), still=not L['size'].animated)
        X0, Y0, w, h = self._box(L, t, cx, cy, s)
        r = self._region(X0, Y0, X0 + w, Y0 + h)
        if not r:
            return None
        m = rect_mask(*r, (X0, Y0, X0 + w, Y0 + h))[..., None]
        return r[0], r[1], np.broadcast_to(np.array(L['color'], np.float32), m.shape[:2] + (3,)), m, None

    def _region(self, X0, Y0, X1, Y1):
        e = self._pad
        ix0, iy0, ix1, iy1 = max(-e, math.floor(X0)), max(-e, math.floor(Y0)), min(self.W + e, math.ceil(X1)), min(self.H + e, math.ceil(Y1))
        return (ix0, iy0, ix1, iy1) if ix0 < ix1 and iy0 < iy1 else None

    def _raster(self, L, t, cx, cy, s, src, size):
        X0, Y0, w, h = self._box(L, t, cx, cy, s)
        r = self._region(X0, Y0, X0 + w, Y0 + h)
        if not r:
            return None
        ix0, iy0, ix1, iy1 = r
        iw, ih = size
        fx, fy = iw / w, ih / h                               # source px per screen px
        box = (PAD + (ix0 - X0) * fx, PAD + (iy0 - Y0) * fy, PAD + (ix1 - X0) * fx, PAD + (iy1 - Y0) * fy)
        # a picture shrunk to a few px (fx > PAD source px per screen px): the window may poke past the padded edge
        box = (max(0.0, box[0]), max(0.0, box[1]), min(src.size[0], box[2]), min(src.size[1], box[3]))
        a = np.asarray(src.resize((ix1 - ix0, iy1 - iy0), Image.LANCZOS, box=box), np.float32)
        m = rect_mask(*r, (X0, Y0, X0 + w, Y0 + h))[..., None]
        if a.shape[2] == 4:                                   # premultiplied RGBa
            al = a[..., 3:] / 255
            return ix0, iy0, a[..., :3] / np.maximum(al, 1e-6), m * al, None
        return ix0, iy0, a, m, None

    def draw_image(self, L, t, cx, cy, s, rot):
        path = img_src(L, t, self.f.fps)
        # a motion-blur sample takes the turned picture's way: Lanczos once to (about) the shown size, kept, then each
        # sample placed by a cubic — not a full Lanczos from the source for every one of up to 32 samples. Moving
        # ≥ 2.5 px within the shutter, the difference is far below what the blur itself does to the detail
        if rot or (self._sampling and not L.get('flip')):
            return self._rotated(L, t, cx, cy, s, rot, load_plain(path, L['crop'], L['grade']), None, still=L.get('seq') is None)
        src, size = load_image(path, L['crop'], L['grade'], L.get('flip', False))
        return self._raster(L, t, cx, cy, s, src, size)

    def draw_video(self, L, t, cx, cy, s, rot):
        fr = video_frame(L, t)
        if L.get('delogo'):
            fr = delogo(fr, L['delogo'])
        if rot:
            a = apply_grade(crop_frame(fr, L['crop']), L['grade'])
            a = a[:, ::-1] if L.get('flip') else a
            return self._rotated(L, t, cx, cy, s, rot, Image.fromarray(a, 'RGB'), None)
        src, size = frame_raster(fr, L['crop'], L['grade'], L.get('flip'))
        return self._raster(L, t, cx, cy, s, src, size)

    def draw_comp(self, L, t, cx, cy, s, rot):
        """a nested composition: rendered on its own (transparent where empty) at the comp's local time, then moved,
        scaled and turned as one picture"""
        r = self._comps.get(L['id'])
        if r is None:
            for cid in [c for c in self._comps if not (self._ids[c]['in'] <= t < self._ids[c]['out'])]:
                del self._comps[cid]                            # a comp off screen drops its renderer and its caches
            r = self._comps[L['id']] = Renderer(L['film'], self.k)
        rgb, a = r.frame_rgba(comp_time(L, t))
        pm = np.concatenate([np.clip(rgb, 0, 255), a * 255], -1)
        pm = np.clip(np.round(pm), 0, 255).astype(np.uint8)
        if rot:
            return self._rotated(L, t, cx, cy, s, rot, Image.fromarray(pm, 'RGBa'), None)
        src = Image.fromarray(np.pad(pm, ((PAD, PAD), (PAD, PAD), (0, 0))), 'RGBa')
        return self._raster(L, t, cx, cy, s / self.k, src, (pm.shape[1], pm.shape[0]))   # rendered at k× already

    def _shape_paint(self, c, L, t, path=None, pe=None, trim=None, b=None):
        """fill, stroke (dash, trim), arrowheads of a shape on a canvas already in the shape's own coordinates"""
        import skia
        if path is None:
            path = shape_path(L, t)
            st, trim = L['stroke'], L['trim'].at(t)
            if st and (trim[0] > 0 or trim[1] < 1):
                if trim[1] <= trim[0]:
                    return
                pe = skia.TrimPathEffect.Make(float(trim[0]), float(trim[1]))
            b = path.computeTightBounds()
        from . import paths
        cs = paths.copies(L, t)
        if cs is not None and any(abs(o - 1) > 1e-6 for _, o in cs):   # a repeater whose copies fade: each drawn apart
            one = paths.shaped(L, t)
            ob = one.computeTightBounds()
            for M, o in cs:
                if o <= 0:
                    continue
                c.save()
                c.concat(M)
                c.saveLayerAlpha(None, int(round(255 * min(1.0, o))))
                self._paint_path(c, L, t, one, pe, trim, ob)
                c.restore()
                c.restore()
            return
        self._paint_path(c, L, t, path, pe, trim, b)

    def _paint_path(self, c, L, t, path, pe, trim, b):
        import skia
        st = L['stroke']
        if L['fill'] and path_closed(L):
            c.drawPath(path, paint_of(L['fill'], b))
        if st:
            paint = paint_of(st['color'], b)
            paint.setStyle(skia.Paint.kStroke_Style)
            paint.setStrokeWidth(st['width'])
            paint.setStrokeCap({'round': skia.Paint.kRound_Cap, 'butt': skia.Paint.kButt_Cap, 'square': skia.Paint.kSquare_Cap}[st['cap']])
            paint.setStrokeJoin({'round': skia.Paint.kRound_Join, 'miter': skia.Paint.kMiter_Join, 'bevel': skia.Paint.kBevel_Join}[st['join']])
            dash = skia.DashPathEffect.Make(list(st['dash']), float(st['dash_offset'].at(t))) if st['dash'] else None
            effects = [e for e in (pe, dash) if e]
            if effects:
                paint.setPathEffect(effects[0] if len(effects) == 1 else skia.PathEffect.MakeCompose(effects[1], effects[0]))
            c.drawPath(path, paint)
            if L['arrow']:
                for end in (('end', 'both'), ('start', 'both')):
                    if L['arrow'] not in end:
                        continue
                    head = arrow_head(path, trim[1] if end[0] == 'end' else trim[0], st['width'], forward=end[0] == 'end')
                    if head is not None:
                        c.drawPath(head, paint_of(st['color'], b))

    def draw_shape(self, L, t, cx, cy, s, rot, ss=2):
        """vector shape in its own coordinates (pivot `anchor`): fill, stroke, trim (draw-on), arrowheads; Skia with
        anti-aliasing at ss× the output resolution, box-reduced — edges move by fractions of a pixel without stepping"""
        import skia
        path = shape_path(L, t)
        st = L['stroke']
        trim = L['trim'].at(t)
        if st and (trim[0] > 0 or trim[1] < 1):
            if trim[1] <= trim[0]:
                return None
            pe = skia.TrimPathEffect.Make(float(trim[0]), float(trim[1]))
        else:
            pe = None
        ax, ay = L['anchor']
        b = path.computeTightBounds()
        grow = (st['width'] * 3 + 4) if st else 2
        th = math.radians(rot)
        pts = [(cx + s * ((x - ax) * math.cos(th) - (y - ay) * math.sin(th)), cy + s * ((x - ax) * math.sin(th) + (y - ay) * math.cos(th)))
               for x in (b.left() - grow, b.right() + grow) for y in (b.top() - grow, b.bottom() + grow)]
        e = 2 + self._pad
        X0, Y0 = max(-e, math.floor(min(q[0] for q in pts))), max(-e, math.floor(min(q[1] for q in pts)))
        X1, Y1 = min(self.W + e, math.ceil(max(q[0] for q in pts))), min(self.H + e, math.ceil(max(q[1] for q in pts)))
        if X0 >= X1 or Y0 >= Y1:
            return None
        surf = skia.Surface((X1 - X0) * ss, (Y1 - Y0) * ss)
        c = surf.getCanvas()
        c.clear(skia.Color4f(0, 0, 0, 0))
        c.scale(ss, ss)
        c.translate(cx - X0, cy - Y0)
        c.rotate(rot)
        c.scale(s, s)
        c.translate(-ax, -ay)
        self._shape_paint(c, L, t, path, pe, trim, b)
        return self._skia_out(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType), X0, Y0, ss)

    def draw_plugin(self, L, t, cx, cy, s, rot, ss=2):
        """a layer type from a plugin (@layer_type): the plugin draws its box (0..w, 0..h, centred on the position) on a
        Skia canvas already moved, scaled and turned — or returns pixels (h, w, 4) for the box at `fx.scale` px per unit"""
        import skia
        fn, kind = L['plugin']
        pv = {n: (v.at(t) if isinstance(v, Prop) else v) for n, v in L['params'].items()}
        w, h = L['size'].at(t)
        fx = Fx(s, t, t - L['in'], self.k)
        if kind == 'pixels':
            px = np.asarray(fn(pv, w, h, fx), np.float32)
            px = px * (255.0 if px.max() <= 1.0 else 1.0)
            pm = np.concatenate([px[..., :3] * px[..., 3:] / 255, px[..., 3:]], -1)
            pm = np.clip(np.round(pm), 0, 255).astype(np.uint8)
            if rot:
                return self._rotated(L, t, cx, cy, s, rot, Image.fromarray(pm, 'RGBa'), None)
            return self._raster(L, t, cx, cy, s, Image.fromarray(np.pad(pm, ((PAD, PAD), (PAD, PAD), (0, 0))), 'RGBa'), (pm.shape[1], pm.shape[0]))
        th = math.radians(rot)
        g = float(pv.get('_bleed', 0)) + 2                     # drawing may spill past the box (strokes, glows): `_bleed` px
        pts = [(cx + s * (x * math.cos(th) - y * math.sin(th)), cy + s * (x * math.sin(th) + y * math.cos(th)))
               for x in (-w / 2 - g, w / 2 + g) for y in (-h / 2 - g, h / 2 + g)]
        e = 2 + self._pad
        X0, Y0 = max(-e, math.floor(min(q[0] for q in pts))), max(-e, math.floor(min(q[1] for q in pts)))
        X1, Y1 = min(self.W + e, math.ceil(max(q[0] for q in pts))), min(self.H + e, math.ceil(max(q[1] for q in pts)))
        if X0 >= X1 or Y0 >= Y1:
            return None
        surf = skia.Surface((X1 - X0) * ss, (Y1 - Y0) * ss)
        c = surf.getCanvas()
        c.clear(skia.Color4f(0, 0, 0, 0))
        c.scale(ss, ss)
        c.translate(cx - X0, cy - Y0)
        c.rotate(rot)
        c.scale(s, s)
        c.translate(-w / 2, -h / 2)
        fn(c, pv, w, h, fx)
        return self._skia_out(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType), X0, Y0, ss)

    def draw_noise(self, L, t, cx, cy, s, rot):
        """static fractal value noise (like SVG feTurbulence fractalNoise): film grain, paper fibre, mottle"""
        key = (L['id'], self.W, self.H, self.ox, self.oy)
        if key not in self._grad:
            n = fractal_noise(self.f.W * self.k, self.f.H * self.k, L['frequency'], L['octaves'], L['seed'], self.k)
            n = n[self.oy:self.oy + self.H, self.ox:self.ox + self.W]
            a, b = L['levels']
            if L['color'] is None:
                self._grad[key] = (np.repeat(n[..., None] * 255, 3, -1), np.ones(n.shape + (1,), np.float32))
            else:
                self._grad[key] = (np.broadcast_to(np.array(L['color'], np.float32), n.shape + (3,)), np.clip(a * n + b, 0, 1)[..., None])
        col, al = self._grad[key]
        return 0, 0, col, al, None

    def draw_gradient(self, L, t, cx, cy, s, rot):
        """full-frame linear (CSS angle: 180 = top→bottom) or elliptical radial gradient of colour + alpha stops"""
        if L.get('box') is not None:                           # a gradient in its own box: position = centre, size, scale
            bw, bh = L['box'].at(t)
            X0, Y0, w, h = cx - s * bw / 2, cy - s * bh / 2, s * bw, s * bh
            r = self._region(X0, Y0, X0 + w, Y0 + h)
            if not r or w <= 0 or h <= 0:
                return None
            ys, xs = np.mgrid[r[1]:r[3], r[0]:r[2]].astype(np.float32)
            col, al = gradient_field(L, (xs + 0.5 - X0) / w, (ys + 0.5 - Y0) / h, w, h, t)
            al = al * rect_mask(*r, (X0, Y0, X0 + w, Y0 + h))[..., None]
            return r[0], r[1], col, al, None
        moving = L['center'].animated
        key = (L['id'], self.W, self.H, self.ox, self.oy) if not moving else None
        if key is not None and key not in self._grad and len(self._grad) > 32:
            self._grad.clear()
        if key is None or key not in self._grad:
            FW, FH = self.f.W * self.k, self.f.H * self.k
            ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
            xs, ys = (xs + self.ox + 0.5) / FW, (ys + self.oy + 0.5) / FH
            if L['kind'] == 'linear':
                a = math.radians(L['angle'])
                dx, dy = math.sin(a), -math.cos(a)
                half = 0.5 * (abs(dx) * FW + abs(dy) * FH)
                u = ((xs - 0.5) * FW * dx + (ys - 0.5) * FH * dy) / (2 * half) + 0.5
            else:
                (gx, gy), (rx, ry) = L['center'].at(t), L['radius']
                u = np.sqrt(((xs - gx) / rx) ** 2 + ((ys - gy) / ry) ** 2)
            col, al = stops_at(u, L['stops'], L.get('premul', True))
            if moving:
                return 0, 0, col, al, None
            self._grad[key] = (col, al)
        col, al = self._grad[key]
        return 0, 0, col, al, None

    def draw_dust(self, L, t, cx, cy, s, rot):
        """floating dust: each mote drifts at its own speed and wraps around, twinkles; drawn as a Gaussian at its exact
        sub-pixel position (no snapping), small ones sharp, big ones soft"""
        rng = np.random.default_rng(L['seed'])
        n, k, fr = L['count'], self.k, t * self.f.fps
        vx0, vx1, vy0, vy1 = L['drift']
        s0, s1, b0, b1 = L['sizes']
        a0, a1, ab = L['alphas']
        big = rng.random(n) > 0.86
        X = (rng.random(n) * 2000 + fr * (vx0 + rng.random(n) * (vx1 - vx0))) % 2000 - 40
        Y = (rng.random(n) * 1160 + fr * (vy0 + rng.random(n) * (vy1 - vy0)) + 1160 * 8) % 1160 - 40
        size = np.where(big, b0 + rng.random(n) * (b1 - b0), s0 + rng.random(n) * (s1 - s0))
        per, ph = 14 + rng.random(n) * 30, np.arange(n)
        alpha = np.where(big, ab, a0 + rng.random(n) * (a1 - a0)) * (0.55 + 0.45 * (0.5 + 0.5 * np.sin(fr / per + ph)))
        acc = np.zeros((self.H, self.W), np.float32)
        for x, y, sz, al, bg in zip(X * k - self.ox, Y * k - self.oy, size * k, alpha, big):
            sig = sz / 2 / 1.5 + (4.0 if bg else 0.4) * k      # disc of diameter sz, then CSS blur 4 / 0.4 px
            R = int(math.ceil(3 * sig)) + 1
            x0, y0 = int(math.floor(x)) - R, int(math.floor(y)) - R
            if x0 >= self.W or y0 >= self.H or x0 + 2 * R + 2 <= 0 or y0 + 2 * R + 2 <= 0:
                continue
            gx = np.exp(-0.5 * ((np.arange(2 * R + 2) + x0 + 0.5 - x) / sig) ** 2)
            gy = np.exp(-0.5 * ((np.arange(2 * R + 2) + y0 + 0.5 - y) / sig) ** 2)
            blob = np.outer(gy, gx) * al * min(1.0, (sz / 2) ** 2 * math.pi / (2 * math.pi * sig ** 2))
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + 2 * R + 2, self.W), min(y0 + 2 * R + 2, self.H)
            acc[ya:yb, xa:xb] += blob[ya - y0:yb - y0, xa - x0:xb - x0]
        return 0, 0, np.broadcast_to(np.array(L['color'], np.float32), acc.shape + (3,)), np.minimum(1, acc)[..., None], None

    def _glyphs(self, L, text, align, cx, cy, s, rot, colors, lin=None, sel=None, per=None):
        """(X0, Y0, {rgb: coverage}) for a laid-out text, turned by `rot` degrees around the layer position;
        colors(char_index) → rgb. With a `stroke` the glyph outlines are stroked (FreeType stroker) instead of filled."""
        glyphs, (bw, bh) = text_layout(L['font'], L['size'], text, align, L['line_height'], L.get('tracking', 0.0), L.get('features', ''))
        faces = [face(n) for n in str(L['font']).split('|')]
        if lin is None:                                        # layer px offset → frame px offset: turn and even scale
            th = math.radians(rot)
            co, si = math.cos(th), math.sin(th)
            lin = (s * co, -s * si, s * si, s * co)
        m00, m01, m10, m11 = lin
        mxx, mxy, myx, myy = m00, -m01, -m10, m11             # FreeType is y-up
        ax, ay = anchor(L, (bw, bh), getattr(self, '_now', 0.0))
        lx, ly = -bw * ax, -bh * ay                            # layout origin relative to the layer position, in layer px
        sm = math.sqrt(max(m00 * m00 + m10 * m10, m01 * m01 + m11 * m11))
        pad = sm * L['size'] + 6 + (sm * L.get('stroke', {}).get('width', 0) if L.get('stroke') else 0)
        q = [(m00 * x + m01 * y, m10 * x + m11 * y) for x in (lx, lx + bw) for y in (ly, ly + bh)]
        e = self._pad
        X0 = max(-e, math.floor(cx + min(v[0] for v in q) - pad))
        Y0 = max(-e, math.floor(cy + min(v[1] for v in q) - pad))
        X1 = min(self.W + e, math.ceil(cx + max(v[0] for v in q) + pad))
        Y1 = min(self.H + e, math.ceil(cy + max(v[1] for v in q) + pad))
        if X0 >= X1 or Y0 >= Y1:
            return None
        return self._glyph_raster(L, glyphs, faces, lin, (mxx, mxy, myx, myy), lx, ly, cx, cy, sm, X0, Y0, X1, Y1, colors, sel, per)

    def _glyph_raster(self, L, glyphs, faces, lin, base, lx, ly, cx, cy, sm, X0, Y0, X1, Y1, colors, sel, per):
        """FreeType's coverage of every glyph at its fractional place, summed per colour (capped at 1)"""
        m00, m01, m10, m11 = lin
        stroke = L.get('stroke')
        stroker = None
        if stroke:
            stroker = ft.Stroker()
            stroker.set(int(round(stroke['width'] * sm * 64 / 2)), ft.FT_STROKER_LINECAP_ROUND, ft.FT_STROKER_LINEJOIN_ROUND, 0)
        cov = {}
        for g, gx, gy, ci in glyphs:
            if sel is not None and ci not in sel:
                continue
            px, py = lx + gx, ly + gy
            if per is not None and ci in per:                  # a text animator moves and scales this glyph on its own
                dx, dy, gs, ux, uy = per[ci]
                px, py = lx + ux + gs * (gx - ux) + dx, ly + uy + gs * (gy - uy) + dy
                mxx, mxy, myx, myy = (v * gs for v in base)
            else:
                mxx, mxy, myx, myy = base
            X, Y = cx + m00 * px + m01 * py, cy + m10 * px + m11 * py   # glyph origin (baseline) in frame px, fractional
            ix, iy = math.floor(X), math.floor(Y)
            f, g = faces[g >> FALLBACK], g & ((1 << FALLBACK) - 1)
            f.set_char_size(int(round(L['size'] * 64)))
            f.set_transform(ft.Matrix(int(round(mxx * 65536)), int(round(mxy * 65536)), int(round(myx * 65536)), int(round(myy * 65536))),
                            ft.Vector(int(round((X - ix) * 64)), int(round(-(Y - iy) * 64))))
            f.load_glyph(g, FT_FLAGS)
            if stroker:
                gl = f.glyph.get_glyph()
                gl.stroke(stroker, True)
                bg = gl.to_bitmap(ft.FT_RENDER_MODE_NORMAL, ft.Vector(0, 0), True)
                bm, left, top = bg.bitmap, bg.left, bg.top
            else:
                f.glyph.render(ft.FT_RENDER_MODE_NORMAL)
                bm, left, top = f.glyph.bitmap, f.glyph.bitmap_left, f.glyph.bitmap_top
            if bm.rows == 0 or bm.width == 0:
                continue
            a = glyph_pixels(bm)
            x0, y0 = ix + left - X0, iy - top - Y0
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + bm.width, X1 - X0), min(y0 + bm.rows, Y1 - Y0)
            if xa < xb and ya < yb:
                c = cov.setdefault(colors(ci), np.zeros((Y1 - Y0, X1 - X0), np.float32))[ya:yb, xa:xb]
                np.minimum(1, c + a[ya - y0:yb - y0, xa - x0:xb - x0], out=c)
        for f in faces:
            f.set_transform(ft.Matrix(65536, 0, 0, 65536), ft.Vector(0, 0))
        return X0, Y0, cov

    def _text(self, L, text, align, cx, cy, s, rot, colors, key, lin=None):
        """one colour raster for a text; a text standing still (same scale, turn and sub-pixel position) is rasterized once"""
        key = (L['id'], text, key, round(s, 7), round(rot, 5), round(cx, 4), round(cy, 4), self._pad,
               tuple(round(v, 7) for v in lin) if lin else None)
        if key in self._tcache:
            return self._tcache[key]
        res = self._glyphs(L, text, align, cx, cy, s, rot, colors, lin)
        out = self._text_raster(*res, key) if res and res[2] else None
        if len(self._tcache) > 64:
            self._tcache = {k: v for k, v in self._tcache.items() if k[0] == 'rot'}
        self._tcache[key] = out
        return out

    def _text_raster(self, X0, Y0, cov, key):
        """{colour: coverage} → the text's raster (colours mixed by coverage)"""
        a = np.minimum(1, sum(cov.values()))
        rgb = sum(c[..., None] * np.array(col, np.float32) for col, c in cov.items()) / np.maximum(sum(cov.values()), 1e-6)[..., None]
        return X0, Y0, rgb, a[..., None], key

    def draw_text(self, L, t, cx, cy, s, rot, lin=None):
        if L.get('animate'):
            return self._animated_text(L, t, cx, cy, s, rot, lin)
        return self._text(L, text_of(L, t), L['align'], cx, cy, s, rot, lambda ci: L['color'], None, lin)

    def _animated_text(self, L, t, cx, cy, s, rot, lin):
        """a text animator: units at rest are drawn together; each moving unit gets its own raster, blur and opacity"""
        A = L['animate']
        text = text_of(L, t)
        desc, units = text_units(L['font'], L['size'], text, L['align'], L['line_height'], L.get('tracking', 0.0), L.get('features', ''), A['by'])
        rest, moving = set(), []
        for n, (cis, x0, x1, y) in enumerate(units):
            st = {k: v for k, v in self.f.ANIM_PROPS.items()} if hasattr(self.f, 'ANIM_PROPS') else {'x': 0.0, 'y': 0.0, 'scale': 1.0, 'opacity': 1.0, 'blur': 0.0}
            t0 = A['at'] + (min(cis) if A['by'] == 'letters' else n) * A['stagger']   # letters: spaces take their step too, as in CSS
            done = True
            for k, (v0, d, e) in A['props'].items():
                u = min(1.0, max(0.0, (t - t0) / d)) if d > 0 else (1.0 if t >= t0 else 0.0)
                q = ease(u, e)
                st[k] = v0 + (st[k] - v0) * q
                done &= u >= 1
            if done:
                rest |= cis
            elif st['opacity'] > 0:
                moving.append((cis, x0, x1, y, st))
        parts = []
        if rest:
            r = self._glyphs(L, text, L['align'], cx, cy, s, rot, lambda ci: L['color'], lin, sel=rest)
            if r and r[2]:
                parts.append((r[0], r[1], np.minimum(1, sum(r[2].values())), 0.0, 1.0))
        for cis, x0, x1, y, st in moving:
            uy = y + (desc if A['origin'] == 'bottom' else -L['size'] * 0.35)
            per = {ci: (st['x'], st['y'], st['scale'], (x0 + x1) / 2, uy) for ci in cis}
            r = self._glyphs(L, text, L['align'], cx, cy, s, rot, lambda ci: L['color'], lin, sel=cis, per=per)
            if r and r[2]:
                parts.append((r[0], r[1], np.minimum(1, sum(r[2].values())), st['blur'], st['opacity']))
        if not parts:
            return None
        sm = s if lin is None else math.sqrt(abs(lin[0] * lin[3] - lin[1] * lin[2]))
        return self._combine_parts(L, parts, sm)

    def _combine_parts(self, L, parts, sm):
        """letters in flight, each with its own blur and opacity, laid over each other like layers"""
        boxes = []
        for X0, Y0, c, b, o in parts:
            R = int(math.ceil(3 * b * sm)) + 1 if b > 0.05 else 0
            boxes.append((X0 - R, Y0 - R, X0 + c.shape[1] + R, Y0 + c.shape[0] + R))
        X0, Y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        X1, Y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        keep = np.ones((Y1 - Y0, X1 - X0), np.float32)       # 1 − alpha: units over each other combine like layers
        for (px0, py0, c, b, o), bx in zip(parts, boxes):
            if b > 0.05:
                R = px0 - bx[0]
                c = gauss_shift(np.pad(c, R), b * sm, 0, 0)
                px0, py0 = bx[0], bx[1]
            keep[py0 - Y0:py0 - Y0 + c.shape[0], px0 - X0:px0 - X0 + c.shape[1]] *= 1 - np.clip(c * o, 0, 1)
        a = (1 - keep)[..., None]
        return X0, Y0, np.broadcast_to(np.array(L['color'], np.float32), a.shape[:2] + (3,)), a, None

    def draw_captions(self, L, t, cx, cy, s, rot):
        p = phrase_at(L, t)
        colors, span = (lambda ci: L['color']), None
        if L['highlight']:
            pos = 0                                           # char range of the word being spoken (lines break only at spaces)
            for w, ws, we in p['words']:
                if ws <= t < we:
                    span = (pos, pos + len(w))
                pos += len(w) + 1
            if span:
                colors = lambda ci: L['highlight'] if span[0] <= ci < span[1] else L['color']
        return self._text(L, p['text'], 'center', cx, cy, s, rot, colors, span)


_dither = {}


# ---------------- vector shapes ----------------
def skia_matrix(M):
    import skia
    return skia.Matrix.MakeAll(float(M[0, 0]), float(M[0, 1]), float(M[0, 2]), float(M[1, 0]), float(M[1, 1]), float(M[1, 2]),
                               float(M[2, 0]), float(M[2, 1]), float(M[2, 2]))


def paint_of(spec, b):
    """a Skia paint for a shape's fill or stroke: a colour, or {gradient: linear | radial, stops, from/to | center/radius}
    in the shape's own coordinates (default: top → bottom of the shape; radial from its middle)"""
    import skia
    if not isinstance(spec, dict):                             # (r, g, b) or (r, g, b, alpha 0..1)
        return skia.Paint(AntiAlias=True, Color=skia.Color(*[int(round(v)) for v in spec[:3]], int(round(255 * (spec[3] if len(spec) > 3 else 1)))))
    cols = [skia.Color(*[int(round(c)) for c in col], int(round(al * 255))) for _, col, al in spec['stops']]
    pos = [st[0] for st in spec['stops']]
    fl = int(skia.GradientShader.kInterpolateColorsInPremul_Flag) if spec.get('premul', True) else 0   # as CSS (doc.interp_mode)
    if spec['kind'] == 'linear':
        p0 = spec.get('from') or (b.left(), b.top())
        p1 = spec.get('to') or (b.left(), b.bottom())
        sh = skia.GradientShader.MakeLinear([skia.Point(*p0), skia.Point(*p1)], cols, pos, flags=fl)
    else:
        c = spec.get('center') or (b.centerX(), b.centerY())
        sh = skia.GradientShader.MakeRadial(skia.Point(*c), float(spec.get('radius') or max(b.width(), b.height()) / 2), cols, pos, flags=fl)
    return skia.Paint(AntiAlias=True, Shader=sh)


class Fx:
    """what an effect is told: scale (screen px per layer px — multiply sizes by it), film time t, layer time lt, k;
    origin — the raster px of the picture's top-left pixel; inv — raster px → the layer's own content px (3×3, None
    where unknown), so a texture an effect makes can stick to the layer as it moves; helpers to blur and grow"""
    def __init__(self, scale, t, lt, k, origin=(0, 0), inv=None, J=None):
        self.scale, self.t, self.lt, self.k = scale, t, lt, k
        self.origin, self.inv, self.J = origin, inv, J        # J: raster px per layer px (2×2) around the pivot

    def layer_cov(self, sigma):
        """the covariance (px²) of a blur of `sigma` layer px as the frame sees it (isotropic × scale without J)"""
        J = self.J if self.J is not None else np.eye(2) * self.scale
        return sigma * sigma * (J @ J.T)

    def blur_layer(self, x, sigma):
        """Gaussian blur of `sigma` layer px, squashed and turned with the layer (an effect with space: layer)"""
        return blur_cov(x, self.layer_cov(sigma)) if sigma > 0 else x

    def content_xy(self, h, w):
        """the layer's content px (u, v) of every pixel of an (h, w) picture at `origin` (None without inv)"""
        if self.inv is None:
            return None
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        X, Y = xs + self.origin[0] + 0.5, ys + self.origin[1] + 0.5
        m = self.inv
        z = m[2, 0] * X + m[2, 1] * Y + m[2, 2]
        return (m[0, 0] * X + m[0, 1] * Y + m[0, 2]) / z, (m[1, 0] * X + m[1, 1] * Y + m[1, 2]) / z

    @staticmethod
    def blur(x, sigma):
        """Gaussian blur of an (h, w, c) array, same size (pad first if the result must spread)"""
        return gauss_shift(x, sigma, 0, 0) if sigma > 0.25 else x

    @staticmethod
    def grow(rgb, a, n):
        """(rgb, a, n): the raster with n transparent px on every side — return this from an effect that spreads"""
        n = int(n)
        return np.pad(rgb, ((n, n), (n, n), (0, 0)), mode='edge'), np.pad(a, ((n, n), (n, n), (0, 0))), n


def svg_path(d):
    """SVG path data (M L H V C Q Z, absolute and relative) → skia.Path"""
    import skia
    toks = re.findall(r'[MmLlHhVvCcQqAaZz]|-?\d*\.?\d+(?:e-?\d+)?', d)
    p, i, cmd, x, y, sx, sy = skia.Path(), 0, None, 0.0, 0.0, 0.0, 0.0
    num = lambda: float(toks.pop(0))
    while toks:
        if re.match(r'[A-Za-z]', toks[0]):
            cmd = toks.pop(0)
            if cmd in 'Zz':
                p.close()
                x, y = sx, sy
                continue
        rel = cmd.islower()
        c = cmd.upper()
        ox, oy = (x, y) if rel else (0.0, 0.0)
        if c == 'M':
            x, y = ox + num(), oy + num()
            p.moveTo(x, y)
            sx, sy = x, y
            cmd = 'l' if rel else 'L'
        elif c == 'L':
            x, y = ox + num(), oy + num()
            p.lineTo(x, y)
        elif c == 'H':
            x = ox + num()
            p.lineTo(x, y)
        elif c == 'V':
            y = oy + num()
            p.lineTo(x, y)
        elif c == 'C':
            x1, y1, x2, y2, x3, y3 = ox + num(), oy + num(), ox + num(), oy + num(), ox + num(), oy + num()
            p.cubicTo(x1, y1, x2, y2, x3, y3)
            x, y = x3, y3
        elif c == 'Q':
            x1, y1, x2, y2 = ox + num(), oy + num(), ox + num(), oy + num()
            p.quadTo(x1, y1, x2, y2)
            x, y = x2, y2
        elif c == 'A':                                         # rx ry x-rotation large-arc sweep x y
            rx, ry, rot_, large, sweep = num(), num(), num(), num(), num()
            x2, y2 = ox + num(), oy + num()
            p.arcTo(rx, ry, rot_, skia.Path.ArcSize.kLarge_ArcSize if large else skia.Path.ArcSize.kSmall_ArcSize,
                    skia.PathDirection.kCW if sweep else skia.PathDirection.kCCW, x2, y2)
            x, y = x2, y2
        else:
            toks.pop(0)
    return p


def shape_path(L, t):
    """the layer's whole outline at t — kadr/paths.py (morphs, merged paths, offset, text outlines, every copy of the
    repeater): its bounds are what the layer covers"""
    from . import paths
    return paths.final(L, t)


def path_closed(L):
    return L['shape'] in ('rect', 'ellipse', 'polygon', 'text') or L['closed'] or bool(L.get('paths')) or \
        (L['shape'] == 'path' and 'z' in (L['d'] or '').lower())


def arrow_head(path, at, width, forward=True):
    """a filled triangle at fraction `at` of the outline, pointing along it"""
    import skia
    m = skia.PathMeasure(path, False)
    length = m.getLength()
    if length <= 0:
        return None
    pos, tan = m.getPosTan(max(0.0, min(length, at * length)))
    if pos is None:
        return None
    dx, dy = (tan.x(), tan.y()) if forward else (-tan.x(), -tan.y())
    size = max(8.0, width * 4)
    bx, by = pos.x() - dx * size, pos.y() - dy * size
    head = skia.Path()
    head.moveTo(pos.x() + dx * width, pos.y() + dy * width)
    head.lineTo(bx - dy * size * 0.55, by + dx * size * 0.55)
    head.lineTo(bx + dy * size * 0.55, by - dx * size * 0.55)
    head.close()
    return head


def to_u8(img):
    """float → 8 bit with a fixed (frame-to-frame identical) triangular dither of ±1 level: smooth dark gradients
       otherwise fall into a few wide bands («rings»), and a static pattern adds no flicker"""
    if img.shape not in _dither:
        rng = np.random.default_rng(12345)
        _dither[img.shape] = (rng.random(img.shape[:2], np.float32) + rng.random(img.shape[:2], np.float32) - 1)[..., None]
    return np.clip(np.round(img + _dither[img.shape]), 0, 255).astype(np.uint8)


STENCILS = ('stencil-alpha', 'stencil-luma', 'silhouette-alpha', 'silhouette-luma')


def blend_mode(mode, a, b):
    """a: what is under (0..1), b: the layer (0..1) → the blended colour, before opacity"""
    if mode == 'multiply':
        return a * b
    if mode == 'screen':
        return 1 - (1 - a) * (1 - b)
    if mode == 'add':
        return np.minimum(1, a + b)
    if mode == 'overlay':
        return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == 'soft-light':                                   # W3C formula
        d = np.where(a <= 0.25, ((16 * a - 12) * a + 4) * a, np.sqrt(a))
        return np.where(b <= 0.5, a - (1 - 2 * b) * a * (1 - a), a + (2 * b - 1) * (d - a))
    if mode == 'darken':
        return np.minimum(a, b)
    if mode == 'lighten':
        return np.maximum(a, b)
    if mode == 'color-dodge':                                  # W3C compositing: the separable modes, as CSS mix-blend-mode
        return np.where(a <= 0, 0.0, np.where(b >= 1, 1.0, np.minimum(1.0, a / np.maximum(1 - b, 1e-6))))
    if mode == 'color-burn':
        return np.where(a >= 1, 1.0, np.where(b <= 0, 0.0, 1 - np.minimum(1.0, (1 - a) / np.maximum(b, 1e-6))))
    if mode == 'hard-light':
        return np.where(b <= 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == 'difference':
        return np.abs(a - b)
    if mode == 'exclusion':
        return a + b - 2 * a * b
    if mode == 'subtract':
        return np.maximum(0.0, a - b)
    if mode in ('hue', 'saturation', 'color', 'luminosity'):   # the non-separable four
        if mode == 'hue':
            return _set_lum(_set_sat(b, _sat(a)), _lum(a))
        if mode == 'saturation':
            return _set_lum(_set_sat(a, _sat(b)), _lum(a))
        if mode == 'color':
            return _set_lum(b, _lum(a))
        return _set_lum(a, _lum(b))
    return b


def _lum(c):
    return (c @ np.array([0.3, 0.59, 0.11], np.float32))[..., None]


def _sat(c):
    return c.max(-1, keepdims=True) - c.min(-1, keepdims=True)


def _set_sat(c, s):
    lo, hi = c.min(-1, keepdims=True), c.max(-1, keepdims=True)
    return np.where(hi > lo, (c - lo) * s / np.maximum(hi - lo, 1e-12), 0.0)


def _set_lum(c, l):
    c = c + (l - _lum(c))
    L, n, x = _lum(c), c.min(-1, keepdims=True), c.max(-1, keepdims=True)
    c = np.where(n < 0, L + (c - L) * L / np.maximum(L - n, 1e-12), c)
    return np.where(x > 1, L + (c - L) * (1 - L) / np.maximum(x - L, 1e-12), c)


def fractal_noise(W, H, freq, octaves, seed, k=1):
    """sum of octaves of smoothly interpolated random lattices; lattice step = 1/frequency frame px; 0..1 around 0.5"""
    rng = np.random.default_rng(seed)
    out, amp, tot = np.zeros((H, W), np.float32), 1.0, 0.0
    fx, fy = freq
    for o in range(octaves):
        gx, gy = max(2, int(np.ceil(W / k * fx * 2 ** o)) + 2), max(2, int(np.ceil(H / k * fy * 2 ** o)) + 2)
        g = Image.fromarray(rng.random((gy, gx), dtype=np.float32), 'F')
        sx, sy = (gx - 2) / W, (gy - 2) / H
        out += amp * np.asarray(g.resize((W, H), Image.BICUBIC, box=(0.5, 0.5, 0.5 + sx * W, 0.5 + sy * H)), np.float32)
        tot += amp
        amp *= 0.5
    return np.clip(out / tot, 0, 1)
