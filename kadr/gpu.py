"""Kadr's renderer with the pixel work on the GPU (kadr_core: Rust + wgpu, Metal on a Mac).

Everything the CPU Renderer decides stays its decision — the order of layers, placements, opacity, blur, shadows,
masks, groups, motion-blur samples — this class only changes where pixels are made: the canvas, collapsing groups,
mattes and motion-blur accumulators are GPU pictures; pictures without a turn are resampled there (Pillow's Lanczos,
ported); blur, shadows, filters and compositing run there. Layers not ported yet (text, shapes, turned pictures,
footage, plugin drawings) are still drawn by the CPU code into numpy rasters and uploaded; effects from plugins get
the picture back on the CPU. Each ported step is tested against the numpy code it replaces (core/tests)."""
import ctypes, threading
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import kadr_core as kc
from .doc import Prop
from .render import shadow_cov, cov_reach, video_frame, comp_time, FT_FLAGS, FALLBACK, Renderer, place, Fx, load_image, load_plain, img_src, PAD, reader, video_time, delogo, crop_frame, apply_grade, content_size, anchor

PM = ('premultiplied',)                                     # a raster's colour is premultiplied (see up)
ADDED = ('added',)                                         # a result already put into its sink (accumulator or canvas)


class Sink(tuple):
    """where a layer's picture may go straight from the kernel that makes it, with no picture in between: mode 1 — added
    into a motion-blur accumulator `img` whose frame box is `box` (only if it fits inside); mode 2 — laid over the canvas
    `img` (box: its origin) with `opacity` and `clip`, as composite() would; mode 4 — a motion-blur sample through a
    matrix noted in the list `img` (drawn with the others in one pass). Only for the layer `L` itself"""
    __slots__ = ()

    def __new__(cls, img, box, L, mode, opacity=1.0, clip=None):
        return super().__new__(cls, (img, box, L, mode, opacity, clip))

    def args(self, X0, Y0, X1, Y1):
        img, box, _, mode, op, clip = self
        if mode == 4:                                          # noting placements (_placed only): others draw a picture
            return None
        if mode == 1 and not (box[0] <= X0 and box[1] <= Y0 and X1 <= box[2] and Y1 <= box[3]):
            return None
        return dict(acc=img, ax=X0 - box[0], ay=Y0 - box[1], mode=mode, opacity=op, clip=clip)
MODES = {'normal': 0, 'multiply': 1, 'screen': 2, 'overlay': 3, 'soft-light': 4, 'add': 5, 'darken': 6, 'lighten': 7, 'color-dodge': 8,
         'color-burn': 9, 'hard-light': 10, 'difference': 11, 'exclusion': 12, 'subtract': 13, 'hue': 14, 'saturation': 15, 'color': 16,
         'luminosity': 17, 'stencil-alpha': 0, 'stencil-luma': 0, 'silhouette-alpha': 0, 'silhouette-luma': 0}
GRADE_OPS = {'contrast': 1, 'saturation': 2, 'sepia': 3, 'brightness': 4, 'gamma': 5}


def _inside(need, crop, dw, dh):
    """the pixels `need` (None: all) lie inside `crop` of a dw × dh picture"""
    if need is None:
        return crop == (0, 0, dw, dh)
    return crop[0] <= max(0, need[0]) and crop[1] <= max(0, need[1]) and min(dw, need[2]) <= crop[2] and min(dh, need[3]) <= crop[3]


def _mtime(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


_DECODERS = None                                           # background decoding of image-sequence frames


def decode(path, crop):
    """a picture file → its 8-bit RGB or RGBA (as render.load_image reads it, before the grade) and its size"""
    im = Image.open(path)
    if im.mode not in ('RGB', 'RGBA'):
        im = im.convert('RGBA')
    if crop:
        x, y, w, h = crop
        im = im.crop((int(x), int(y), int(x + w), int(y + h)))
    return np.ascontiguousarray(np.asarray(im)), im.size


_FT = False


def _ft_ready():
    """hand Rust the FreeType that Python has loaded: its functions' addresses and the struct offsets from freetype-py's
    own definitions — the glyphs are then the same bytes as from Python"""
    global _FT
    if _FT:
        return
    import freetype.raw as fr
    from freetype.ft_structs import FT_GlyphSlotRec, FT_FaceRec, FT_Bitmap
    addr = lambda fn: ctypes.cast(fn, ctypes.c_void_p).value
    kc.ft_setup([addr(fr.FT_Set_Char_Size), addr(fr.FT_Set_Transform), addr(fr.FT_Load_Glyph), addr(fr.FT_Render_Glyph)],
                [FT_FaceRec.glyph.offset, FT_GlyphSlotRec.bitmap.offset, FT_GlyphSlotRec.bitmap_left.offset,
                 FT_GlyphSlotRec.bitmap_top.offset, FT_Bitmap.rows.offset, FT_Bitmap.width.offset, FT_Bitmap.pitch.offset,
                 FT_Bitmap.buffer.offset])
    _FT = True


_REDRAW = object()              # _shape_pic: too large to keep as a picture — draw it the CPU engine's way

_thread_mark = threading.local()


def _here():
    """this thread's own mark (a thread's number is given again to a later thread once it has ended — a mark is not)"""
    m = getattr(_thread_mark, 'mark', None)
    if m is None:
        m = _thread_mark.mark = object()
    return m


class GpuRenderer(Renderer):
    def __init__(self, film, k=1, view=None):
        super().__init__(film, k, view)
        self._src8, self._gplates, self._dither = {}, {}, None
        self._sink = None
        self._ahead = {}                                       # (file, crop) → its decoding in the background
        self._comp_still = {}                                  # comp id → (layers on, its picture) while all stand still
        self._thread = _here()                                 # what it puts on the GPU lives on this thread's device

    # ---------- frames ----------
    def frame_gpu(self, t):
        if _here() is not self._thread:                        # said in words: the core itself dies here («non-existent resource»)
            raise RuntimeError('этот рендерер создан в другом потоке: у каждого потока своё устройство видеокарты, и картинки рендерера '
                               'лежат на устройстве того потока. Создавайте GpuRenderer в том же потоке, где просите кадры '
                               '(или держите один поток для всей работы с GPU, как kadr serve)')
        layers = self.seen(t)
        n = 0                                                  # the bottom run of still layers is composited once
        while n < len(layers) and self.static(layers[n][0], t):
            n += 1
        key = tuple(L['id'] for L, _ in layers[:n])
        cached = bool(n) and key in self._gplates
        canvas = kc.copy(self._gplates[key]) if cached else kc.new_img(self.W, self.H, [*map(float, self.f.background), 1.0])
        for j, (L, p) in enumerate(layers):
            if j < n and cached:
                continue
            if j == n and n and not cached:
                if len(self._gplates) > 4:
                    self._gplates.clear()
                self._gplates[key] = kc.copy(canvas)
            self.layer(canvas, L, p, t)
        if n == len(layers) and n and not cached:
            self._gplates[key] = kc.copy(canvas)
        return canvas

    def frame_comp(self, t):
        """the canvas of this film as a nested comp: over a transparent ground unless the comp has a background
        (render.py frame_rgba)"""
        bg = self.f.background
        canvas = kc.new_img(self.W, self.H, [*map(float, bg), 1.0] if bg is not None else [0.0, 0, 0, 0])
        for L, p in self.order(t):
            self.layer(canvas, L, p, t)
        return canvas

    def frame_rgba_start(self, t):
        """the frame over nothing (export with transparency), sent to the GPU and not waited for: `.get()` gives 8-bit
        RGBA with straight colour — cli._rgba's arithmetic done on the GPU (straight8.wgsl) instead of numpy"""
        self.f.background = None
        return kc.to_rgba8_start(self.frame_comp(t))

    def frame_u8(self, t):
        """the frame as 8-bit RGB with Kadr's fixed dither (render.to_u8)"""
        if self._dither is None:
            rng = np.random.default_rng(12345)
            d = rng.random((self.H, self.W), np.float32) + rng.random((self.H, self.W), np.float32) - 1
            self._dither = kc.upload_plane(np.ascontiguousarray(d))
        return kc.to_u8(self.frame_gpu(t), self._dither)[..., :3]

    def frame_rgba8(self, t):
        """the same as 8-bit RGBA (alpha 255) — straight to an encoder reading rgba, no repacking"""
        return self.frame_start(t).get()

    def frame_start(self, t):
        """the frame sent to the GPU, not waited for: `.get()` on the result gives its 8-bit RGBA. A renderer keeps the
        next frame's CPU work going while the GPU draws this one"""
        if self._dither is None:
            rng = np.random.default_rng(12345)
            d = rng.random((self.H, self.W), np.float32) + rng.random((self.H, self.W), np.float32) - 1
            self._dither = kc.upload_plane(np.ascontiguousarray(d))
        return kc.to_u8_start(self.frame_gpu(t), self._dither)

    def frame(self, t):
        """float colour like Renderer.frame (for tools that want it)"""
        return self.frame_gpu(t).download()[..., :3]

    # ---------- a raster on the GPU ----------
    @staticmethod
    def up(ras):
        """(X0, Y0, rgb, a, key) → (X0, Y0, Img, k); a GPU raster is (X0, Y0, Img, None, None) already, or
        (X0, Y0, Img, PM, k) — premultiplied colour to be × k (a group's canvas: k = 1; an accumulator: 1 / samples)
        and divided back where it is used, not by a pass of its own. k = 0: not premultiplied"""
        X0, Y0, rgb, a = ras[:4]
        if isinstance(rgb, kc.Img):
            return X0, Y0, rgb, (ras[4] if a is PM else 0.0)
        a = np.asarray(a, np.float32)
        if a.size == 0:
            return None
        rgb = np.broadcast_to(np.asarray(rgb, np.float32), a.shape[:2] + (3,))
        return X0, Y0, kc.upload(np.ascontiguousarray(np.concatenate([rgb, a], -1), np.float32)), 0.0

    # ---------- one layer ----------
    def layer(self, canvas, L, p, t):
        from .render import transition, chain_opacity, STENCILS
        if L.get('posterize') and L['id'] not in self._held:   # Posterize Time: the layer as it was at its last step
            fps_ = L['posterize']
            tq = L['in'] + math.floor((t - L['in']) * fps_ + 1e-6) / fps_
            self._held.add(L['id'])
            try:
                q = place(self.f, L, tq)
                if q is not None:
                    self.layer(canvas, L, q, tq)
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
                        self.layer(canvas, L, q, tk)
            finally:
                self._dim = 1.0
                self._echoing.discard(L['id'])
        fade, clip = transition(self.f, L, t)
        op = min(1.0, L['opacity'].at(t)) * fade * self._dim
        if L.get('parent_layer') is not None:
            op *= chain_opacity(L, t)
        if op <= 0:
            return
        if L.get('blend', 'normal') in STENCILS and not self._stencil:   # render.py: the canvas so far × the layer's matte
            kind, by = L['blend'].split('-')
            self._stencil = True
            try:
                A = self.matte({'src': L['id'], 'mode': by + ('_inverted' if kind == 'silhouette' else '')}, t)
            finally:
                self._stencil = False
            kept = kc.new_img(canvas.w, canvas.h, [0.0, 0, 0, 0])
            if A is not None:
                kc.composite(kept, canvas, 0, 0, 1.0, None, A, 0, 0, 0, None, 1.0)
            kc.put(canvas, kept)
            return
        k, ox, oy = self.k, self.ox, self.oy
        s = p[2]
        clip = [clip[0] * k - ox, clip[1] * k - oy, clip[2] * k - ox, clip[3] * k - oy] if clip else None
        sk = s * k
        bv = L['blur'].at(t) if L.get('blur') else 0.0
        bx, by = (bv[0] * sk, bv[1] * sk) if isinstance(bv, tuple) else (bv * sk, bv * sk)
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
        self._pad = int(math.ceil(reach))
        # a plain layer (nothing done to its picture after it is made) is laid onto the canvas by the kernel that makes it
        plain = (max(bx, by) <= 0.25 and not L.get('filter') and not fxs and not L.get('shadow') and M is None
                 and L.get('blend', 'normal') == 'normal')
        outer = self._sink
        try:
            n, h = self.mb_samples(L, t)
            if n <= 1:
                self._sink = Sink(canvas, (0, 0), L, 2, float(op), clip) if plain else None
                ras = self._draw_at(L, t, p)
            else:
                ras = self._motion_blur(L, t, n, h)
        finally:
            self._pad = 0
            self._sink = outer
        if ras is None or ras is ADDED:
            return
        g = self.up(ras)
        if g is None:
            return
        X0, Y0, im, k = g
        if bcov is not None and max(bx, by) > 0.25:            # in the layer's own px (render.py): two passes along its axes
            Rx, Ry = int(math.ceil(3 * bx)) + 1, int(math.ceil(3 * by)) + 1
            pmi = kc.padded(im if k == 1 else kc.pm(self._plain(im, k) if k else im, 0), im.w + 2 * Rx, im.h + 2 * Ry, Rx, Ry)
            im, k = kc.pm(self._gblur_cov(pmi, bcov), 1), 0.0
            X0, Y0 = X0 - Rx, Y0 - Ry
        elif max(bx, by) > 0.25:                               # CSS blur(r): a Gaussian of σ = r (all of it, no shortcut)
            Rx, Ry = int(math.ceil(3 * bx)) + 1, int(math.ceil(3 * by)) + 1
            pmi = kc.padded(im if k == 1 else kc.pm(self._plain(im, k) if k else im, 0), im.w + 2 * Rx, im.h + 2 * Ry, Rx, Ry)
            im, k = kc.pm(kc.gauss(pmi, bx, by), 1), 0.0
            X0, Y0 = X0 - Rx, Y0 - Ry
        if k and (L.get('filter') or fxs or (L.get('shadow') and k != 1)):
            im, k = self._plain(im, k), 0.0                    # these need the colour itself
        if L.get('filter'):
            im = kc.grade(im, [(GRADE_OPS[k_], float(v.at(t))) for k_, v in L['filter']])
        flat_done = L['id'] in self._flat_fx                   # drawn flat: its layer-space effects are on already
        self._flat_fx.discard(L['id'])
        for (fx, pv), mp in zip(fxs, maps):                    # plugin effects work on numpy: the picture comes back
            if flat_done and getattr(fx['def'][0], 'kadr_flat', False):
                continue
            if getattr(fx['def'][0], 'kadr_gpu', False):       # a shader: the picture never leaves the GPU
                fxo = Fx(sk, t, t - L['in'], self.k, (X0, Y0), self._raster_inv(L, p, t), J)
                fxo.frame, fxo.map, fxo.pivot = (self.W, self.H), mp, self._fx_pivot(fxo.inv, p)
                got = fx['def'][0](im, pv, fxo)
                im, gx = got if isinstance(got, tuple) else (got, 0)
                X0, Y0 = X0 - gx, Y0 - gx
                continue
            if fx['def'][0].__name__ == 'motion_tile' and X0 == 0 and Y0 == 0 and im.w == self.W and im.h == self.H:
                tw, th = pv['tile'][0] * self.k, pv['tile'][1] * self.k
                cx_, cy_ = pv['center'][0] * self.k, pv['center'][1] * self.k
                im = kc.motion_tile(im, [int(round(cx_ - tw / 2)), int(round(cy_ - th / 2)), int(round(cx_ + tw / 2)), int(round(cy_ + th / 2))],
                                    bool(pv['mirror']))
                continue
            arr = im.download()
            got = fx['def'][0](np.ascontiguousarray(arr[..., :3]), np.ascontiguousarray(arr[..., 3:]), pv,
                               Fx(sk, t, t - L['in'], self.k, (X0, Y0), self._raster_inv(L, p, t), J))
            rgb, a = got[0], got[1]
            if len(got) > 2:
                gx, gy = got[2] if isinstance(got[2], tuple) else (got[2], got[2])
                X0, Y0 = X0 - gx, Y0 - gy
            im = self.up((0, 0, rgb, a))[2]
        for sh in L.get('shadow') or []:
            sb, so, sop = sh['blur'].at(t), sh['offset'].at(t), sh['opacity'].at(t)
            scov = None
            if sh.get('space') == 'layer':
                dx, dy = J[0, 0] * so[0] + J[0, 1] * so[1], J[1, 0] * so[0] + J[1, 1] * so[1]
                bs = math.sqrt(abs(np.linalg.det(J)))
                scov = shadow_cov(J, sb)
            else:
                dx, dy, bs = so[0] * sk, so[1] * sk, sk
            sig = sb * bs / 2
            sp = sh['spread'].at(t) * bs                       # CSS spread, px (render.spread_alpha)
            WX0, WY0, src = X0, Y0, im
            Hm = self._shadow_h(L, p, t, so) if sh.get('space') == 'layer' else None
            if Hm is not None:                                 # a 3D layer: the offset through its perspective (render.py)
                Mc = Hm @ np.array([[1, 0, X0], [0, 1, Y0], [0, 0, 1.0]])
                r = self._warp_box(Mc, im.w, im.h, X0, Y0)
                if r is None:
                    continue
                src = kc.project(im, float(im.w), float(im.h), [float(v) for v in np.linalg.inv(Mc).ravel()], list(r))
                WX0, WY0, dx, dy = r[0], r[1], 0.0, 0.0
            if scov is not None:                               # squashed or skewed: the blur an ellipse (render.blur_cov)
                base, g = (kc.spread(src, float(sp)), int(math.ceil(abs(sp))) + 1) if abs(sp) >= 1e-6 else (src, 0)
                R = int(math.ceil(max(cov_reach(scov)) + max(abs(dx), abs(dy)))) + 1
                sa = self._gblur_cov(kc.padded(base, base.w + 2 * R, base.h + 2 * R, R, R), scov, (dx, dy))
                if sh['outside']:
                    kc.cut_under(sa, im, R + g + X0 - WX0, R + g + Y0 - WY0)
                kc.composite(canvas, sa, WX0 - R - g, WY0 - R - g, sop * op, clip, M, 0, 0, 0, [float(c) for c in sh['color']])
                continue
            R = int(math.ceil(3 * sig + max(abs(dx), abs(dy)))) + 1
            if M is None:                                      # blurred, cut and laid on in two or three passes
                kc.shadow(canvas, src, float(sig), float(dx), float(dy), R, WX0 - R, WY0 - R, float(sop * op), clip,
                          [float(c) for c in sh['color']], bool(sh['outside']), float(sp), im, X0 - WX0, Y0 - WY0)
                continue
            base, g = (kc.spread(src, float(sp)), int(math.ceil(abs(sp))) + 1) if abs(sp) >= 1e-6 else (src, 0)
            sa = kc.gauss(kc.padded(base, base.w + 2 * R, base.h + 2 * R, R, R), sig, sig, dx, dy)
            if sh['outside']:
                kc.cut_under(sa, im, R + g + X0 - WX0, R + g + Y0 - WY0)
            kc.composite(canvas, sa, WX0 - R - g, WY0 - R - g, sop * op, clip, M, 0, 0, 0, [float(c) for c in sh['color']])
        kc.composite(canvas, im, X0, Y0, op, clip, M, 0, 0, MODES[L.get('blend', 'normal')], None, float(k))

    @staticmethod
    def _gblur_cov(im, C, shift=(0.0, 0.0), half_out=False):
        """render.blur_cov on the GPU: two gauss_dir passes along the covariance's axes (or x and y)"""
        from .render import cov_axes
        ax = cov_axes(C)
        if ax is None:
            ax = ((math.sqrt(max(C[0, 0], 0)), (1.0, 0.0)), (math.sqrt(max(C[1, 1], 0)), (0.0, 1.0)))
        (s1, u1), (s2, u2) = ax
        a = kc.gauss_dir(im, float(s1), float(u1[0]), float(u1[1]), float(shift[0]), float(shift[1]), True)
        return kc.gauss_dir(a, float(s2), float(u2[0]), float(u2[1]), 0.0, 0.0, half_out)

    @staticmethod
    def _plain(im, k):
        """a premultiplied picture (× k) back to (rgb, a)"""
        return kc.pm(im, 1) if k == 1 else kc.pm(im, 2, k)

    @staticmethod
    def _gpu_ras(X0, Y0, im):
        return X0, Y0, im, None, None

    def fx_map(self, src, t):
        S = self._ids[src]
        p = place(self.f, S, t) if S['type'] != 'camera' else None
        c = kc.new_img(self.W, self.H, [0.0, 0, 0, 0])
        if p is not None:
            self.layer(c, S, p, t)
        return c

    def matte(self, mt, t):
        S = self._ids[mt['src']]
        p = place(self.f, S, t) if S['type'] != 'camera' else None
        inv = mt['mode'].endswith('_inverted')
        if p is None:
            return kc.new_img(self.W, self.H, [1.0, 0, 0, 0]) if inv else None
        c = kc.new_img(self.W, self.H, [0.0, 0, 0, 0])
        self.layer(c, S, p, t)
        return kc.matte(c, mt['mode'].startswith('luma'), inv)

    def draw_group(self, L, t):
        members = self._members(L, t)
        if not members:
            return None
        ext = self._extent(L, t, members)
        x0, y0, x1, y1 = (0, 0, self.W, self.H) if ext is None else ext
        if x1 <= x0 or y1 <= y0:
            return None
        view, pad = (self.ox, self.oy, self.W, self.H), self._pad
        self.ox, self.oy, self.W, self.H = self.ox + x0, self.oy + y0, x1 - x0, y1 - y0
        c = kc.new_img(self.W, self.H, [0.0, 0, 0, 0])
        try:
            for C, q in members:
                self.layer(c, C, q, t)
        finally:
            self._pad = pad
            self.ox, self.oy, self.W, self.H = view
        return x0, y0, c, PM, 1.0

    def _motion_blur(self, L, t, n, h):
        """n placements over the shutter, averaged (premultiplied). A picture placed through a matrix in every sample —
        the same prepared picture, only the matrix moving — is drawn in one pass for all samples (project_mb); other
        samples are drawn one by one into an accumulator"""
        acc, box, got = None, None, 0
        plan = []                                              # (picture, w, h, inverse, box) of samples noted, in order
        order = [0, n - 1] + list(range(1, n - 1)) if n > 2 else list(range(n))
        outer = self._sink
        try:
            for i in order:                                    # the two ends first: their union is (nearly) the whole sweep,
                ts = t - h + (i + 0.5) * 2 * h / n             # so the accumulator is not regrown and copied every sample
                q = place(self.f, L, ts)
                if q is None:
                    continue
                self._sink = Sink(plan, None, L, 4)
                self._sampling += 1                            # one moment: nothing inside is blurred again
                try:
                    r = self._draw_at(L, ts, q, tc=t if (L['type'] == 'video' or L.get('seq')) else ts)
                finally:
                    self._sampling -= 1
                self._sink = None
                got += 1
                if r is None or r is ADDED:
                    continue
                g = self.up(r)
                if g is None:
                    continue
                X0, Y0, im, k = g
                acc, box = self._acc_add(acc, box, X0, Y0, im.w, im.h)
                if k not in (0.0, 1.0):
                    im, k = self._plain(im, k), 0.0
                kc.accumulate(acc, im, X0 - box[0], Y0 - box[1], k == 1.0)
        finally:
            self._sink = outer
        if not got or (acc is None and not plan):
            return None
        mips = [p for p in plan if p[0] == 'mip']
        if mips and acc is None and len(mips) == len(plan) and all(p[1] is mips[0][1] and p[4:] == mips[0][4:] for p in mips):
            bx = [min(p[3][0] for p in mips), min(p[3][1] for p in mips), max(p[3][2] for p in mips), max(p[3][3] for p in mips)]
            _, pyr, _, _, tm, wrap = mips[0]
            # the samples not drawn (off screen) count as empty: the mean is over every sample taken
            return bx[0], bx[1], kc.mip(*pyr, [p[2] for p in mips], bx, list(tm), wrap, k=1.0 / got), None, None
        if plan and acc is None and not mips and all(p[0] is plan[0][0] and p[1:3] == plan[0][1:3] and p[5] == plan[0][5] for p in plan):
            pic, w, h = plan[0][:3]
            bx = [min(p[4][0] for p in plan), min(p[4][1] for p in plan), max(p[4][2] for p in plan), max(p[4][3] for p in plan)]
            return bx[0], bx[1], kc.project_mb(pic, w, h, [p[3] for p in plan], bx, 1.0 / got, self._plain_rect(L), plan[0][5]), None, None
        groups = {}                                            # mip samples of one pyramid: summed in one pass each
        for e in plan:
            if e[0] == 'mip':
                groups.setdefault((id(e[1][0]), e[1][1:], e[4], e[5]), []).append(e)
        for g in groups.values():
            bx = [min(e[3][0] for e in g), min(e[3][1] for e in g), max(e[3][2] for e in g), max(e[3][3] for e in g)]
            acc, box = self._acc_add(acc, box, bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1])
            kc.mip(*g[0][1], [e[2] for e in g], bx, list(g[0][4]), g[0][5], acc, bx[0] - box[0], bx[1] - box[1], 1, k=1.0)
        for e in plan:                                         # mixed or changing pictures: one by one after all
            if e[0] == 'mip':
                continue
            pic, w, h, inv, (X0, Y0, X1, Y1), crop = e
            acc, box = self._acc_add(acc, box, X0, Y0, X1 - X0, Y1 - Y0)
            kc.project(pic, w, h, inv, [X0, Y0, X1, Y1], acc, X0 - box[0], Y0 - box[1], 1, crop=crop)
        return box[0], box[1], acc, PM, 1.0 / got

    @staticmethod
    def _acc_add(acc, box, X0, Y0, w, h):
        """the motion-blur accumulator grown to hold a w × h sample at (X0, Y0)"""
        if acc is None:
            return kc.new_img(w, h, [0.0, 0, 0, 0]), [X0, Y0, X0 + w, Y0 + h]
        nb = [min(box[0], X0), min(box[1], Y0), max(box[2], X0 + w), max(box[3], Y0 + h)]
        if nb != box:
            acc = kc.padded(acc, nb[2] - nb[0], nb[3] - nb[1], box[0] - nb[0], box[1] - nb[1])
        return acc, nb

    def _skia_out(self, px, X0, Y0, ss):
        """Skia's pixels go up as 8 bits; the supersample mean and the split into colour and alpha run on the GPU"""
        if px.size == 0:
            return None
        return X0, Y0, kc.from_skia(np.ascontiguousarray(px), ss), None, None

    # ---------- turned, mirrored, skewed and 3D pictures: Lanczos to the shown size, then the matrix ----------
    def _plain8(self, im, keep):
        """a PIL picture (RGB / RGBA / RGBa) → premultiplied RGBA8 on the GPU; a still one is kept"""
        hit = self._src8.get(('plain', id(im))) if keep else None
        if hit is not None and hit[1] is im:
            return hit[0]
        if im.mode == 'RGB':
            a = np.asarray(im)
            a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
        else:
            a = np.asarray(im if im.mode == 'RGBa' else im.convert('RGBa'))
        g = kc.upload_rgba8(np.ascontiguousarray(a))
        if keep:
            if len(self._src8) > 24:
                self._src8.pop(next(iter(self._src8)))
            self._src8[('plain', id(im))] = (g, im)
        return g

    def _placed(self, pic, w, h, M, r, sink=None, crop=None, mip=False):
        """the prepared picture `pic` (all of the content w × h, or its part `crop` in content px) through matrix M"""
        X0, Y0, X1, Y1 = r
        if abs(np.linalg.det(M)) < 1e-12:                      # scaled to nothing (a pop's first frame): nothing to draw
            return None
        if mip or self._deep(M, w, h):                         # into the distance: the mip sampler (see render.py _deep)
            from . import mip
            g, w0, h0, n = self._pyramid_of(pic)
            c = crop or (0.0, 0.0, w, h)
            return self._mip_placed(sink, (g, w0, h0, n, float(w), float(h)), M, r,
                                    (w0 / (c[2] - c[0]), h0 / (c[3] - c[1]), c[0], c[1]), mip.CLAMP)
        inv = [float(v) for v in np.linalg.inv(M).ravel()]
        if sink is not None and sink[3] == 4:                  # a motion-blur sample: noted, drawn with the others at once
            sink[0].append((pic, float(w), float(h), inv, r, crop))
            return ADDED
        put = sink.args(X0, Y0, X1, Y1) if sink is not None else None
        if put is not None:                                    # straight into the accumulator / onto the canvas
            kc.project(pic, float(w), float(h), inv, [X0, Y0, X1, Y1], crop=crop, **put)
            return ADDED
        return X0, Y0, kc.project(pic, float(w), float(h), inv, [X0, Y0, X1, Y1], crop=crop), None, None

    def _prescaled(self, key, still, ident, need, cap, w, h, M, get_src, isize):
        """the source Lanczos-resized to the size it shows at (`need` px wide; a still picture 15 % larger, kept while it
        fits; `cap`: not past the source's own width unless it shows larger) — or, zoomed far in, only the part of
        that picture the view can see, with a margin: the same pixels, as Pillow's resize with a box gives exactly that
        part of the whole. Kept per layer: (picture, width, identity, crop in its pixels, height)"""
        hit = self._tcache.get(key) if still else None
        if hit is not None and hit[2] == ident and 0.6 * hit[1] <= need <= hit[1] and _inside(self._visible(M, w, h, hit[1], hit[4]), hit[3], hit[1], hit[4]):
            pic, dw, crop_px, dh = hit[0], hit[1], hit[3], hit[4]
        else:
            dw = max(1, int(math.ceil(need * (1.15 if still else 1.0))))
            if cap:
                dw = min(dw, max(isize[0], int(math.ceil(need))))
            dh = max(1, int(round(dw * h / w)))
            crop_px = self._crop(self._visible(M, w, h, dw, dh), dw, dh)
            x0, y0, x1, y1 = crop_px
            fx, fy = isize[0] / dw, isize[1] / dh
            full = crop_px == (0, 0, dw, dh)
            pic = kc.lanczos(get_src(), x1 - x0, y1 - y0, [0.0, 0.0, float(isize[0]), float(isize[1])] if full
                             else [x0 * fx, y0 * fy, x1 * fx, y1 * fy], [-x0, -y0, dw - x0, dh - y0])
            if still:
                self._tcache[key] = (pic, dw, ident, crop_px, dh)
        x0, y0, x1, y1 = crop_px
        crop = None if crop_px == (0, 0, dw, dh) else [x0 * w / dw, y0 * h / dh, w if x1 == dw else x1 * w / dw, h if y1 == dh else y1 * h / dh]
        return pic, crop

    def _visible(self, M, w, h, dw, dh):
        """the pixels of a dw × dh picture of the content (w × h) the view can show through M, with the taps of the cubic
        around them; None: all of it (a perspective that folds the view behind the eye)"""
        e = 4 + self._pad
        try:
            inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return None
        pts = []
        for x, y in ((-e, -e), (self.W + e, -e), (-e, self.H + e), (self.W + e, self.H + e)):
            v = inv @ np.array([x, y, 1.0])
            if v[2] <= 1e-9:
                return None
            pts.append((v[0] / v[2] * dw / w, v[1] / v[2] * dh / h))
        return (math.floor(min(p[0] for p in pts)) - 3, math.floor(min(p[1] for p in pts)) - 3,
                math.ceil(max(p[0] for p in pts)) + 3, math.ceil(max(p[1] for p in pts)) + 3)

    @staticmethod
    def _crop(need, dw, dh):
        """the part to make: what is needed and a quarter of it more on each side (a moving zoom reuses it), clamped;
        all of the picture when that is most of it anyway"""
        if need is None:
            return (0, 0, dw, dh)
        mx, my = (need[2] - need[0]) // 4, (need[3] - need[1]) // 4
        c = (max(0, need[0] - mx), max(0, need[1] - my), min(dw, need[2] + mx), min(dh, need[3] + my))
        if c[2] <= c[0] or c[3] <= c[1] or (c[2] - c[0]) * (c[3] - c[1]) > 0.6 * dw * dh:
            return (0, 0, dw, dh)
        return c

    def _sink_for(self, L):
        """the sink this layer may draw straight into (only the layer itself — not a member of a group being drawn,
        which goes into the group's own picture)"""
        sk = self._sink
        return sk if sk is not None and sk[2] is L else None

    def _plain_src(self, path, crop, grade):
        """a picture file (cropped, graded, premultiplied) on the GPU, kept by its name — the CPU's decoded-picture cache
        holds only a few, and re-decoding a PNG every frame was a good part of what was left"""
        key = ('psrc', path, _mtime(path), crop, grade)        # a file rewritten under the same name is read again
        hit = self._src8.get(key)
        if hit is None:
            if len(self._src8) > 48:
                self._src8.pop(next(iter(self._src8)))
            hit = self._src8[key] = self._prepared(path, crop, grade, False, 0) or self._cpu_plain(path, crop, grade)
        return hit

    @staticmethod
    def _cpu_plain(path, crop, grade):
        im = load_plain(path, crop, grade)
        return kc.upload_rgba8(np.ascontiguousarray(np.asarray(im))), im.size

    def _prepared(self, path, crop, grade, flip, pad):
        """the file decoded on the CPU (that much stays there), everything after it — grade, mirror, edge padding,
        premultiplying — on the GPU (kc.prep8, byte for byte render.load_image / load_plain); None: a grade the GPU
        does not do (colour balance), the CPU path then"""
        if grade and (len(grade) > 6 or any(k not in GRADE_OPS for k, _ in grade)):
            return None
        fu = self._ahead.pop((path, crop), None)
        a, size = fu.result() if fu is not None else decode(path, crop)
        ops = [(GRADE_OPS[k], float(v)) for k, v in grade] if grade else None
        return kc.prep8(a, ops, bool(flip), pad), size

    def _look_ahead(self, L, t, crop):
        """an image sequence: the next two frames' files are decoded in the background while this frame is drawn
        (PNG decoding lets go of Python's lock, so it really runs alongside)"""
        global _DECODERS
        if L.get('seq') is None:
            return
        if _DECODERS is None:
            _DECODERS = ThreadPoolExecutor(2)
        now = img_src(L, t, self.f.fps)
        for k in (1, 2):
            p = img_src(L, t + k / self.f.fps, self.f.fps)
            if p != now and (p, crop) not in self._ahead and not any(p in key[:3] for key in self._src8):
                self._ahead[(p, crop)] = _DECODERS.submit(decode, p, crop)
        while len(self._ahead) > 8:                            # a jump (a preview at another time): forget the oldest
            self._ahead.pop(next(iter(self._ahead))).cancel()

    def _as_numpy(self, r):
        """a raster this renderer made (GPU picture) as numpy (X0, Y0, rgb, a) — for text drawn flat (render.py _flat_text)"""
        g = self.up(r)
        if g is None:
            return None
        X0, Y0, im, k = g
        if k:
            im = self._plain(im, k)
        a = im.download()
        return X0, Y0, a[..., :3], a[..., 3:]

    def _project(self, L, im, w, h, M, still):
        return self._project_src(L, lambda: self._plain8(im, still), im.size, id(im), w, h, M, still)

    def _project_src(self, L, get_src, size, ident, w, h, M, still):
        r = self._quad_bounds(M, (0, 0, w, h))
        if r is None or abs(np.linalg.det(M)) < 1e-12:
            return None
        need = max(1.0, w * self._mscale(M, (0, 0, w, h)))
        pic, crop = self._prescaled(('gmat', L['id']), still, ident, need, True, w, h, M, get_src, size)
        return self._placed(pic, w, h, M, r, self._sink_for(L), crop)

    # ---------- a plain filled rectangle (most masks and plates): the picture path with one colour, no Skia ----------
    @staticmethod
    def _plain_rect(L):
        return (L['type'] == 'shape' and L['shape'] == 'rect' and not L['radius'].animated and not L['radius'].value and not L['stroke'] and not L.get('arrow')
                and L['fill'] is not None and not isinstance(L['fill'], dict)
                and not L.get('paths') and L.get('offset') is None and not L.get('repeat'))

    def _solid_rect(self, L, t, M):
        """the rectangle 0..w × 0..h through M with exact edge coverage (as a turned picture's edges), filled with its
        colour as Skia stores it: 8-bit alpha, colour premultiplied to 8 bits"""
        w, h = L['size'].at(t)
        r = self._quad_bounds(M, (0, 0, w, h))
        if r is None or w <= 0 or h <= 0:
            return None
        spec = L['fill']
        rgb = [int(round(v)) for v in spec[:3]]
        a8 = int(round(255 * (spec[3] if len(spec) > 3 else 1)))
        key = ('rect', tuple(rgb), a8)
        pic = self._src8.get(key)
        if pic is None:
            if not a8:
                return None
            pm = [((c * a8 + 128) + ((c * a8 + 128) >> 8)) >> 8 for c in rgb]   # Skia's premultiply to 8 bits
            pic = self._src8[key] = kc.new_img(8, 8, [float(v) * 255 / a8 for v in pm] + [a8 / 255])
        return self._placed(pic, w, h, M, r, self._sink_for(L))

    def draw_shape(self, L, t, cx, cy, s, rot, ss=2):
        ax, ay = L['anchor']
        th = math.radians(rot)
        c, si = math.cos(th), math.sin(th)
        M = np.array([[s * c, -s * si, cx], [s * si, s * c, cy], [0, 0, 1.0]]) @ np.array([[1, 0, -ax], [0, 1, -ay], [0, 0, 1.0]])
        if self._plain_rect(L):
            return self._solid_rect(L, t, M)
        if not self._sampling:                                 # one moment: Skia straight in frame px — exact edges (a picture
            return super().draw_shape(L, t, cx, cy, s, rot, ss)   # resampled rings 1 px at a hard edge, up to 72 levels)
        got = self._shape_pic(L, t, M, ss)
        return super().draw_shape(L, t, cx, cy, s, rot, ss) if got is _REDRAW else got

    def _shape_pic(self, L, t, M, ss):
        """for the motion-blur samples of a moving shape: drawn by Skia once in its own coordinates — the part the view can see and a quarter more, at the
        size it shows at and 15 % more, supersampled ss× — then moved, turned and motion-blurred on the GPU like a
        picture. Drawn again only when its outline or paint changes, its scale leaves 0.6…1× of the drawing's, or the
        view leaves the drawn part. v1 drew it anew for every motion-blur sample on a canvas as large as its box on
        screen: a reel's two sweeping polygons were 26 full-frame Skia drawings (600 MB up) a frame"""
        import skia
        from .render import shape_path
        path = shape_path(L, t)
        st = L['stroke']
        trim = L['trim'].at(t)
        pe = None
        if st and (trim[0] > 0 or trim[1] < 1):
            if trim[1] <= trim[0]:
                return None
            pe = skia.TrimPathEffect.Make(float(trim[0]), float(trim[1]))
        b = path.computeTightBounds()
        grow = (st['width'] * 3 + 4) if st else 2
        box = (b.left() - grow, b.top() - grow, b.right() + grow, b.bottom() + grow)
        if box[2] <= box[0] or box[3] <= box[1] or abs(np.linalg.det(M)) < 1e-12:
            return None
        sc = self._mscale(M, box)
        vis = self._local_view(M)                              # the content px the frame can see (None: all of it)
        need = box if vis is None else (max(box[0], vis[0]), max(box[1], vis[1]), min(box[2], vis[2]), min(box[3], vis[3]))
        if need[2] <= need[0] or need[3] <= need[1]:
            return None
        from . import paths
        ident = repr((paths.stamp(L, t),
                      trim, repr(L['fill']), L.get('arrow'), ss,
                      st and sorted((k, v.at(t) if isinstance(v, Prop) else v) for k, v in st.items())))
        key = ('gshape', L['id'])
        hit = self._tcache.get(key)
        if not (hit is not None and hit[1] == ident and 0.6 * hit[2] <= sc <= hit[2]
                and hit[3][0] <= need[0] and hit[3][1] <= need[1] and hit[3][2] >= need[2] and hit[3][3] >= need[3]):
            gx, gy = (need[2] - need[0]) / 4, (need[3] - need[1]) / 4
            reg = (max(box[0], need[0] - gx), max(box[1], need[1] - gy), min(box[2], need[2] + gx), min(box[3], need[3] + gy))
            dsc = sc * 1.15
            dw, dh = max(1, int(math.ceil((reg[2] - reg[0]) * dsc))), max(1, int(math.ceil((reg[3] - reg[1]) * dsc)))
            if dw * dh > 12e6:                                 # enormous even in part: Skia straight in frame px, as before
                return _REDRAW
            surf = skia.Surface(dw * ss, dh * ss)
            cv = surf.getCanvas()
            cv.clear(skia.Color4f(0, 0, 0, 0))
            cv.scale(ss * dw / (reg[2] - reg[0]), ss * dh / (reg[3] - reg[1]))
            cv.translate(-reg[0], -reg[1])
            self._shape_paint(cv, L, t, path, pe, trim, b)
            pic = kc.from_skia(np.ascontiguousarray(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType,
                                                                                     alphaType=skia.kPremul_AlphaType)), ss)
            hit = self._tcache[key] = (pic, ident, dsc, reg)
        pic, reg = hit[0], hit[3]
        Mg = M @ np.array([[1, 0, reg[0]], [0, 1, reg[1]], [0, 0, 1.0]])
        r = self._quad_bounds(Mg, (0, 0, reg[2] - reg[0], reg[3] - reg[1]))
        if r is None:
            return None
        return self._placed(pic, reg[2] - reg[0], reg[3] - reg[1], Mg, r, self._sink_for(L))

    def _local_view(self, M):
        """the frame (with the layer's padding and a few px) in the content coordinates of matrix M; None when a frame
        corner lies behind the eye"""
        try:
            inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return None
        e = 8 + self._pad
        pts = np.array([[-e, -e, 1], [self.W + e, -e, 1], [-e, self.H + e, 1], [self.W + e, self.H + e, 1]], float) @ inv.T
        if (pts[:, 2] <= 1e-9).any():
            return None
        x, y = pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    # ---------- a pure plugin drawing (plugins.layer_type(pure=True)): drawn once in its own box, placed as a picture ----
    @staticmethod
    def _pure(L):
        return 'plugin' in L and L['plugin'][1] == 'skia' and getattr(L['plugin'][0], 'kadr_pure', False)

    def draw_plugin(self, L, t, cx, cy, s, rot, ss=2):
        if not self._pure(L):
            return super().draw_plugin(L, t, cx, cy, s, rot, ss)
        w, h = L['size'].at(t)
        th = math.radians(rot)
        c, si = math.cos(th), math.sin(th)
        M = np.array([[s * c, -s * si, cx], [s * si, s * c, cy], [0, 0, 1.0]]) @ np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
        return self._pure_plugin(L, t, M)

    def _pure_plugin(self, L, t, M, ss=1):
        """the plugin's box (0..w × 0..h, plus its bleed) drawn by Skia at about the size it shows at (15 % more, kept
        while it fits), then through M like a picture — the drawing again only when a parameter or the size changes.
        Not supersampled: drawn finer than it shows and then resampled, 2× supersampling changed nothing the eye sees
        (the Barcode grooves: same picture, ≤ 0.16 mean levels) and cost three times the time of a drawing redone
        every frame"""
        import skia
        fn, _ = L['plugin']
        pv = {n: (v.at(t) if isinstance(v, Prop) else v) for n, v in L['params'].items()}
        w, h = L['size'].at(t)
        if w <= 0 or h <= 0 or abs(np.linalg.det(M)) < 1e-12:
            return None
        g = float(pv.get('_bleed', 0)) + 2
        need = max(1.0, (w + 2 * g) * self._mscale(M, (-g, -g, w + g, h + g)))
        if max(w, h) > 2 * min(w, h) and (self._deep(M, w, h) or need * need * (h + 2 * g) / (w + 2 * g) > 4e6):
            return self._pure_bands(L, t, M, fn, pv, w, h, g, ss)
        key, ident = ('gplug', L['id']), repr((sorted(pv.items()), w, h))
        hit = self._tcache.get(key)
        if hit is not None and hit[2] == ident and 0.6 * hit[1] <= need <= hit[1]:
            pic = hit[0]
        else:
            dw = max(1, int(math.ceil(need * 1.15)))
            sc = dw / (w + 2 * g)
            dh = max(1, int(round((h + 2 * g) * sc)))
            surf = skia.Surface(dw * ss, dh * ss)
            cv = surf.getCanvas()
            cv.clear(skia.Color4f(0, 0, 0, 0))
            cv.scale(ss * sc, ss * sc)
            cv.translate(g, g)
            fn(cv, pv, w, h, Fx(sc, t, t - L['in'], self.k))
            pic = kc.from_skia(np.ascontiguousarray(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType,
                                                                                     alphaType=skia.kPremul_AlphaType)), ss)
            self._tcache[key] = (pic, dw, ident)
        Mg = M @ np.array([[1, 0, -g], [0, 1, -g], [0, 0, 1.0]])
        r = self._quad_bounds(Mg, (0, 0, w + 2 * g, h + 2 * g))
        if r is None:
            return None
        return self._placed(pic, w + 2 * g, h + 2 * g, Mg, r, self._sink_for(L))

    def _pure_bands(self, L, t, M, fn, pv, w, h, g, ss):
        """a long pure drawing running into the distance (grooves to the horizon): one picture at the near end's size
        would be enormous, so its box is cut across its length into up to 16 bands, each drawn by Skia (clipped to the
        band, the drawing's filters intact across the cut) at the size that band shows at — the far bands are a few
        pixels — and kept while that size holds. Each band is placed with the exact coverage of its own strip, and the
        strips are added premultiplied, so where two meet their coverages sum to one: no seam"""
        import skia
        along_y = h >= w
        long_, short = (h, w) if along_y else (w, h)
        n = int(min(16, math.ceil(long_ / max(short, long_ / 16))))
        cuts = [-g] + [long_ * i / n for i in range(1, n)] + [long_ + g]
        ident = repr((sorted(pv.items()), w, h))
        parts, box = [], None
        for i in range(n):
            a, b = cuts[i], cuts[i + 1]
            band = (-g, a, w + g, b) if along_y else (a, -g, b, h + g)
            Mb = M @ np.array([[1, 0, band[0]], [0, 1, band[1]], [0, 0, 1.0]])
            bw, bh = band[2] - band[0], band[3] - band[1]
            r = self._quad_bounds(Mb, (0, 0, bw, bh))
            if r is None:
                continue
            need = max(1.0, max(bw, bh) * self._mscale(Mb, (0, 0, bw, bh)))
            key = ('gband', L['id'], i)
            hit = self._tcache.get(key)
            m = 4                                              # content px of margin: the cubic's taps across the cut
            vis = self._visible(Mb, bw, bh, bw, bh)            # the part of the band the view can see (content px)
            vis = (-m, -m, bw + m, bh + m) if vis is None else (max(-m, vis[0]), max(-m, vis[1]), min(bw + m, vis[2]), min(bh + m, vis[3]))
            if vis[0] >= vis[2] or vis[1] >= vis[3]:
                continue
            # kept from need = its size down to a quarter of it (the mip sampler shrinks it cleanly), drawn 1.25× ahead
            # and without supersampling (already finer than it shows),
            # while what is seen lies inside what was drawn
            if (hit is not None and hit[2] == ident and 0.25 * hit[1] <= need <= hit[1] and hit[3][0] <= vis[0] and
                    hit[3][1] <= vis[1] and vis[2] <= hit[3][2] and vis[3] <= hit[3][3]):
                pic, crop = hit[0], hit[3]
            else:
                dl = max(2, int(math.ceil(need * 1.25)))
                sc = dl / max(bw, bh)
                mx, my = (vis[2] - vis[0]) / 8, (vis[3] - vis[1]) / 8    # an eighth more each way: a moving view reuses it
                crop = (max(-m, vis[0] - mx), max(-m, vis[1] - my), min(bw + m, vis[2] + mx), min(bh + m, vis[3] + my))
                cw_, ch_ = crop[2] - crop[0], crop[3] - crop[1]
                dw, dh = max(1, int(math.ceil(cw_ * sc))), max(1, int(math.ceil(ch_ * sc)))
                surf = skia.Surface(dw, dh)
                cv = surf.getCanvas()
                cv.clear(skia.Color4f(0, 0, 0, 0))
                cv.scale(dw / cw_, dh / ch_)
                cv.translate(-crop[0] - band[0], -crop[1] - band[1])
                cv.clipRect(skia.Rect.MakeLTRB(band[0] + crop[0], band[1] + crop[1], band[0] + crop[2], band[1] + crop[3]))
                _t0 = time.time()
                fn(cv, pv, w, h, Fx(sc, t, t - L['in'], self.k))
                pic = kc.from_skia(np.ascontiguousarray(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType,
                                                                                         alphaType=skia.kPremul_AlphaType)), 1)
                if os.environ.get('KADR_BANDS'):
                    print(f'  band {i} t={t:.2f} need {need:.0f} {dw}x{dh} {time.time() - _t0:.2f}s', 'hit' if hit else 'new',
                          hit and (hit[2] == ident, round(need / hit[1], 2)))
                self._tcache[key] = (pic, dl, ident, crop)
            parts.append((pic, bw, bh, Mb, r, crop))
            box = list(r) if box is None else [min(box[0], r[0]), min(box[1], r[1]), max(box[2], r[2]), max(box[3], r[3])]
        if not parts:
            return None
        sink = self._sink_for(L)
        if sink is not None and sink[3] == 4:                  # a motion-blur sample: every band noted (added one by one)
            for pic, bw, bh, Mb, r, crop in parts:
                self._placed(pic, bw, bh, Mb, r, sink, crop, mip=True)
            return ADDED
        acc = kc.new_img(box[2] - box[0], box[3] - box[1], [0.0, 0, 0, 0])
        add = Sink(acc, box, L, 1)
        for pic, bw, bh, Mb, r, crop in parts:
            self._placed(pic, bw, bh, Mb, r, add, crop, mip=True)
        return box[0], box[1], acc, PM, 1.0

    # ---------- a nested comp: rendered on the GPU by a renderer of its own, then placed like a picture ----------
    def _comp_src(self, L, t, pad=0):
        """the comp's frame as a packed source picture; pad: with that many empty px around it (what render.py's
        draw_comp hands to _raster — the Lanczos at the picture's edge blends with nothing, not with a held edge)"""
        r = self._comps.get(L['id'])
        if r is None:
            for cid in [c for c in self._comps if not (self._ids[c]['in'] <= t < self._ids[c]['out'])]:
                del self._comps[cid]                            # a comp off screen drops its renderer and its caches
                for k_ in [k_ for k_ in self._comp_still if k_[0] == cid]:
                    del self._comp_still[k_]
            r = self._comps[L['id']] = GpuRenderer(L['film'], self.k)
        tc = comp_time(L, t)
        # a comp whose every layer stands still (a calendar sheet flying past as a whole) looks the same at every moment
        # its set of layers is on: made once, not again for each motion-blur sample and frame
        fl = L['film']
        on = tuple(C['id'] for C in fl.layers if C['in'] <= tc < C['out'])
        still = fl.camera is None and all(r.static(C, tc) for C in fl.layers if C['in'] <= tc < C['out'])
        hit = self._comp_still.get((L['id'], pad))
        if still and hit is not None and hit[0] == on:
            return hit[1]
        canvas = r.frame_comp(tc)
        if pad:
            canvas = kc.padded(canvas, canvas.w + 2 * pad, canvas.h + 2 * pad, pad, pad)
        src = kc.canvas_to8(canvas)
        if still:
            self._comp_still[(L['id'], pad)] = (on, src)
        else:
            self._comp_still.pop((L['id'], pad), None)
        return src

    def draw_comp(self, L, t, cx, cy, s, rot):
        if rot:
            src = self._comp_src(L, t)
            return self._rotated_src(L, t, cx, cy, s, rot, lambda: src, (src.w, src.h), ('comp', id(src)), False)
        # not turned: Lanczos straight from the comp's frame, as render.py draw_comp → _raster (at 1:1 the comp's own
        # pixels; the turned picture's cubic softened thin lines by up to 19 levels there)
        src = self._comp_src(L, t, PAD)
        return self._lanczos_box(L, src, src.w - 2 * PAD, src.h - 2 * PAD, *self._box(L, t, cx, cy, s / self.k))

    def draw_matrix(self, L, t, M, cx, cy):
        if L['type'] == 'image' and L.get('tile'):
            return self._tile(L, t, M)
        if self._plain_rect(L):
            return self._solid_rect(L, t, M)
        if self._pure(L):
            return self._pure_plugin(L, t, M)
        if L['type'] == 'comp':
            w, h = content_size(self.f, L, t)
            if w <= 0 or h <= 0:
                return None
            src = self._comp_src(L, t)
            return self._project_src(L, lambda: src, (src.w, src.h), ('comp', id(src)), w, h, M, False)
        if L['type'] != 'image':
            return super().draw_matrix(L, t, M, cx, cy)
        w, h = content_size(self.f, L, t)
        if w <= 0 or h <= 0:
            return None
        if L.get('flip'):
            M = M @ np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1.0]])
        key = (img_src(L, t, self.f.fps), L['crop'], L['grade'])
        self._look_ahead(L, t, L['crop'])
        src, size = self._plain_src(*key)
        return self._project_src(L, lambda: src, size, key, w, h, M, L.get('seq') is None)

    # ---------- a tiled picture (an endless floor): its mip pyramid through the matrix, anisotropic (mip.wgsl) ----------
    def _tile(self, L, t, M):
        w, h = content_size(self.f, L, t)
        if L.get('flip'):
            M = M @ np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1.0]])
        r = self._tile_bounds(M, w, h)
        if r is None or w <= 0 or h <= 0:
            return None
        lv, pk, size0, tm, wrap = self._tile_src(L, t)
        key = ('mip', id(pk))
        hit = self._src8.get(key)
        if hit is None or hit[1] is not pk:
            hit = self._src8[key] = (kc.upload(pk, half=True), pk)
        return self._mip_placed(self._sink_for(L), (hit[0], size0[0], size0[1], len(lv), float(w), float(h)), M, r, tm, wrap)

    def _pyramid_of(self, pic):
        """a prepared picture's mip pyramid on the GPU (premultiplied, kadr/mip.py), kept while the picture is"""
        from . import mip
        key = ('pyr', id(pic))
        hit = self._src8.get(key)
        if hit is None or hit[1] is not pic:
            a = pic.download()
            lv = mip.pyramid(np.concatenate([a[..., :3] * a[..., 3:], a[..., 3:]], -1))
            hit = self._src8[key] = (kc.upload(mip.packed(lv), half=True), pic, len(lv))
        return hit[0], pic.w, pic.h, hit[2]

    def _mip_placed(self, sink, pyr, M, r, tm, wrap):
        """a pyramid (Img, w0, h0, levels, cw, ch) through M: noted for a motion-blur pass, put straight into the sink,
        or drawn as a picture of its own"""
        inv = [float(v) for v in np.linalg.inv(M).ravel()]
        if sink is not None and sink[3] == 4:
            sink[0].append(('mip', pyr, inv, r, tuple(tm), wrap))
            return ADDED
        put = sink.args(*r) if sink is not None else None
        if put is not None:
            kc.mip(*pyr, [inv], list(r), list(tm), wrap, **put)
            return ADDED
        return r[0], r[1], kc.mip(*pyr, [inv], list(r), list(tm), wrap), None, None

    def _rotated(self, L, t, cx, cy, s, rot, im, size, still=False):
        return self._rotated_src(L, t, cx, cy, s, rot, lambda: self._plain8(im, still), im.size, id(im), still)

    def _rotated_src(self, L, t, cx, cy, s, rot, get_src, isize, ident, still):
        w, h = content_size(self.f, L, t)
        ax, ay = anchor(L, (w, h), t)
        th = math.radians(rot)
        c, s_ = math.cos(th), math.sin(th)
        M = np.array([[c, -s_, cx], [s_, c, cy], [0, 0, 1.0]]) @ np.array([[s, 0, -ax * s * w], [0, s, -ay * s * h], [0, 0, 1.0]])
        r = self._quad_bounds(M, (0, 0, w, h))
        if r is None or s <= 1e-9:
            return None
        pic, crop = self._prescaled(('grot', L['id']), still, ident, s * w, False, w, h, M, get_src, isize)
        return self._placed(pic, w, h, M, r, self._sink_for(L), crop)

    # ---------- text: glyphs rasterized in Rust (same FreeType), coverage goes up as one plane, coloured on the GPU ----
    def _glyph_raster(self, L, glyphs, faces, lin, base, lx, ly, cx, cy, sm, X0, Y0, X1, Y1, colors, sel, per):
        if L.get('stroke'):                                    # outlined text: FreeType's stroker, still from Python
            return super()._glyph_raster(L, glyphs, faces, lin, base, lx, ly, cx, cy, sm, X0, Y0, X1, Y1, colors, sel, per)
        _ft_ready()
        cols, idx = [], {}
        group = [-1] * (max((ci for *_, ci in glyphs), default=-1) + 1)
        for _, _, _, ci in glyphs:
            if group[ci] < 0 and (sel is None or ci in sel):
                c = colors(ci)
                if c not in idx:
                    idx[c] = len(cols)
                    cols.append(c)
                group[ci] = idx[c]
        got = kc.glyph_cover([ctypes.cast(f._FT_Face, ctypes.c_void_p).value for f in faces], int(round(L['size'] * 64)), FT_FLAGS,
                             glyphs, group, len(cols), per or {}, list(map(float, lin)), list(map(float, base)), float(lx), float(ly),
                             float(cx), float(cy), X0, Y0, X1 - X0, Y1 - Y0, FALLBACK)
        return X0, Y0, {cols[k]: plane for k, plane in got}

    # ---------- text: coverage goes up as one plane, the colour is added on the GPU ----------
    def _text_raster(self, X0, Y0, cov, key):
        if len(cov) != 1:                                      # several colours (a highlighted word): mixed on the CPU
            return super()._text_raster(X0, Y0, cov, key)
        (col, c), = cov.items()
        plane = kc.upload_plane(np.ascontiguousarray(np.minimum(1, c), np.float32))
        return X0, Y0, kc.colorize(plane, [float(v) for v in col]), None, key

    def _combine_parts(self, L, parts, sm):
        boxes = []
        for X0, Y0, c, b, o in parts:
            R = int(math.ceil(3 * b * sm)) + 1 if b > 0.05 else 0
            boxes.append((X0 - R, Y0 - R, X0 + c.shape[1] + R, Y0 + c.shape[0] + R))
        X0, Y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        X1, Y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        canvas = kc.new_img(X1 - X0, Y1 - Y0, [0.0, 0, 0, 0])
        col = [float(v) for v in L['color']]
        for (px0, py0, c, b, o), bx in zip(parts, boxes):
            im = kc.colorize(kc.upload_plane(np.ascontiguousarray(c, np.float32)), col)
            if b > 0.05:
                R = px0 - bx[0]
                im = kc.gauss(kc.padded(im, im.w + 2 * R, im.h + 2 * R, R, R), b * sm, b * sm)
                px0, py0 = bx[0], bx[1]
            kc.composite(canvas, im, px0 - X0, py0 - Y0, float(o), None, None, 0, 0, 0, col)
        return X0, Y0, canvas, PM, 1.0

    # ---------- footage without a turn: the frame goes up as it is, Lanczos reads it as if edge-padded ----------
    def draw_video(self, L, t, cx, cy, s, rot):
        if rot:
            return super().draw_video(L, t, cx, cy, s, rot)
        fr = video_frame(L, t)
        if L.get('delogo'):
            fr = delogo(fr, L['delogo'])
        a = crop_frame(fr, L['crop'])
        if L.get('flip'):
            a = a[:, ::-1]
        a = np.ascontiguousarray(a)                            # graded on the GPU as it is read (render.apply_grade)
        ih, iw = a.shape[:2]
        X0, Y0, w, h = self._box(L, t, cx, cy, s)
        r = self._region(X0, Y0, X0 + w, Y0 + h)
        if not r:
            return None
        ix0, iy0, ix1, iy1 = r
        fx, fy = iw / w, ih / h
        box = [PAD + (ix0 - X0) * fx, PAD + (iy0 - Y0) * fy, PAD + (ix1 - X0) * fx, PAD + (iy1 - Y0) * fy]
        box = [max(0.0, box[0]), max(0.0, box[1]), min(iw + 2.0 * PAD, box[2]), min(ih + 2.0 * PAD, box[3])]
        src = kc.upload_source(a, PAD)
        gr = [(GRADE_OPS[k_], float(v)) for k_, v in (L['grade'] or ()) if k_ in GRADE_OPS] if a.shape[2] == 3 else None
        if L['grade'] and any(k_ not in GRADE_OPS for k_, _ in L['grade']):   # colour balance: the CPU grade, as before
            src, gr = kc.upload_source(np.ascontiguousarray(apply_grade(a, L['grade'])), PAD), None
        cover = [X0 - ix0, Y0 - iy0, X0 + w - ix0, Y0 + h - iy0]
        sk = self._sink_for(L)
        put = sk.args(ix0, iy0, ix1, iy1) if sk is not None else None
        if put is not None:
            kc.lanczos(src, ix1 - ix0, iy1 - iy0, box, cover, gr or None, **put)
            return ADDED
        return ix0, iy0, kc.lanczos(src, ix1 - ix0, iy1 - iy0, box, cover, gr or None), None, None

    # ---------- a gradient in its own box ----------
    def draw_gradient(self, L, t, cx, cy, s, rot):
        if L.get('box') is None:
            return super().draw_gradient(L, t, cx, cy, s, rot)   # full-frame: computed once and kept
        bw, bh = L['box'].at(t)
        X0, Y0, w, h = cx - s * bw / 2, cy - s * bh / 2, s * bw, s * bh
        r = self._region(X0, Y0, X0 + w, Y0 + h)
        if not r or w <= 0 or h <= 0:
            return None
        if L['kind'] == 'linear':
            an = math.radians(L['angle'])
            dx, dy = math.sin(an), -math.cos(an)
            geo = [dx, dy, 0.5 * (abs(dx) * w + abs(dy) * h), 0, 0, 1, 1]
        else:
            (gx, gy), (rx, ry) = L['center'].at(t), L['radius']
            geo = [0, 0, 1, gx, gy, rx, ry]
        stops = [[float(p_), *map(float, c_), float(a_)] for p_, c_, a_ in L['stops']]
        im = kc.gradient(r[2] - r[0], r[3] - r[1], [X0 - r[0], Y0 - r[1], w, h], 0 if L['kind'] == 'linear' else 1, [float(v) for v in geo], stops,
                         bool(L.get('premul', True)) and len({st[4] for st in stops}) > 1)
        return r[0], r[1], im, None, None

    # ---------- pictures without a turn: Pillow's Lanczos on the GPU ----------
    def _source(self, path, crop, grade, flip):
        key = (path, _mtime(path), crop, grade, flip)
        if key not in self._src8:
            if len(self._src8) > 64:                           # GPU memory: keep the pictures of the moment (a reel has dozens)
                self._src8.pop(next(iter(self._src8)))
            got = self._prepared(path, crop, grade, flip, PAD)
            if got is None:
                im, size = load_image(path, crop, grade, flip)
                got = kc.upload_rgba8(np.ascontiguousarray(np.asarray(im))), size
            self._src8[key] = got
        return self._src8[key]

    def draw_image(self, L, t, cx, cy, s, rot):
        self._look_ahead(L, t, L['crop'])
        if rot or (self._sampling and not L.get('flip')):      # a motion-blur sample: as render.py draw_image
            key = (img_src(L, t, self.f.fps), L['crop'], L['grade'])
            src, size = self._plain_src(*key)
            return self._rotated_src(L, t, cx, cy, s, rot, lambda: src, size, key, L.get('seq') is None)
        src, (iw, ih) = self._source(img_src(L, t, self.f.fps), L['crop'], L['grade'], L.get('flip', False))
        return self._lanczos_box(L, src, iw, ih, *self._box(L, t, cx, cy, s))

    def _lanczos_box(self, L, src, iw, ih, X0, Y0, w, h):
        """a source picture padded by PAD (its own iw × ih inside) into the screen box X0, Y0, w × h — render._raster"""
        r = self._region(X0, Y0, X0 + w, Y0 + h)
        if not r:
            return None
        ix0, iy0, ix1, iy1 = r
        fx, fy = iw / w, ih / h
        box = [PAD + (ix0 - X0) * fx, PAD + (iy0 - Y0) * fy, PAD + (ix1 - X0) * fx, PAD + (iy1 - Y0) * fy]
        box = [max(0.0, box[0]), max(0.0, box[1]), min(src.w, box[2]), min(src.h, box[3])]
        cover = [X0 - ix0, Y0 - iy0, X0 + w - ix0, Y0 + h - iy0]
        sk = self._sink_for(L)
        put = sk.args(ix0, iy0, ix1, iy1) if sk is not None else None
        if put is not None:
            kc.lanczos(src, ix1 - ix0, iy1 - iy0, box, cover, None, **put)
            return ADDED
        return ix0, iy0, kc.lanczos(src, ix1 - ix0, iy1 - iy0, box, cover), None, None
