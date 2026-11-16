"""Each GPU kernel against the numpy/Pillow code it ports (kadr/render.py). Run: python core/tests/test_kernels.py"""
import os, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import kadr_core as kc
import kadr.gpu as G
from kadr import render as R

rng = np.random.default_rng(7)
ok = True


def check(name, got, want, tol):
    global ok
    d = np.abs(np.asarray(got, np.float64) - np.asarray(want, np.float64))
    good = d.max() <= tol
    ok &= good
    print(f'{"OK  " if good else "FAIL"} {name}: max {d.max():.4g}, mean {d.mean():.3g} (допуск {tol})')


def layer(h, w):
    rgb = rng.random((h, w, 3), np.float32) * 255
    a = rng.random((h, w, 1), np.float32)
    a[a < 0.3] = 0
    return rgb, a


# ---- composite: every blend mode, clip, matte, a colour (shadow)
for mode, name in enumerate(['normal', 'multiply', 'screen', 'overlay', 'soft-light', 'add']):
    H, W = 70, 90
    base = rng.random((H, W, 3), np.float32) * 255
    rgb, a = layer(40, 50)
    X0, Y0, op, clip = 30, -5, 0.8, (31.3, 2.6, 70.2, 30.4)
    matte = rng.random((H, W), np.float32)
    r = R.Renderer.__new__(R.Renderer)
    r.W, r.H, r.alpha, r._matte, r.mode = W, H, np.zeros((H, W, 1), np.float32), matte, name
    img = base.copy()
    r._put(img, X0, Y0, a * op, rgb, clip)
    dst = kc.upload(np.ascontiguousarray(np.concatenate([base, np.zeros((H, W, 1), np.float32)], -1)))
    kc.composite(dst, kc.upload(np.ascontiguousarray(np.concatenate([rgb, a], -1))), X0, Y0, op, list(clip), kc.upload_plane(matte), 0, 0,
                 mode)
    got = dst.download()
    check(f'composite {name}', got[..., :3], img, 2e-3)
    check(f'composite {name} alpha', got[..., 3:], r.alpha, 1e-5)

# ---- gauss with a fractional offset, and a directional one
x = rng.random((60, 80, 4), np.float32)
for sig, dx, dy in ((2.3, 0.4, -1.7), (5.5, 3.0, 2.0)):
    want = R._gauss_direct(x, sig, dx, dy)
    got = kc.gauss(kc.upload(x), sig, sig, dx, dy).download()
    check(f'gauss σ={sig} shift ({dx}, {dy})', got, want, 3e-4)   # the first pass is kept in f16 (2^-11 relative)
for sig, dx, dy in ((7.5, 1.3, -2.2), (19.0, 0.0, 0.0), (31.0, 4.0, 6.5)):
    xx = rng.random((143, 211, 4), np.float32)
    want = R.gauss_shift(xx, sig, dx, dy)
    got = kc.gauss(kc.upload(xx), sig, sig, dx, dy).download()
    check(f'gauss wide σ={sig} shift ({dx}, {dy})', got, want, 8e-4)   # the small passes in f16
want = R.blur_axis(R.blur_axis(x, 3.0, 1), 0.7, 0)
check('gauss directional (3, 0.7)', kc.gauss(kc.upload(x), 3.0, 0.7).download(), want, 3e-4)

# ---- Lanczos: Pillow's resize of a padded premultiplied picture, as render.py _raster does
src = np.zeros((300, 420, 4), np.uint8)
src[..., :3] = (rng.random((300, 420, 3)) * 255).astype(np.uint8)
src[..., 3] = 255
src[40:120, 50:200, 3] = 90
pim = Image.fromarray(src, 'RGBA').convert('RGBa')
pm8 = np.ascontiguousarray(np.asarray(pim))
gs = kc.upload_rgba8(pm8)
for (ow, oh, box) in ((137, 95, (16.0, 16.0, 404.0, 284.0)), (500, 330, (20.3, 18.7, 380.2, 270.9)), (61, 44, (16.5, 17.25, 393.0, 280.5))):
    want = np.asarray(pim.resize((ow, oh), Image.LANCZOS, box=box), np.float32)
    cover = (-0.4, 0.3, ow - 0.2, oh + 0.6)
    m = R.rect_mask(0, 0, ow, oh, cover)[..., None]
    al = want[..., 3:] / 255
    t0 = time.time(); got = kc.lanczos(gs, ow, oh, list(box), list(cover)).download(); dt = time.time() - t0
    check(f'lanczos {ow}×{oh} alpha', got[..., 3:], m * al, 1 / 255 + 1e-6)
    sel = (al[..., 0] > 0.5)
    check(f'lanczos {ow}×{oh} colour (opaque px)', got[..., :3][sel], (want[..., :3] / np.maximum(al, 1e-6))[sel], 1.01 / 0.35)

# ---- a picture file made ready on the GPU: grade, mirror, edge padding, Pillow's premultiplication (load_image)
for chans, grade, flip, pad in ((4, (('contrast', 1.18), ('saturation', 0.7), ('brightness', 1.06)), True, 16),
                                (3, (('sepia', 0.4), ('gamma', 1.3)), False, 0), (4, None, False, 16), (3, (('contrast', 0.8),), True, 0)):
    raw = (rng.random((37, 53, chans)) * 256).astype(np.uint8)
    a = raw if chans == 4 else np.concatenate([raw, np.full((37, 53, 1), 255, np.uint8)], -1)
    b = a.copy()
    b[..., :3] = R.apply_grade(b[..., :3], grade)
    if flip:
        b = b[:, ::-1]
    b = np.pad(b, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    want = np.asarray(Image.fromarray(np.ascontiguousarray(b), 'RGBA').convert('RGBa')).astype(np.float32)
    ops = [(G.GRADE_OPS[k], float(v)) for k, v in grade] if grade else None
    got = kc.prep8(np.ascontiguousarray(raw), ops, flip, pad).download().astype(np.float32)
    check(f'prep8 {chans} каналов, цветокор {len(grade or ())}, зеркало {flip}, поля {pad}', got, want, 0.5 if grade else 0)

# ---- to 8 bit with the dither
img = rng.random((50, 70, 3), np.float32) * 260 - 2
d = (rng.random((50, 70), np.float32) + rng.random((50, 70), np.float32) - 1)
want = np.clip(np.round(img + d[..., None]), 0, 255).astype(np.uint8)
got = kc.to_u8(kc.upload(np.ascontiguousarray(np.concatenate([img, np.ones((50, 70, 1), np.float32)], -1))), kc.upload_plane(d))
check('to_u8 + dither', got[..., :3], want, 1)
print('ВСЁ СХОДИТСЯ' if ok else 'ЕСТЬ РАСХОЖДЕНИЯ')
