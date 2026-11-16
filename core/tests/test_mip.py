"""mip.wgsl against kadr/mip.py: a mirrored texture on a plane to the horizon, a clamped picture turned in 3D, three
motion-blur matrices — max difference in 0..255 levels"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import kadr_core as kc
from kadr import mip
from kadr.render import turn3d

rng = np.random.default_rng(3)
tex = rng.random((64, 128, 4)).astype(np.float32)
tex[..., 3] = 0.6 + 0.4 * tex[..., 3]
tex[..., :3] *= 255 * tex[..., 3:]                       # premultiplied
lv = mip.pyramid(tex)
gp = kc.upload(mip.packed(lv))
W, H = 320, 240
def M_at(rx, dx=0.0):
    return np.array([[1, 0, 160 + dx], [0, 1, 200], [0, 0, 1.0]]) @ turn3d(rx, 8, 300) @ np.array([[0.5, 0, -2000], [0, 0.5, -4000], [0, 0, 1.0]])
worst = 0
for wrap, cw, ch, tm in ((2, 4000, 8000, (1 / 3, 1 / 3, 37.0, 11.0)), (1, 4000, 8000, (0.5, 0.25, 0, 0)), (0, 4000, 8000, (128 / 4000, 64 / 8000, 0, 0))):
    for mats in ([M_at(70)], [M_at(70, d) for d in (-6, 0, 6)], [M_at(30)]):
        invs = [[float(v) for v in np.linalg.inv(M).ravel()] for M in mats]
        box = (0, 0, W, H)
        t0 = time.time(); rgb, a = mip.place(lv, cw, ch, invs, box, tm, wrap); t1 = time.time()
        g = kc.mip(gp, 128, 64, len(lv), cw, ch, invs, list(box), list(tm), wrap).download()
        ref = np.concatenate([rgb * a, a * 255], -1); got = np.concatenate([g[..., :3] * g[..., 3:], g[..., 3:] * 255], -1)
        d = np.abs(ref - got)
        worst = max(worst, d.max())
        print(f'wrap {wrap} matrices {len(mats)}: max {d.max():.3f} mean {d.mean():.4f} (numpy {t1 - t0:.1f} s)')
print('worst', worst)
assert worst < 2.0
