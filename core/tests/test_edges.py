"""edge coverage of a turned rectangle against the exact area (16×16 samples per pixel): CPU (Skia) and GPU"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from kadr.doc import Film
from kadr.render import Renderer
from kadr.gpu import GpuRenderer
errs = {'cpu (Skia)': [], 'gpu': []}
for rot in (3.0, 17.0, 33.0, 45.0, 61.0, 88.5):
    doc = {'film': {'size': [200, 200], 'fps': 25, 'duration': 1, 'background': '#000000'},
           'layers': [{'id': 'r', 'type': 'solid', 'color': '#ffffff', 'size': [91.3, 57.7], 'position': [100.37, 99.81], 'rotation': rot}]}
    f = Film(data=doc)
    n = 16
    ys, xs = np.mgrid[0:200 * n, 0:200 * n].astype(np.float64)
    xs, ys = (xs + 0.5) / n - 100.37, (ys + 0.5) / n - 99.81
    th = math.radians(rot)
    u = xs * math.cos(th) + ys * math.sin(th); v = -xs * math.sin(th) + ys * math.cos(th)
    ref = ((np.abs(u) <= 91.3 / 2) & (np.abs(v) <= 57.7 / 2)).reshape(200, n, 200, n).mean(axis=(1, 3)) * 255
    edge = (ref > 0.5) & (ref < 254.5)
    for k, r in (('cpu (Skia)', Renderer(f)), ('gpu', GpuRenderer(f))):
        errs[k].append(np.abs(r.frame(0.0)[..., 0] - ref)[edge].mean())
for k, v in errs.items():
    print(f'{k:11} средняя ошибка покрытия на краю: {np.mean(v):.2f} уровня (по углам: {" ".join(f"{x:.2f}" for x in v)})')
