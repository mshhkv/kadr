"""kadr selftest — renders a slow push-in on small text and a camera pass at 1x and at 8x (box-downsampled = ideal),
   tracks each text line's ink centroid and reports how far the 1x path wobbles around the ideal one.
   Chrome measured 0.29–0.31 px on the same kind of move; the guarantee here is < 0.08 px."""
import os, tempfile
import numpy as np
import yaml
from .doc import Film
from .render import Renderer

LIMIT = 0.08
K = 8
DOC = {
    'film': {'size': [480, 270], 'fps': 25, 'duration': 2, 'background': '#0b1530'},
    'layers': [
        {'id': 'push', 'type': 'text', 'text': 'Источник: NASA · Πηγή: NASA', 'size': 14, 'position': [240, 90],
         'scale': {0: 1, 2: 1.3}},
        {'id': 'cam', 'type': 'camera', 'zoom': 1000, 'position': {0: [-30, 0, 0], 2: [30, 0, 250]}},
        {'id': 'pass', 'type': 'text', 'text': 'Капитуляция гарнизона · Σιγκαπούρη', 'size': 16, 'z': 1100, 'position': [0, 40]},
    ],
}


def centroids(img, bands):
    g = img.mean(axis=2) - 30
    g = np.clip(g, 0, None)
    out = []
    for y0, y1 in bands:
        w = g[y0:y1]
        ys, xs = np.mgrid[y0:y1, 0:g.shape[1]]
        out.append(((w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()))
    return out


def run():
    path = os.path.join(tempfile.mkdtemp(), 'selftest.yaml')
    yaml.safe_dump(DOC, open(path, 'w', encoding='utf-8'), allow_unicode=True)
    film = Film(path)
    r1, r8 = Renderer(film, 1), Renderer(film, K)
    bands = [(60, 120), (160, 230)]
    p1, p8 = [], []
    for i in range(film.frames):
        t = i / film.fps
        p1.append(centroids(r1.frame(t), bands))
        a = r8.frame(t)
        a = a.reshape(film.H, K, film.W, K, 3).mean(axis=(1, 3))
        p8.append(centroids(a, bands))
    d = np.array(p1) - np.array(p8)                       # frames × lines × 2
    d -= d.mean(axis=0)
    wob = np.sqrt((d ** 2).sum(-1).mean(0))
    ok = all(w < LIMIT for w in wob)
    for name, w in zip(['наезд на текст', 'пролёт камеры'], wob):
        print(f'{name:16} уход с траектории {w:.3f} px  (порог {LIMIT})')
    print('OK: дрожи нет' if ok else 'ОШИБКА: текст дрожит')
    return 0 if ok else 1
