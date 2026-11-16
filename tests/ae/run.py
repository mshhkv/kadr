"""The AE coverage films (AE.md): every tests/ae/*.yaml must load without errors, pass `kadr check` without errors,
draw the same on both renderers and match its kept frames (tests/ae/ref). python tests/ae/run.py [save] [sheet-dir]"""
import glob, os, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from kadr.doc import Film, DocError
from kadr.render import Renderer, to_u8
from kadr.gpu import GpuRenderer
from kadr import check

save = 'save' in sys.argv[1:]
sheet = next((a for a in sys.argv[1:] if a != 'save'), None)
bad = []
for y in sorted(glob.glob(os.path.join(HERE, '*.yaml'))):
    name = os.path.splitext(os.path.basename(y))[0]
    if name.startswith('_'):                                   # _showcase.yaml: all of them in a row, for human eyes
        continue
    try:
        f = Film(y)
    except DocError as e:
        print(f'FAIL {name}: документ\n{e}'); bad.append(name); continue
    errs = [i for i in check.run(f) if i.level == 'error']
    if errs:
        print(f'FAIL {name}: check\n' + '\n'.join(map(str, errs))); bad.append(name); continue
    c, g, row, worst, refd = Renderer(f), GpuRenderer(f), [], 0, 0.0
    for n, u in enumerate((0.2, 0.5, 0.8)):
        t = u * f.duration
        a, b = to_u8(c.frame(t)), g.frame_u8(t)
        d = np.abs(a.astype(int) - b.astype(int))
        worst = max(worst, float((d > 3).mean() * 100))
        ref = os.path.join(HERE, 'ref', f'{name}_{n}.png')
        if save or not os.path.exists(ref):
            Image.fromarray(b).save(ref)
        else:
            r = np.asarray(Image.open(ref).convert('RGB')).astype(int)
            refd = max(refd, float((np.abs(r - b.astype(int)) > 3).mean() * 100) if r.shape == b.shape else 100.0)
        row.append(b)
    ok = worst <= 0.5 and refd <= 0.5
    print(f'{"OK  " if ok else "FAIL"} {name}: движки расходятся на {worst:.2f} % пикселей, с эталоном — на {refd:.2f} %')
    if not ok:
        bad.append(name)
    if sheet:
        Image.fromarray(np.concatenate(row, 1)).save(os.path.join(sheet, name + '.jpg'), quality=88)
print('ВСЁ СХОДИТСЯ' if not bad else f'НЕ ПРОШЛИ: {bad}')
sys.exit(1 if bad else 0)
