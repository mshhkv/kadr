"""CPU regression: hash the CPU renderer's frames of every example and of the small test films here at 4 moments;
`reg.py save` (after a change meant to alter pixels) or `reg.py cmp` (after one that must not)"""
import sys, glob, json, hashlib, os, traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from kadr.doc import Film
from kadr.render import Renderer
OUT = os.path.join(HERE, 'base.json')
res = {}
for y in sorted(glob.glob(os.path.join(ROOT, 'examples/*/*.yaml'))) + sorted(glob.glob(os.path.join(HERE, '*.yaml'))):
    if 'minute' in os.path.basename(y) and 'examples' in y:      # a half-hour film: too long for this
        continue
    try:
        f = Film(y)
        r = Renderer(f, 1, (0, 0, f.W, f.H))
        hs = []
        for u in (0.1, 0.35, 0.6, 0.85):
            img = r.frame(min(f.duration - 1e-3, u * f.duration))
            hs.append(hashlib.md5(np.round(img, 3).tobytes()).hexdigest()[:10])
        res[os.path.relpath(y, ROOT)] = hs
    except Exception as e:
        res[os.path.relpath(y, ROOT)] = 'ERR ' + repr(e)[:120]
if len(sys.argv) > 1 and sys.argv[1] == 'save':
    json.dump(res, open(OUT, 'w'), indent=0)
    print(len(res), 'saved;', sum(isinstance(v, str) for v in res.values()), 'errors')
    for k, v in res.items():
        if isinstance(v, str): print(k, v)
else:
    base = json.load(open(OUT))
    bad = [k for k in base if base[k] != res.get(k)]
    print(len(base) - len(bad), 'same,', len(bad), 'differ')
    for k in bad: print(' ', k, base[k], '→', res.get(k))
