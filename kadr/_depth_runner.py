"""Run by the Python that has torch and transformers (kadr/depth.py finds it): depth maps of pictures.
python _depth_runner.py MODEL EDGE REGIONS SRC OUT [SRC OUT …] → one JSON line. REGIONS: JSON [{pts: [[x, y], …],
}] —
surfaces of the photo that are smooth whatever is printed on them"""
import json, sys, time

model, edge, regions, pairs = sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4:]
t0 = time.time()
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _depth_smooth import smooth                               # numpy and Pillow only: tests call it without the model

dev = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    proc = AutoImageProcessor.from_pretrained(model, local_files_only=True)
    net = AutoModelForDepthEstimation.from_pretrained(model, local_files_only=True).to(dev).eval()
except Exception as e:
    print(json.dumps({'error': f'модели {model} нет на этом компьютере ({type(e).__name__}): kadr depth --download'}))
    sys.exit(2)
loaded, done = time.time() - t0, []
for src, out in zip(pairs[0::2], pairs[1::2]):
    t1 = time.time()
    im = Image.open(src).convert('RGB')
    inputs = proc(images=im, return_tensors='pt').to(dev)
    with torch.no_grad():
        d = net(**inputs).predicted_depth                      # larger = nearer
    d = torch.nn.functional.interpolate(d[:, None].float(), size=(im.height, im.width), mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
    lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.5)     # the picture's own range, a few stray pixels aside
    n = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    rng = np.random.default_rng(7)
    agreed = [round(smooth(n, r['pts'], 2, rng), 2) for r in regions]
    g = Image.fromarray(np.round(n * 255).astype(np.uint8), 'L')     # white is near — what the parallax effect reads
    if edge > 0:
        # a thing's rim in the photo is a pixel or two wider than in the map, and a rim left at the background's depth
        # tears off when the picture is shifted: the near depth is spread outwards by `edge` thousandths of the width
        from PIL import ImageFilter
        r = max(1, int(round(edge * im.width / 1000)))
        g = g.filter(ImageFilter.MaxFilter(2 * r + 1)).filter(ImageFilter.GaussianBlur(max(0.6, r / 3)))
    g.save(out)
    done.append({'src': src, 'out': out, 'size': [im.width, im.height], 'seconds': round(time.time() - t1, 2), 'agreed': agreed})
print(json.dumps({'device': dev, 'load_seconds': round(loaded, 2), 'maps': done}))
