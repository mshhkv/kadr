"""kadr depth — a depth map of a photograph (white is near), the second picture of the `parallax` effect:

    kadr depth photo.jpg                →  photo.depth.png beside it
    - {id: d, type: image, src: photo.depth.png, size: …, position: …, visible: false}      # same box as the photo
    - {id: p, type: image, src: photo.jpg, …, effects: [{type: parallax, map: d, shift: {0: [-40, 0], 4: [40, 0]}}]}

A picture printed on a flat thing (a bowl on a cereal box, a poster, a screen) reads as relief to the model. Name the
surface and its depth is made smooth, so the print rides its surface as print does:
    kadr depth box.jpg --smooth "50,300 465,300 465,965 50,965" --smooth "543,510,850,960"
(a rectangle x0,y0,x1,y1 or a polygon, photo px: the whole face of the box, the can's wall — the surface, not the print).

The model is Depth Anything V2 Small (Apache-2.0, 99 MB; the larger ones are non-commercial) through transformers and
torch. Kadr itself does not depend on them: the map is made by whichever Python has both — $KADR_ML_PYTHON, the path
written in ~/.kadr/ml_python, or this one — and read later as an ordinary picture; rendering never runs the model."""
import json, os, subprocess, sys

MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'
NEEDS = 'import torch, transformers, PIL, numpy'


def ml_python():
    """the interpreter that can run the model, or None"""
    cands = [os.environ.get('KADR_ML_PYTHON')]
    cfg = os.path.expanduser('~/.kadr/ml_python')
    if os.path.exists(cfg):
        cands.append(open(cfg, encoding='utf-8').read().strip())
    cands.append(sys.executable)
    for py in cands:
        if py and os.path.exists(py) and subprocess.run([py, '-c', NEEDS], capture_output=True).returncode == 0:
            return py
    return None


def out_path(src):
    return os.path.splitext(src)[0] + '.depth.png'


# the exact snapshot each known model is taken at: what is downloaded is what was looked at, not whatever the page holds today
REVISIONS = {MODEL: '5426e4f0f36572d16453bbda7a8389317b1bef99'}


def download(model=MODEL):
    py = ml_python()
    if py is None:
        raise RuntimeError(_no_python())
    code = ("from huggingface_hub import snapshot_download; "
            f"print(snapshot_download({model!r}, revision={REVISIONS.get(model)!r}, allow_patterns=['config.json', 'model.safetensors', 'preprocessor_config.json']))")
    r = subprocess.run([py, '-c', code], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'модель не скачалась')
    return r.stdout.strip().splitlines()[-1]


def _no_python():
    return ('нет Python с torch и transformers: укажите его в переменной KADR_ML_PYTHON или в файле ~/.kadr/ml_python '
            '(например, окружение, где стоят torch и transformers)')


def region(text):
    """«x0,y0,x1,y1» (a rectangle) or «x,y x,y x,y …» (a polygon), photo px → {pts}"""
    nums = [float(v) for v in text.replace(',', ' ').split()]
    if len(nums) == 4:
        x0, y0, x1, y1 = nums
        pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    elif len(nums) >= 6 and len(nums) % 2 == 0:
        pts = [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]
    else:
        raise RuntimeError(f'область «{text}»: нужен прямоугольник x0,y0,x1,y1 или многоугольник из трёх и более точек x,y')
    return {'pts': pts}


def run(srcs, outs=None, model=MODEL, edge=4.0, regions=()):
    """depth maps of the pictures `srcs` → the runner's report {device, load_seconds, maps: [{src, out, size, seconds,
    agreed}]}. edge: how far the near depth is spread past a thing's rim, thousandths of the picture's width (0: the
    model's map as it is). regions: [{pts}] — surfaces that are smooth (flat or gently curved) whatever is printed on
    them: the depth there becomes that smooth surface, meeting its surroundings at the region's edge (a printed bowl
    stops bulging out of its box); `agreed` — the share of each region the surface was found from"""
    py = ml_python()
    if py is None:
        raise RuntimeError(_no_python())
    outs = outs or [out_path(s) for s in srcs]
    args = [x for pair in zip(srcs, outs) for x in pair]
    r = subprocess.run([py, os.path.join(os.path.dirname(os.path.abspath(__file__)), '_depth_runner.py'), model, str(edge), json.dumps(list(regions)), *args], capture_output=True, text=True)
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''
    try:
        rep = json.loads(line)
    except ValueError:
        raise RuntimeError((r.stderr.strip().splitlines() or ['карта глубины не посчиталась'])[-1])
    if 'error' in rep:
        raise RuntimeError(rep['error'])
    return rep
