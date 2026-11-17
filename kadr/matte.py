"""kadr matte — a person (or the main thing) cut out of footage or of a photograph (AE.md 47; After Effects: Roto Brush):

    kadr matte shot.mp4                 →  shot.matte.mp4 beside it: white where the people are, frame for frame
    kadr matte photo.jpg                →  photo.matte.png
        --what person   the people of the frame (default)
        --what subject  whatever stands out — a product, an animal, a thing in a hand; with --point x,y (the clip's
                        px in its first frame) only the thing under that point, followed from frame to frame
        --what soft     the people with soft edges (fine hair, motion blur), each frame knowing the ones before it
        --sheet out.jpg frames with the cut-out over green, to look at
    in a document:
        - {id: shot, type: video, src: shot.mp4}
        - {id: word, type: text, text: "СЛОВО", …}
        - {id: front, type: video, src: shot.mp4, cutout: true}      # the person again, over the word: the word is behind them
        cutout: {invert: true}   everything but the person;   cutout: {feather: 3}   px of soft edge
    (`cutout` is a matte under the hood: a hidden copy of the layer that plays shot.matte.mp4 in the same place and time.)

The cutting is done by what macOS has built in — the Vision framework, the same that lifts a subject out of a picture
in Photos: nothing is downloaded, nothing leaves the machine, about 0.03–0.07 s a frame. A small Swift program
(kadr/matte.swift) is compiled on first use; it needs the Xcode command line tools (xcode-select --install).

Windows and Linux have no such thing built in, so there the program is one that is written once for that machine:
kadr/matte_tool.py (or any program named in $KADR_MATTE_TOOL) that speaks the language of matte.swift — frames in,
mattes out; see its head, AGENTS.md «Windows and Linux» and the working skeleton tests/ae/matte_tool_example.py.
Everything else here (kadr matte, cutout, kadr reframe, kadr puppet --pose) goes through tool() and does not care
which program answers."""
import hashlib, os, subprocess, sys

IMAGES = ('.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.heic', '.bmp')
WHAT = ('person', 'subject', 'soft')


def out_path(src):
    """where a file's matte lives: beside it"""
    stem, ext = os.path.splitext(src)
    return stem + ('.matte.png' if ext.lower() in IMAGES else '.matte.mp4')


OWN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matte_tool.py')     # the matte program of a system that is not macOS, once someone has written it


def _python():
    """the Python a matte program written in Python runs in: the one named for the models (as kadr depth), or this one"""
    cfg = os.path.expanduser('~/.kadr/ml_python')
    for py in (os.environ.get('KADR_ML_PYTHON'), open(cfg, encoding='utf-8').read().strip() if os.path.exists(cfg) else None):
        if py and os.path.exists(py):
            return py
    return sys.executable


def tool():
    """the matte program, as the start of a command line («W H what …» is added to it).
    $KADR_MATTE_TOOL names one on any system; macOS has its own (matte.swift, built once into ~/.cache/kadr/bin, again
    when its source changes); elsewhere it is kadr/matte_tool.py, if it has been written. A .py is run by _python()"""
    own = os.environ.get('KADR_MATTE_TOOL') or (OWN if sys.platform != 'darwin' and os.path.exists(OWN) else None)
    if own:
        if not os.path.exists(own):
            raise RuntimeError(f'KADR_MATTE_TOOL: нет файла {own}')
        return [_python(), own] if own.lower().endswith('.py') else [own]
    if sys.platform != 'darwin':
        raise RuntimeError('на этой системе программы вырезания ещё нет: на macOS её роль играет встроенная Vision, здесь её нужно один раз написать — '
                           'файл kadr/matte_tool.py (или любая программа, названная в KADR_MATTE_TOOL). Что она читает и что отвечает: AGENTS.md, '
                           '«Windows and Linux»; рабочая заготовка — tests/ae/matte_tool_example.py. Готовые вырезки (*.matte.mp4, *.matte.png) играют и без неё')
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matte.swift')
    exe = os.path.join(os.path.expanduser('~/.cache/kadr/bin'), 'kadr-matte-' + hashlib.sha1(open(src, 'rb').read()).hexdigest()[:10])
    if not os.path.exists(exe):
        os.makedirs(os.path.dirname(exe), exist_ok=True)
        r = subprocess.run(['swiftc', '-O', '-swift-version', '5', src, '-o', exe + '.tmp'], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError('не собрать программу вырезания (нужны инструменты командной строки Xcode: xcode-select --install):\n' + r.stderr[-1500:])
        os.replace(exe + '.tmp', exe)
    return [exe]


def run(src, what='person', point=None, out=None):
    """the matte of a clip or a photograph → {out, frames, empty (frames where nothing was found), cover (the share
    of the frame taken, on average)}"""
    if what not in WHAT:
        raise RuntimeError(f'--what «{what}»: нужно ' + ' | '.join(WHAT))
    if point is not None and what != 'subject':
        raise RuntimeError('--point выбирает один предмет: он работает с --what subject')
    exe, out = tool(), out or out_path(src)
    still = os.path.splitext(src)[1].lower() in IMAGES
    if still:
        from PIL import Image
        im = Image.open(src).convert('RGB')
        w, h = im.size
        r, g, b = im.split()
        feed = Image.merge('RGBA', (b, g, r, Image.new('L', im.size, 255))).tobytes()     # the program reads B, G, R, A
    else:
        from .media import probe
        p = probe(src)
        w, h = p['w'], p['h']
        rate = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', src],
                              capture_output=True, text=True).stdout.strip().split(',')[0] or str(p['fps'])
    cmd = exe + [str(w), str(h), what] + ([str(int(point[0])), str(int(point[1]))] if point else [])
    if still:
        r = subprocess.run(cmd, input=feed, capture_output=True)
        note = r.stderr.decode('utf-8', 'replace')
        if r.returncode or len(r.stdout) != w * h:
            raise RuntimeError('вырезание не удалось: ' + note[-800:])
        last = [ln for ln in note.splitlines() if ln.startswith('done')][-1].split()
        if int(last[3]) < int(last[1]):                        # nothing found: no file is written (an empty matte over a good one)
            from PIL import Image
            Image.frombytes('L', (w, h), r.stdout).save(out)
    else:
        tmp = out + '.part.mp4'
        dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-i', src, '-an', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        cut = subprocess.Popen(cmd, stdin=dec.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{w}x{h}', '-r', rate, '-i', '-', '-c:v', 'libx264',
                                '-crf', '12', '-pix_fmt', 'gray', tmp], stdin=cut.stdout, stderr=subprocess.PIPE)
        dec.stdout.close()
        cut.stdout.close()
        note = cut.stderr.read().decode('utf-8', 'replace')
        cut.wait(), dec.wait()
        err = enc.communicate()[1].decode('utf-8', 'replace')
        if cut.returncode or enc.returncode or 'done' not in note:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError('вырезание не удалось: ' + (note[-600:] + err[-600:]))
        last = [ln for ln in note.splitlines() if ln.startswith('done')][-1].split()
        if int(last[3]) < int(last[1]):
            os.replace(tmp, out)
        else:
            os.remove(tmp)
    last = [ln for ln in note.splitlines() if ln.startswith('done')][-1].split()
    return dict(out=out if int(last[3]) < int(last[1]) else None, frames=int(last[1]), empty=int(last[3]), cover=float(last[5]), size=(w, h))


def pose(src):
    """the joints of the person in a picture, as macOS sees them → {name: (x, y, confidence)} in the picture's px.
    Paired joints are named by where they are in the picture: shoulder_l is the one further left"""
    import numpy as np
    from PIL import Image
    im = Image.open(src).convert('RGB')
    k = min(1.0, 1536 / max(im.size))
    small = im.resize((max(16, int(round(im.width * k))), max(16, int(round(im.height * k))))) if k < 1 else im
    bgra = np.concatenate([np.asarray(small)[..., ::-1], np.full((small.height, small.width, 1), 255, np.uint8)], -1)
    r = subprocess.run(tool() + [str(small.width), str(small.height), 'pose'], input=bgra.tobytes(), capture_output=True)
    raw = {}
    for ln in r.stdout.decode().splitlines():
        p = ln.split()
        if len(p) == 4:
            raw[p[0]] = (float(p[1]) * im.width / small.width, float(p[2]) * im.height / small.height, float(p[3]))
    return name_joints(raw)


def name_joints(raw):
    """the system's names (a person's own left and right, here _a and _b) → names by where the joint is in the picture:
    of a pair, _l is the one further left; a joint seen without its pair has no side"""
    out = {}
    for name, v in raw.items():
        base, _, side = name.rpartition('_')
        if side not in ('a', 'b'):
            out[name] = v
            continue
        other = raw.get(f'{base}_{"b" if side == "a" else "a"}')
        out[base if other is None else f'{base}_{"l" if (v[0], side) <= (other[0], "b" if side == "a" else "a") else "r"}'] = v
    return out


def sheet(src, out, count=8):
    """frames of the clip with the cut-out kept and the rest turned green → a picture to look at"""
    import numpy as np
    from PIL import Image, ImageDraw
    m = out_path(src)
    if os.path.splitext(src)[1].lower() in IMAGES:
        pairs = [(np.asarray(Image.open(src).convert('RGB')), np.asarray(Image.open(m).convert('L')), '')]
    else:
        from .media import VideoReader
        a, b = VideoReader(src), VideoReader(m)
        picks = sorted(set(int(round(i)) for i in np.linspace(0, min(a.last, b.last), count)))
        pairs = [(a.at((i + .5) / a.fps).copy(), b.at((i + .5) / b.fps)[..., 0].copy(), f'{i / a.fps:.2f} s') for i in picks]
        a.close(), b.close()
    cols = min(4, len(pairs))
    k = min(1.0, 480 / pairs[0][0].shape[1])
    tw, th = int(pairs[0][0].shape[1] * k), int(pairs[0][0].shape[0] * k)
    page = Image.new('RGB', (cols * tw, -(-len(pairs) // cols) * (th + 18)), '#111111')
    for j, (fr, mt, label) in enumerate(pairs):
        al = mt[..., None].astype(np.float32) / 255
        comp = (fr * al + np.array([0, 200, 0], np.float32) * (1 - al)).astype(np.uint8)
        page.paste(Image.fromarray(comp).resize((tw, th)), ((j % cols) * tw, (j // cols) * (th + 18) + 18))
        ImageDraw.Draw(page).text(((j % cols) * tw + 4, (j // cols) * (th + 18) + 3), label, fill='#ffd060')
    page.save(out, quality=90)
    return out


def describe(rep):
    """the report in words, with what to do when it went badly"""
    if rep['out'] is None:
        return f"ни на одном из {rep['frames']} кадров ничего не найдено, файл не записан: людей в кадре нет? для предмета — --what subject"
    lines = [f"{rep['out']}: кадров {rep['frames']}, вырезанное занимает в среднем {rep['cover'] * 100:.0f} % кадра"]
    if rep['empty']:
        lines.append(f"  на {rep['empty']} кадрах ничего не найдено")
    return '\n'.join(lines)
