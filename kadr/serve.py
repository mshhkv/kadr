"""kadr serve — Kadr as an MCP server (stdio) for an agent.

One process lives for the agent's whole session: fonts, pictures, footage readers and the GPU stay warm, so a frame is
back in a fraction of a second instead of a new Python with everything loaded again. A film is read again only when
its document (or anything it includes) changed on disk; pictures are kept by file and modification time.

Tools: frame (a frame as a picture the model sees), sheet (a contact sheet), check (the defect report), info, render
(the whole film in the background) and render_status."""
import functools, io, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image as PILImage
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .doc import DocError, Film

mcp = MCPServer(
    'kadr',
    instructions=(
        "Kadr — the user's video engine: a film is a YAML document, frames come from the GPU renderer kept warm here. "
        "Paths are the film document's path (absolute, or relative to the server's working directory). Times: seconds "
        "(14.2) or anything a document accepts (word:Сингапур#2.end+0.3). frame/sheet return pictures you see directly; "
        "pass out= to also keep the full-size PNG on disk. After editing a document just call again — it is re-read when "
        "its files change. render runs the whole film in the background (kadr render --gpu); poll render_status."
    ),
)

_films = {}                                    # path → (stamp, Film, renderer)
_gpu = None                                    # the one thread every picture is made on (see on_gpu)
_jobs = {}                                     # id → (Popen, log path, out path, started)


def on_gpu(fn):
    """A tool that draws runs on one thread of this server's own, one call at a time.

    The core keeps a GPU device per thread, and what a renderer has put on the GPU (pictures, fonts, footage) belongs
    to the device of the thread that made it. The MCP library runs each call on whichever of its worker threads is
    free, so a renderer kept from an earlier call was sometimes asked for a frame from another thread: the core died
    there («Cannot get non-existent resource»), its error is not an ordinary Python exception, the library sent no
    answer, and the agent waited out the client's whole timeout (half an hour, 16.11.2026) on a sheet that takes three
    seconds. Now: one thread for all of it, and whatever goes wrong inside comes back as words at once — after a
    failure of the core itself the thread and everything kept on it are dropped, the next call starts clean."""
    @functools.wraps(fn)
    def run(*a, **kw):
        global _gpu
        if _gpu is None:
            _gpu = ThreadPoolExecutor(1, thread_name_prefix='kadr-gpu')
        try:
            return _gpu.submit(fn, *a, **kw).result()
        except Exception:
            raise
        except BaseException as e:                         # a panic of the Rust core (pyo3's PanicException), not Ctrl-C
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            _films.clear()
            _gpu.shutdown(wait=False)
            _gpu = None
            raise ToolError(f'ядро Kadr остановилось на этом вызове ({type(e).__name__}: {str(e)[:300]}). Сервер жив и начал с чистого листа: '
                            f'повторите вызов; если повторится — тот же кадр из командной строки (kadr frame … --gpu) покажет, в фильме дело или в сервере')
    return run


def _stamp(path, film=None):
    """what a film's document depends on: its file and every file it was built from (styles, templates, plugins it
    names), by modification time and size"""
    files = [path] + sorted(getattr(film, 'sources', ()) or ())
    out = []
    for f in files:
        try:
            st = os.stat(f)
            out.append((f, st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((f, None, None))
    return tuple(out)


def _film(path):
    from .gpu import GpuRenderer
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise ToolError(f'нет файла {path}')
    hit = _films.get(path)
    if hit is not None and hit[0] == _stamp(path, hit[1]):
        return hit[1], hit[2]
    try:
        film = Film(path)
    except DocError as e:
        raise ToolError(f'документ не читается:\n{e}')
    r = GpuRenderer(film)
    if hit is not None:                        # pictures on the GPU are kept by file and its modification time
        r._src8.update(hit[2]._src8)
    _films[path] = (_stamp(path, film), film, r)
    return film, r


def _view(img, width):
    """what the model looks at: scaled to `width`, JPEG 90 (a tenth of a PNG's size; the full PNG goes to `out`)"""
    if width and img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.convert('RGB').save(buf, 'JPEG', quality=90)
    return Image(data=buf.getvalue(), format='jpeg')


def _out(path, kinds=('.png', '.jpg', '.jpeg', '.webp')):
    """where a tool may save: a file of the kind it makes, and never over a file of another kind (an agent told by a
    document's text to «save to ~/.zshrc» is refused here)"""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.splitext(path)[1].lower() not in kinds:
        raise ValueError(f'out: {os.path.basename(path)} — сюда сохраняются только файлы {", ".join(kinds)}')
    return path


@mcp.tool()
@on_gpu
def frame(film: str, at: str, width: int = 540, out: str = '') -> list:
    """One frame of the film at time `at`, as a picture (scaled to `width` px wide; 0 = full size). `out`: also save
    the full-size PNG there."""
    f, r = _film(film)
    t = f.time(at)
    t0 = time.time()
    img = PILImage.fromarray(r.frame_u8(t))
    dt = time.time() - t0
    if out:
        img.save(_out(out))
    w = f.word_near(t)
    return [f'{t:.3f} с' + (f' · «{w}»' if w else '') + f' · {dt * 1000:.0f} мс' + (f' · {out}' if out else ''), _view(img, width)]


@mcp.tool()
@on_gpu
def sheet(film: str, every: float = 1.0, at: list[str] | None = None, start: str = '', end: str = '', width: int = 1400,
          out: str = '') -> list:
    """A contact sheet: frames every `every` s between start and end (default the whole film), or at the given times,
    four across, each captioned with its time and the word spoken then. `width`: the sheet's width as returned."""
    from .cli import sheet_image
    f, r = _film(film)
    if at:
        times = [f.time(x) for x in at]
    else:
        a = f.time(start) if start else 0.0
        b = f.time(end) if end else f.duration
        times = list(np.arange(a, b, every))
    if len(times) > 64:
        raise ToolError(f'{len(times)} кадров — больше 64; возьмите шаг крупнее или отрезок короче')
    t0 = time.time()
    img = sheet_image(f, r, times)
    dt = time.time() - t0
    if out:
        img.save(_out(out))
    return [f'{len(times)} кадров за {dt:.1f} с' + (f' · {out}' if out else ''), _view(img, width)]


@mcp.tool()
@on_gpu
def check(film: str, strict: bool = False) -> str:
    """The defect report of the document (kadr check): what will look wrong, by time and layer — no rendering."""
    from .check import run, report
    f, _ = _film(film)
    return report(run(f, strict=strict))


@mcp.tool()
@on_gpu
def info(film: str) -> str:
    """Size, frame rate, duration, layers and words of the film."""
    f, _ = _film(film)
    kinds = {}
    for L in f.layers:
        kinds[L['type']] = kinds.get(L['type'], 0) + 1
    return (f'{f.W}×{f.H}, {f.fps:g} к/с, {f.duration:.2f} с ({f.frames} кадров); слои: '
            + ', '.join(f'{k} {v}' for k, v in sorted(kinds.items())) + f'; слов озвучки: {len(f.words)}')


@mcp.tool()
def render(film: str, out: str, hw: bool = False, draft: bool = False, start: str = '', end: str = '', alpha: bool = False) -> str:
    """The whole film (or start..end) to an mp4 in the background — kadr render --gpu. hw: the Mac's hardware HEVC
    (as good as x264 medium, almost no CPU, twice the file); draft: x264 veryfast. alpha: with transparency, for laying
    over other footage in an editor — the film's `background` is left out, out must be .mov (ProRes 4444), .webm (VP9)
    or a folder pattern dir/%05d.png. Returns a job id for render_status."""
    film = os.path.abspath(os.path.expanduser(film))
    out = _out(out, ('.mp4', '.mov', '.webm', '.png'))
    cmd = [sys.executable, '-m', 'kadr', 'render', film, '-o', out, '--gpu'] + (['--hw'] if hw else []) + (['--draft'] if draft else [])
    cmd += (['--from', start] if start else []) + (['--to', end] if end else []) + (['--alpha'] if alpha else [])
    log = tempfile.NamedTemporaryFile('w', prefix='kadr_render_', suffix='.log', delete=False)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=tempfile.gettempdir(), env={**os.environ, 'PYTHONSAFEPATH': '1'})   # not in the film's folder: nothing there is to be imported
    job = f'r{len(_jobs) + 1}'
    _jobs[job] = (p, log.name, out, time.time())
    return f'{job}: собирается {out}'


@mcp.tool()
def render_status(job: str) -> str:
    """How a background render is going: frames done so far, or the result."""
    if job not in _jobs:
        raise ToolError(f'нет сборки {job}; есть: {", ".join(_jobs) or "ни одной"}')
    p, log, out, t0 = _jobs[job]
    text = open(log, encoding='utf-8', errors='replace').read()
    last = text.replace('\r', '\n').strip().splitlines()[-1:] or ['']
    if p.poll() is None:
        return f'идёт {time.time() - t0:.0f} с: {last[0]}'
    if p.returncode:
        return f'ошибка (код {p.returncode}):\n{text[-1500:]}'
    return f'готово: {last[0]}'


@mcp.tool()
def depth(photos: list[str], edge: float = 4.0, smooth: list[str] | None = None) -> str:
    """Depth maps of photographs for the `parallax` effect (white is near): each is saved beside its photo as
    <name>.depth.png. Use it as a hidden image layer of the same box (`visible: false`) and give it to the effect:
    effects: [{type: parallax, map: <that layer>, shift: {0: [-30, 0], 4: [30, 0]}}]. A fraction of a second a photo.
    A picture printed on a flat thing (a bowl on a box, a poster, a screen, a label on a can) reads as relief: for ONE
    photo name such surfaces — smooth: ["x0,y0,x1,y1" or "x,y x,y x,y …"], photo px, the whole surface (the box's face),
    not just the print — and their depth becomes smooth; the answer says what share of each region the surface was
    found from (under ~30 %: the region is wider than the surface)."""
    from . import depth as dp
    miss = [f for f in photos if not os.path.exists(f)]
    if miss:
        return 'нет файла: ' + ', '.join(miss)
    try:
        regions = [dp.region(t) for t in smooth or []]
        if regions and len(photos) != 1:
            return 'smooth задаётся для одного фото: области — в его пикселях'
        rep = dp.run(photos, edge=edge, regions=regions)
    except RuntimeError as e:
        return str(e)
    out = []
    for m in rep['maps']:
        out.append(f"{m['out']}  {m['size'][0]}×{m['size'][1]}  {m['seconds']} с")
        out += [f"  область {i + 1}: поверхность найдена по {a * 100:.0f} % её площади" for i, a in enumerate(m.get('agreed') or [])]
    return '\n'.join(out)


@mcp.tool()
def track(clip: str, name: str, at: float, region: str, model: str = '', start: float | None = None, end: float | None = None, sheet: str = '', surface: str = '') -> str:
    """Follow a thing through a video clip: `region` — "x0,y0,x1,y1" or four corners "x,y x,y x,y x,y" (top left, top
    right, bottom right, bottom left) in the clip's px at second `at` (look at a frame of the clip first). Saved beside
    the clip as <name of clip>.track.json; in a document, with a video layer of that clip:
    position: =track("layer", "name").position · corner_pin: =track("layer", "name").corners (a screen replaced) ·
    .scale · .rotation · opacity: =1 - track("layer", "name").lost. model: plane (default for four corners: a flat
    thing from any side), similar (default for a rectangle: place, size, turn), point, affine. sheet: a .jpg of frames
    with the found outline — look at it. surface: the four corners a pinned layer lands on when they are not the
    region's own (region: a screen with its frame, which has drawing to hold on to; surface: the glass inside)."""
    from . import track as tk
    if not os.path.exists(clip):
        return 'нет файла: ' + clip
    try:
        quad, four = tk.region(region)
        rep = tk.track_clip(clip, name, at, quad, model or ('plane' if four else 'similar'), start, end, surface=tk.region(surface)[0] if surface else None)
    except (ValueError, OSError) as e:
        return str(e)
    out = f"{rep['out']}: трек «{name}», кадров {rep['frames']} ({rep['first'] / rep['fps']:.2f}–{rep['last'] / rep['fps']:.2f} с), потерян на {rep['lost']}, среднее сходство {rep['q']:.2f}"
    if sheet:
        out += '\nлист кадров: ' + tk.sheet(clip, name, sheet)
    return out


@mcp.tool()
def stabilize(clip: str, region: str = '') -> str:
    """The camera's path through a video clip, saved beside it as <name of clip>.stab.json. In a document:
    `stabilize: true` on the video layer (a calm move; {smooth: seconds, lock: true — a tripod, zoom: auto | number}).
    The answer says how far the camera wanders and how much the picture must be enlarged to hide its edges. region:
    "x0,y0,x1,y1" of the first frame — judge the camera by this part only (a shore, a wall), for a scene half of which
    moves by itself (water, a crowd)."""
    from . import track as tk
    if not os.path.exists(clip):
        return 'нет файла: ' + clip
    try:
        rep = tk.stabilize_clip(clip, region=tk.region(region)[0] if region else None)
    except (ValueError, OSError) as e:
        return str(e)
    return (f"{rep['out']}: кадров {rep['frames']}, среднее сходство {rep['q']:.2f}; камера уходит до {rep['shift']:.0f} px, поворот до {rep['turn']:.1f}°; "
            f"увеличение {rep['zoom_smooth']:.3f} при stabilize: true, {rep['zoom_lock']:.3f} при lock: true")


@mcp.tool()
def matte(files: list[str], what: str = 'person', point: str = '', sheet: str = '') -> str:
    """Cut a person (or the main thing) out of video clips or photographs: saved beside each file as <name>.matte.mp4
    (or .matte.png) — white where the person is, frame for frame. In a document: `cutout: true` on a video or image
    layer of that file shows only the cut-out (a word behind a speaker: the clip, the word, the clip again with
    cutout: true); cutout: {invert: true} — everything but; {feather: 3} — a soft edge. what: person (people),
    subject (whatever stands out: a product, an animal; point "x,y" in the first frame's px picks one thing), soft
    (people with soft edges — fine hair). macOS only, built in, nothing downloaded. sheet: a .jpg of frames with the
    cut-out over green — look at it."""
    from . import matte as mt
    out = []
    try:
        pt = [float(v) for v in point.replace(',', ' ').split()] if point else None
        for f in files:
            if not os.path.exists(f):
                return 'нет файла: ' + f
            rep = mt.run(f, what, pt)
            out.append(mt.describe(rep))
            if sheet and len(files) == 1 and rep['out']:
                out.append('лист кадров: ' + mt.sheet(f, sheet))
    except (RuntimeError, ValueError, OSError) as e:
        return str(e)
    return '\n'.join(out)


@mcp.tool()
def reframe(clip: str, what: str = 'person', sheet: str = '') -> str:
    """Find where the person is through a clip, so that a frame narrower than the footage keeps them in (a horizontal
    film cut into vertical reels): saved beside the clip as <name>.frame.json. In a document: `reframe: true` on the
    video layer of that clip slides it so that the person — the face when one is seen, else the body's middle — stands
    in the middle of the frame, evened like a calm operator, never showing the footage's edge, each shot of the clip
    by itself; reframe: {smooth: 0.3} follows closer. what: person, or subject (whatever stands out: a product, an
    animal). macOS only. sheet: a .jpg of frames with the 9:16 window drawn where it would stand — look at it."""
    from . import reframe as rf
    if not os.path.exists(clip):
        return 'нет файла: ' + clip
    try:
        rep = rf.run(clip, what)
        out = (f"{rep['out']}: планов {rep['shots']}; из {rep['samples']} просмотренных кадров найден на {rep['seen']}, лицо — на {rep['faces']}; "
               f"по ширине ходит на {rep['wander']:.0f} px из {rep['size'][0]}")
        if rep['seen'] < rep['samples'] * .5:
            out += '\nнайден меньше чем на половине кадров: там окно встанет посередине (для предмета или зверя — what: subject)'
        if sheet:
            out += '\nлист кадров: ' + rf.sheet(clip, sheet)
    except (RuntimeError, ValueError, OSError) as e:
        return str(e)
    return out


@mcp.tool()
def beats(file: str) -> str:
    """The pulse of a piece of music: its tempo and the second of every beat. In a document whose film.audio has that
    file, times may be written as beat:1, beat:9+0.1 (cuts and hits on the beat), formulas may read beat() — the
    place within the beat, 0…1."""
    from . import sound
    if not os.path.exists(file):
        return 'нет файла: ' + file
    try:
        tempo, bt = sound.find_beats(sound.track(os.path.abspath(file)))
    except ValueError as e:
        return str(e)
    if len(bt) < 2:
        return f'{file}: ровного бита не слышно'
    return f'{file}: {tempo:.1f} ударов в минуту, {len(bt)} битов (в документе: beat:1 … beat:{len(bt)})\n' + ' '.join(f'{t:.2f}' for t in bt)


@mcp.tool()
def cuts(clip: str) -> str:
    """Where a clip jumps from one shot to another: the seconds of every cut and the shots between them (to lay each
    shot as its own layer with from/until, to cut a film on its own cuts)."""
    from . import track as tr
    if not os.path.exists(clip):
        return 'нет файла: ' + clip
    try:
        found, fps, n = tr.find_cuts(clip)
    except (ValueError, OSError) as e:
        return str(e)
    edges = [0.0] + found + [n / fps]
    return f'{clip}: {n} кадров, {n / fps:.2f} с; склеек {len(found)}' + (' — один план' if not found else '') + ''.join(
        f'\n  план {i + 1}: {t0:.2f} – {t1:.2f} с ({t1 - t0:.2f} с)' for i, (t0, t1) in enumerate(zip(edges, edges[1:])))


@mcp.tool()
def puppet(picture: str, out: str, pins: list[str] | None = None, mesh: int = 0, cutout: bool = False, pose: bool = False) -> str:
    """Where to stick pins into a picture that a `puppet` layer will bend (After Effects: the Puppet tool): draws the
    picture under a ruler with its mesh and the pins tried ("330,250", px of the picture) into `out` (.jpg) — look
    at it. In a document: {type: puppet, src: man.png, pins: {hand: {at: [330, 250], to: {0: [330, 250], 1: [360, 150]}},
    hip: {at: [200, 420]}}} — a pin with `to` (where it is led) or `move` (a shift) leads the picture, a pin with
    neither holds it; what must stay still needs pins of its own. cutout: the picture is a photograph whose cut-out was
    made by the matte tool. pose: a photograph of a person — macOS finds the joints (head, neck, shoulders, elbows,
    wrists, hips, knees, ankles; _l is the one further left in the picture) and they come back as pins ready for the
    document. `check` names folds, stretching and jerks."""
    from . import puppet as pp
    if not os.path.exists(picture):
        return 'нет файла: ' + picture
    try:
        pts = [tuple(float(v) for v in p.replace(',', ' ').split()) for p in (pins or [])]
        found = pp.joints(picture, cutout, mesh or None) if pose else {}
        cell, cells, off = pp.sheet(picture, out, pts + list(found.values()), mesh or None, cutout, [''] * len(pts) + list(found))
    except (ValueError, OSError, RuntimeError) as e:
        return str(e)
    text = f'{out}: сетка с ячейкой {cell} px, ячеек {cells}' + ''.join(f'\nбулавка {p[0]:g},{p[1]:g} стоит на прозрачном месте — мимо сетки' for p in off)
    if pose:
        text += ('\nсуставы (в документ):\n  pins:' + ''.join(f'\n    {k}: {{at: [{x}, {y}]}}' for k, (x, y) in found.items())) if found else \
            '\nчеловека на картинке не нашлось (рисованных фигур системный поиск не видит): булавки — по листу, вручную'
    return text


@mcp.tool()
@on_gpu
def scene(scene: str, node: str = '', layer: str = '', at: str = '0', out: str = '', width: int = 1400, whole: bool = False) -> list:
    """A 3D scene from six sides in an even light (front, back, left, right, top, three quarters) and what cannot be
    right in it. `scene`: a scene's own file (.yaml in Kadr's words, or an exported .json) or a film with a scene3d
    layer (`layer`: its id; `at`: the moment whose pose is shown — a character on his rig as he stands then).
    `node`: only this part with all that hangs on it (a character alone, a mug) — its id; the answer lists the scene's
    things. The check names: a body of one thing running through a body of another (a leg through a chair's seat —
    bodies of one thing, i.e. under one top node, may overlap: that is how a thing is put together); a body poking
    through a sheet; a body wholly inside another (never seen); a solid body touching nothing (in the air); two sheets
    in one plane (they flicker). Pairs meant to meet go into the scene's `accept: [[a, b]]`. Look at the sheet after
    every change to a scene: the film's camera shows one side only. whole: frame the entire scene (by default the few
    bodies far larger than the rest — a wall, a floor — run out of the picture). A character is not built from boxes by
    hand: one node `{id: bob, shape: figure, position, height, top, pants, hair, hat, sits, …}` is a whole block figure
    with its poses (`kadr catalog figure`); look at it with node="bob"."""
    from . import scenecheck
    from .doc import DocError
    try:
        pic, words = scenecheck.look(os.path.expanduser(scene), layer or None, at, node or None, 480, True, whole)
    except DocError as e:
        raise ToolError(str(e))
    if out:
        pic.save(_out(out), quality=90)
    return ['\n'.join(words) + (f'\n{out}' if out else ''), _view(pic, width)]


def main():
    mcp.run()


if __name__ == '__main__':
    main()
