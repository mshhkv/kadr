"""kadr render | frame | sheet | check | srt | selftest"""
import argparse, os, shutil, subprocess, sys, tempfile, time
import threading
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .doc import Film, DocError
from .render import Renderer, to_u8, font_file

# the Mac's hardware HEVC at q 85: PSNR 48.2 against x264 medium's 48.0 on a 42 s reel, steadier frame to frame (mean
# jump 0.18 dB against 0.23), ~3 % of the CPU — the file twice as large. hvc1: the tag Apple players and Instagram want
# the hardware encoders `kadr render --hw` may use, best first: (ffmpeg's name, its arguments, the stream's tag in mp4).
# A row is written only after it was measured on a real machine (AGENTS.md, «Windows and Linux»): its quality figure
# is the one that gave there what x264 crf 18 medium gives, with the same colours. The Mac's: q 85.
# NVIDIA (17.11.2026, Windows 11, RTX 3070, driver 621.14, ffmpeg 8.1.2): constant QP 20, p5 + hq. Against the same
# frames at crf 0, 1080×1920, 119 frames: footage + graphics 45.97 dB / 8.6 MB (x264 crf 18 medium 45.35 / 8.0),
# pure graphics 54.4 dB / 0.96 MB (x264 52.8 / 1.12); encoding ~3× faster (0.6 s against 1.9). Not `-rc vbr -cq`:
# without a -maxrate it sits at a default bitrate ceiling — cq 14 and cq 23 gave the same 5.8 MB, 44 dB, below x264
HW_ENCODERS = [('hevc_videotoolbox', ['-q:v', '85'], 'hvc1'),
               ('hevc_nvenc', ['-preset', 'p5', '-tune', 'hq', '-rc', 'constqp', '-qp', '20'], 'hvc1')]
_hw = []


def hw_encoder():
    """the first of HW_ENCODERS that works on this machine → (ffmpeg arguments, tag), or None. «Works» is a small real
    encode: ffmpeg lists encoders whose hardware is not there. $KADR_HW_ENCODER names the one row to try (to measure
    one encoder against another on a machine that has several)"""
    if not _hw:
        _hw.append(None)
        only = os.environ.get('KADR_HW_ENCODER')
        for name, args, tag in HW_ENCODERS:
            if only and name != only:
                continue
            r = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=gray:s=640x360:r=25:d=0.2', '-c:v', name, *args, '-pix_fmt', 'yuv420p', '-f', 'null', '-'],
                               capture_output=True)
            if r.returncode == 0:
                _hw[0] = (['-c:v', name, *args, '-tag:v', tag], tag)
                break
    return _hw[0]
# GPU render (16.11.2026, 15 s of a reel): 2 processes 39 s / 3.2 GB ≈ 3 processes 38.5 s / 3.5 GB — the GPU is
# full with two; --lite: 2 threads of one process 48 s / 2.1 GB (each thread its own GPU queue; Python runs one thread
# at a time, so the threads overlap only while one waits for the GPU)
GPU_JOBS = 2
BLOCK = 50             # frames per independently encoded piece; pieces are joined without re-encoding
# No B-frames: x264's I/P/B quality cycle makes fine detail pulse (measured on archive photos: flicker ×4–7 at equal size).
SAMPLE = 1e-3          # frame i is drawn 1 ms into it: times are kept to 3 decimals, so a cut on frame k may read k/fps + 0.0005
_film = None


def default_jobs():
    """processes the machine can feed: a worker with its decoders and encoder peaks near 1.5 GB, and the rest of the
    system keeps 6 GB — on 16 GB that is 6, not one per core (10 workers once ran a Mac out of memory into a panic)"""
    ram = 16
    try:
        if sys.platform == 'darwin':
            ram = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout) / 2 ** 30
        elif sys.platform == 'win32':
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [('n', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in ('total', 'avail', 'tp', 'ap', 'tv', 'av', 'ax')]
            m = _Mem(n=ctypes.sizeof(_Mem))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                ram = m.total / 2 ** 30
        else:
            ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30
    except (ValueError, OSError, AttributeError):
        pass
    return max(1, min(os.cpu_count() or 1, int((ram - 6) / 1.5)))


_tl = threading.local()


def _init(path, gpu=False):
    """a worker's renderer: per process (CPU engine) or per thread (GPU engine: the threads of one process share the
    GPU, its buffer pool and the libraries; one prepares its frame while another waits for the GPU)"""
    global _film
    if gpu:                                        # the pixel work on the GPU (kadr_core, Rust + wgpu)
        from .gpu import GpuRenderer
        _tl.film = GpuRenderer(Film(path))
    else:
        _film = Renderer(Film(path))


def _worker_film():
    return getattr(_tl, 'film', None) or _film


def _rgba(r, t):
    """the frame over nothing — what the film draws, with its transparency: 8-bit RGBA, colour not multiplied by alpha (the
    film's own background is left out)"""
    r.f.background = None
    if hasattr(r, 'frame_rgba_start'):             # GPU: the division and the 8 bits on the card, a quarter of the bytes back
        return r.frame_rgba_start(t).get()
    if hasattr(r, 'frame_comp'):
        a = r.frame_comp(t).download()
        rgb, al = a[..., :3], a[..., 3:]
    else:
        rgb, al = r.frame_rgba(t)
    out = np.empty(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(np.round(np.where(al > 1e-5, rgb / np.maximum(al, 1e-5), 0)), 0, 255)
    out[..., 3:] = np.clip(np.round(al * 255), 0, 255)
    return out


ALPHA = {'.mov': ['-c:v', 'prores_ks', '-profile:v', '4444', '-pix_fmt', 'yuva444p10le', '-alpha_bits', '16', '-vendor', 'apl0', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709'],
         '.webm': ['-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-b:v', '0', '-crf', '24', '-row-mt', '1']}


class _Feed:
    """ffmpeg's stdin, written from a thread of its own: a write to the pipe waits while ffmpeg encodes, and without
    this the GPU sat idle for that time — drawing and encoding took turns (Windows, RTX 3070, ProRes 4444 1080×1920:
    ~32 ms to draw + ~45 ms to encode a frame, one after the other). A few frames are held, never more"""

    def __init__(self, pipe, depth=4):
        import queue
        self.pipe, self.q, self.err = pipe, queue.Queue(depth), None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while (b := self.q.get()) is not None:
            if self.err is None:
                try:
                    self.pipe.write(b)
                except Exception as e:                    # ffmpeg gone: said at close(), the drawing does not hang
                    self.err = e

    def write(self, b):
        if self.err is not None:
            raise self.err
        self.q.put(b)

    def close(self):
        self.q.put(None)
        self.t.join()
        self.pipe.close()
        if self.err is not None:
            raise self.err


def _block(job):
    """render frames [i0, i1) in order (footage is read forward, no re-seeking) and encode them as one H.264 piece"""
    i0, i1, out, crf, preset = job
    film = _worker_film()
    f = film.f
    if preset.startswith('alpha'):                 # over nothing: a piece with transparency (ProRes 4444, VP9) or PNG files
        kind = preset.split(':', 1)[1]
        if kind == '.png':
            for i in range(i0, i1):
                Image.fromarray(_rgba(film, i / f.fps + SAMPLE), 'RGBA').save(out % i)
            return i1 - i0
        ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{f.W}x{f.H}', '-r', f'{f.fps:g}', '-i', '-', *ALPHA[kind], out], stdin=subprocess.PIPE)
        if hasattr(film, 'frame_rgba_start'):      # frame i is drawn on the GPU while the CPU prepares frame i + 1
            feed, back = _Feed(ff.stdin), None    # and ffmpeg encodes the frames before them
            for i in range(i0, i1):
                nxt = film.frame_rgba_start(i / f.fps + SAMPLE)
                if back is not None:
                    feed.write(back.get().tobytes())
                back = nxt
            if back is not None:
                feed.write(back.get().tobytes())
            feed.close()
        else:
            for i in range(i0, i1):
                ff.stdin.write(_rgba(film, i / f.fps + SAMPLE).tobytes())
            ff.stdin.close()
        ff.wait()
        from .media import close_all
        close_all()
        return i1 - i0
    gpu = hasattr(film, 'frame_start')
    ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba' if gpu else 'rgb24', '-s', f'{f.W}x{f.H}', '-r', f'{f.fps:g}',
                           '-i', '-', *(hw_encoder()[0] if preset == 'hw' else ['-c:v', 'libx264', '-crf', str(crf), '-preset', preset]), '-bf', '0',
                           # RGB → YUV by the HD matrix, and say so in the file: ffmpeg's default is the SD matrix (BT.601) with no tag, and
                           # every player takes an untagged 1080p file for BT.709 — reds and greens came out a little off on a phone
                           '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
                           '-pix_fmt', 'yuv420p', out],
                          stdin=subprocess.PIPE)
    if gpu:                                        # frame i is drawn on the GPU while the CPU prepares frame i + 1
        feed, back = _Feed(ff.stdin), None        # and ffmpeg encodes the frames before them
        for i in range(i0, i1):
            nxt = film.frame_start(i / f.fps + SAMPLE)
            if back is not None:
                feed.write(back.get().tobytes())
            back = nxt
        if back is not None:
            feed.write(back.get().tobytes())
        feed.close()
    else:
        for i in range(i0, i1):
            ff.stdin.write(to_u8(film.frame(i / f.fps + SAMPLE)).tobytes())
        ff.stdin.close()
    ff.wait()
    from .media import close_all
    close_all()                                    # this worker's next block is elsewhere in the film: free its decoders
    return i1 - i0


def cmd_render(a, film):
    from .audio import mix, SR, LIMITER
    hw = hw_encoder() if a.hw and not a.alpha else None        # the hardware encoder of this machine, if Kadr knows one that works here
    if a.hw and not a.alpha and hw is None:                    # none: x264, said aloud
        print('kadr render --hw: аппаратного кодера, который Kadr знает, на этой машине нет — кодирую x264 (как добавить кодер своей видеокарты: AGENTS.md, «Windows and Linux»)',
              file=sys.stderr)
    t0 = time.time()
    if not a.fg:
        # polite by default (15.11.2026: a render made video and sound on the same Mac stutter «like a chewed tape»):
        # the render and every process it starts (workers, ffmpeg) run at nice 10, so the player, the sound and the UI
        # always come first (the screen's share of the GPU: small command buffers, core/src/lib.rs BATCH)
        try:
            os.nice(10)
        except (OSError, AttributeError):                    # (Windows has no nice)
            pass
    if a.jobs is None:                             # GPU: threads of one process (see _init); CPU: processes
        a.jobs = GPU_JOBS if a.gpu else max(1, default_jobs() - (0 if a.fg else 1))
    i0 = int(round(film.time(a.start) * film.fps)) if a.start is not None else 0
    i1 = int(round(film.time(a.end) * film.fps)) if a.end is not None else film.frames
    tmp = tempfile.mkdtemp(prefix='kadr_')
    ext = os.path.splitext(a.out)[1].lower()
    if a.alpha and ext not in ('.mov', '.webm', '.png'):
        print('kadr render --alpha: файл с прозрачностью — .mov (ProRes 4444, для монтажных программ), .webm (VP9, для веба) или кадры: папка/%05d.png', file=sys.stderr)
        return 1
    if a.alpha and ext == '.png':                  # a file a frame: out is a pattern with the frame's number in it
        if '%' not in a.out:
            print('kadr render --alpha: для кадров нужен шаблон с номером: -o папка/%05d.png', file=sys.stderr)
            return 1
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    piece = lambda b: a.out if (a.alpha and ext == '.png') else os.path.join(tmp, f'{b:07}{ext if a.alpha else ".mp4"}')
    jobs = [(b, min(b + BLOCK, i1), piece(b), a.crf, f'alpha:{ext}' if a.alpha else 'hw' if hw else 'veryfast' if a.draft else a.preset) for b in range(i0, i1, BLOCK)]
    done = 0
    os.environ['PYTHONSAFEPATH'] = '1'               # the workers: no module is ever taken from the current folder (a film's own folder may be a stranger's)
    with (ThreadPool if a.gpu and a.lite else Pool)(a.jobs, initializer=_init, initargs=(film.path, a.gpu)) as pool:
        for n in pool.imap(_block, jobs):
            done += n
            print(f'\r{done}/{i1 - i0} кадров', end='', file=sys.stderr, flush=True)
    if a.alpha and ext == '.png':
        shutil.rmtree(tmp, ignore_errors=True)
        print(f'\r{a.out}: {i1 - i0} кадров с прозрачностью за {time.time() - t0:.1f} с', file=sys.stderr)
        return 0
    open(os.path.join(tmp, 'list.txt'), 'w', encoding='utf-8').write(''.join(f"file '{j[2]}'\n" for j in jobs))
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', os.path.join(tmp, 'list.txt')]
    sound = mix(film, i0 / film.fps, i1 / film.fps) if (film.audio or any(L['type'] == 'video' for L in film.layers)) else None
    if sound is not None and sound.any():
        wav = os.path.join(tmp, 'mix.f32')
        sound.astype(np.float32).tofile(wav)
        cmd += ['-f', 'f32le', '-ac', '2', '-ar', str(SR), '-i', wav, '-map', '0:v', '-map', '1:a', '-filter:a', LIMITER + ',apad',
                '-c:a', 'libopus' if ext == '.webm' else 'aac', '-b:a', '256k', '-shortest']   # the sound padded with silence: the video sets the end (it is
                                                              # up to a frame longer than a duration kept to 3 decimals)
    cmd += ['-c:v', 'copy', *(['-tag:v', hw[1]] if hw else []), *([] if ext == '.webm' else ['-movflags', '+faststart']), a.out]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    dt = time.time() - t0
    print(f'\r{a.out}: {i1 - i0} кадров за {dt:.1f} с ({dt / max(1, i1 - i0) * 1000:.0f} мс/кадр, {a.jobs} {"потоков" if a.gpu and a.lite else "процессов"})', file=sys.stderr)


def cmd_srt(a, film):
    from .captions import phrases, srt
    cap = next((L for L in film.layers if L['type'] == 'captions'), None)
    ph = cap['phrases'] if cap else phrases(film.words)
    open(a.out, 'w', encoding='utf-8').write(srt(ph))
    print(f'{a.out}: {len(ph)} фраз')


def _renderer(a, film):
    if getattr(a, 'gpu', False):
        from .gpu import GpuRenderer
        return GpuRenderer(film)
    return Renderer(film)


def _still(r, t):
    """one frame as 8-bit RGB — on the GPU when the renderer has one"""
    return r.frame_u8(t) if hasattr(r, 'frame_u8') else to_u8(r.frame(t))


def cmd_frame(a, film):
    t = film.time(a.at)
    if a.alpha:                                    # over nothing, with its transparency
        Image.fromarray(_rgba(_renderer(a, film), t), 'RGBA').save(a.out)
    else:
        Image.fromarray(_still(_renderer(a, film), t)).save(a.out)
    print(f'{a.out}: {t:.2f} с')


def sheet_image(film, r, times, cols=4, tw=480):
    """a contact sheet: the frames at `times`, cols across, each tw px wide, captioned with its time and the word then"""
    th = round(tw * film.H / film.W)
    rows = (len(times) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tw, rows * (th + 30)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    fnt = ImageFont.truetype(font_file('Arial'), 18)
    for j, t in enumerate(times):
        x, y = (j % cols) * tw, (j // cols) * (th + 30)
        sheet.paste(Image.fromarray(_still(r, t)).resize((tw, th), Image.LANCZOS), (x, y + 30))
        w = film.word_near(t)
        d.text((x + 8, y + 5), f'{t:.2f} с' + (f' · «{w}»' if w else ''), font=fnt, fill=(255, 200, 100))
    return sheet


def cmd_sheet(a, film):
    times = [film.time(x) for x in a.at] if a.at else list(np.arange(0, film.duration, a.every))
    sheet_image(film, _renderer(a, film), times).save(a.out)
    print(f'{a.out}: {len(times)} кадров')


def cmd_catalog(a):
    """what an agent can write: shot kinds of a style (built-in, plugins, templates) with their parameters"""
    from . import plugins
    from .shots import KINDS, OWN, COMMON
    plugins.load(a.plugins or [])
    if a.style == 'figure':                                    # the block figure of a 3D scene: kadr/figure.py
        from . import figure as _fg
        print(_fg.__doc__)
        return 0
    names = [] if a.style == 'effects' else [a.style] if a.style else sorted(plugins.STYLES)
    for name in names:
        if name not in plugins.STYLES:
            print(f'нет стиля «{name}»; есть: {", ".join(sorted(plugins.STYLES))}')
            return 1
        cls = plugins.STYLES[name]
        import sys as _sys
        doc = cls.__doc__ or getattr(_sys.modules.get(cls.__module__), '__doc__', '') or ''
        print(f'\n== стиль {name}: {doc.strip().splitlines()[0] if doc.strip() else ""}')
        base = plugins.STYLES[name].__mro__[1]
        for k in KINDS:
            m = getattr(cls, k, None)
            own = k in ('video', 'omni', 'line') or (m is not None and m is not getattr(base, k, None) and 'unsupported' not in (m.__code__.co_names if hasattr(m, '__code__') else ()))
            if own:
                doc = (m.__doc__ or '').strip().split('\n')[0] if m is not None and k not in ('video', 'omni', 'line') else ''
                fields = {'chapter': {'n', 'title', 'kicker'}, 'title': {'text', 'sub', 'kicker'}}.get(k, OWN[k])
                print(f'  {k:14} {", ".join(sorted(fields)):60} {doc}')
        for (st, k), (fn, params) in sorted(plugins.ELEMENTS.items(), key=lambda x: x[0][1]):
            if st in (name, '*'):
                doc = (fn.__doc__ or '').strip().split('\n')[0]
                print(f'  {k:14} {", ".join(sorted(params)):60} {doc}{"  [любой стиль]" if st == "*" else ""}')
        for k in ('stamp', 'speaker', 'source'):
            m = getattr(cls, k, None)
            if m is not None and m is not getattr(base, k, None):
                print(f'  ({k})')
    if a.style in (None, 'effects'):
        print('\n== эффекты слоя: effects: [{type: имя, параметр: значение, …}] — параметры анимируются')
        for k, (fn, params, _) in sorted(plugins.EFFECTS.items()):
            sh = getattr(fn, 'shader', None)
            doc = sh.doc if sh is not None else (fn.__doc__ or '').strip().split('\n')[0]
            ps = ', '.join(f'{n}={("pivot" if v is None and sh is not None else v)!r}'.replace("'", '') for n, v in params.items())
            print(f'  {k:13} {doc}\n  {"":13} {ps}' + ('; map: слой — вторая картинка' if sh is not None and sh.takes_map else ''))
        print('  shader        свой шейдер: src: файл.wgsl рядом с фильмом или code: "fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {…}";\n'
              '                что в нём можно вызывать — kadr/shader.py (src_at, map_here, fx_time, fx_layer_xy, fx_px, fx_noise…)')
        print('\n== слои, которые рисует шейдер: {type: имя, size: [w, h], position, …} — двигаются, вращаются и наклоняются в 3D как любой слой')
        for k, (fn, params, kind) in sorted(plugins.LAYER_TYPES.items()):
            sh = getattr(fn, 'shader', None)
            if kind == 'gpu' and sh is not None:
                ps = ', '.join(f'{n}={("pivot" if v is None else v)!r}'.replace("'", '') for n, v in params.items())
                print(f'  {k:13} {sh.doc}\n  {"":13} {ps}')
        print('  shader        свой слой-шейдер: src: файл.wgsl или code: "…"; в нём fx_uv(xy) — место пикселя в рамке слоя 0..1, fx_box() — её размер')
        from . import scene3d as _s3
        print('\n== 3D-сцена (слой type: scene3d; рисует ядро на видеокарте)')
        print('\n'.join('  ' + ln for ln in (_s3.__doc__ or '').split('\nThe drawing itself')[0].split('\n')[2:]))
        from . import figure as _fg
        print('\n== блочная фигурка в 3D-сцене (shape: figure): персонаж из нескольких слов, с позами, лицом и ртом по голосу')
        print('\n'.join('  ' + ln for ln in (_fg.__doc__ or '').split('\n')[2:]))
        from . import psys
        print('\n== системы частиц (слои; рисуются в своей плоскости size и дальше живут как картинка)')
        print('\n'.join('  ' + ln for ln in (psys.__doc__ or '').split('\n\n')[1].split('\n')))
        print('\n'.join('  ' + ln for ln in (psys.__doc__ or '').split('\n\n')[2].split('\n')))
        from . import track as _trk
        print('\n== съёмка: слежение и стабилизация (команды kadr track, kadr stabilize; формула track(), свойство stabilize)')
        print('\n'.join('  ' + ln for ln in (_trk.__doc__ or '').split('\n\n', 1)[1].rsplit('\n\n', 1)[0].split('\n')))
        from . import matte as _mt
        print('\n== съёмка: вырезание человека или предмета (команда kadr matte; свойство cutout у слоёв video и image)')
        print('\n'.join('  ' + ln for ln in (_mt.__doc__ or '').split('\n\n', 1)[1].rsplit('\n\n', 1)[0].split('\n')))
        from . import reframe as _rf, chart as _ch, puppet as _pp, svg as _svg
        body = lambda m: '\n'.join('  ' + ln for ln in (m.__doc__ or '').split('\n\n', 1)[1].rsplit('\n\n', 1)[0].split('\n'))
        print('\n== съёмка: человек в узком кадре (команда kadr reframe; свойство reframe у слоя video)')
        print(body(_rf))
        print('\n== вектор из файла (слой svg)')
        print('\n'.join('  ' + ln for ln in (_svg.__doc__ or '').split('\n\n', 1)[1].split('\n')))
        print('\n== графики из чисел (слой chart)')
        print(body(_ch))
        print('\n== картинка на булавках (слой puppet, команда kadr puppet)')
        print(body(_pp))
        from . import sound as _snd
        print('\n== звук как источник (формулы amp(), band() и слои spectrum, wave)')
        print('\n'.join('  ' + ln for ln in (_snd.__doc__ or '').split('\n\n', 1)[1].rsplit('\n\n', 1)[0].split('\n')))
        print('\n== режимы наложения (blend): ' + ', '.join(sorted(__import__("kadr.doc", fromlist=["BLENDS"]).BLENDS)))
        print('   visible: false — слой не рисуется сам, но работает как карта (map), мат или родитель')
        if a.style == 'effects':
            return 0
    if plugins.TEMPLATES:
        print('\n== шаблоны (YAML)')
        for k, t in sorted(plugins.TEMPLATES.items()):
            print(f'  {k:14} {", ".join(sorted((t.get("params") or {}).keys()))}')
    print(f'\nУ любого кадра: {", ".join(sorted(COMMON))}.  Подробно — DESIGN.md.')
    return 0


def cmd_track(a):
    """follow a thing through a clip (kadr/track.py) → clip.track.json beside it"""
    from . import track
    try:
        quad, four = track.region(a.region)
        rep = track.track_clip(a.clip, a.name, a.at, quad, a.model or ('plane' if four else 'similar'), a.start, a.to, surface=track.region(a.surface)[0] if a.surface else None)
    except (ValueError, OSError) as e:
        print(f'kadr track: {e}', file=sys.stderr)
        return 1
    print(f"{rep['out']}: трек «{a.name}», кадров {rep['frames']} ({rep['first'] / rep['fps']:.2f}–{rep['last'] / rep['fps']:.2f} с), "
          f"потерян на {rep['lost']}, среднее сходство {rep['q']:.2f}")
    if rep['lost'] > rep['frames'] * .3:
        print('  потерян на трети кадров и больше: область без рисунка (гладкая стена, небо), слишком мала или уходит из кадра — назовите другую или отрезок --from … --to')
    if a.sheet:
        print('  лист кадров с найденным контуром:', track.sheet(a.clip, a.name, a.sheet))
    return 0


def cmd_stabilize(a):
    """the camera's path through a clip (kadr/track.py) → clip.stab.json beside it"""
    from . import track
    try:
        rep = track.stabilize_clip(a.clip, region=track.region(a.region)[0] if a.region else None)
    except (ValueError, OSError) as e:
        print(f'kadr stabilize: {e}', file=sys.stderr)
        return 1
    print(f"{rep['out']}: кадров {rep['frames']}, среднее сходство {rep['q']:.2f}; камера уходит до {rep['shift']:.0f} px, поворот до {rep['turn']:.1f}°, "
          f"размер {rep['size'][0]:.3f}–{rep['size'][1]:.3f}")
    print(f"  увеличение, чтобы не показать края: {rep['zoom_smooth']:.3f} при stabilize: true (плавный ход), {rep['zoom_lock']:.3f} при lock: true (штатив)")
    if rep['weak']:
        print(f"  на {rep['weak']} кадрах зацепиться было не за что (склейка, вспышка, гладкий кадр): там камера считается неподвижной")
    return 0


def cmd_matte(a):
    """a person or the main thing cut out of footage or photographs (kadr/matte.py) → file.matte.mp4 / .png beside it"""
    from . import matte
    if not a.files:
        print('kadr matte клип.mp4 | фото.jpg [ещё …] [--what person | subject | soft] [--point x,y] [--sheet лист.jpg]', file=sys.stderr)
        return 1
    try:
        point = [float(v) for v in a.point.replace(',', ' ').split()] if a.point else None
        for f in a.files:
            if not os.path.exists(f):
                raise RuntimeError(f'нет файла {f}')
            rep = matte.run(f, a.what, point, a.out if len(a.files) == 1 else None)
            print(matte.describe(rep))
            if a.sheet and len(a.files) == 1 and not a.out and rep['out']:
                print('  лист кадров:', matte.sheet(f, a.sheet))
    except (RuntimeError, ValueError, OSError) as e:
        print(f'kadr matte: {e}', file=sys.stderr)
        return 1
    return 0


def cmd_reframe(a):
    """where the people are through a clip (kadr/reframe.py) → clip.frame.json beside it"""
    from . import reframe
    if not os.path.exists(a.clip):
        print(f'kadr reframe: нет файла {a.clip}', file=sys.stderr)
        return 1
    try:
        rep = reframe.run(a.clip, a.what)
    except (RuntimeError, ValueError, OSError) as e:
        print(f'kadr reframe: {e}', file=sys.stderr)
        return 1
    print(f"{rep['out']}: планов {rep['shots']}; из {rep['samples']} просмотренных кадров {'человек' if a.what == 'person' else 'предмет'} найден на {rep['seen']}, лицо — на {rep['faces']}; "
          f"по ширине он ходит на {rep['wander']:.0f} px из {rep['size'][0]}")
    if rep['seen'] < rep['samples'] * .5:
        print('  найден меньше чем на половине кадров: там окно встанет посередине. Для предмета или зверя — --what subject')
    if a.sheet:
        print('  лист кадров с окном 9:16:', reframe.sheet(a.clip, a.sheet))
    return 0


def cmd_puppet(a):
    """a picture under a ruler with its mesh and the pins tried (kadr/puppet.py)"""
    from . import puppet
    if not os.path.exists(a.picture):
        print(f'kadr puppet: нет файла {a.picture}', file=sys.stderr)
        return 1
    try:
        pins = [tuple(float(v) for v in p.replace(',', ' ').split()) for p in (a.pins or [])]
        if any(len(p) != 2 for p in pins):
            raise ValueError('--pins: пары x,y через пробел — 330,250 210,60')
        found = {}
        if a.pose:
            found = puppet.joints(a.picture, a.cutout, a.mesh)
        cell, cells, off = puppet.sheet(a.picture, a.out, pins + list(found.values()), a.mesh, a.cutout, [''] * len(pins) + list(found))
    except (ValueError, OSError, RuntimeError) as e:
        print(f'kadr puppet: {e}', file=sys.stderr)
        return 1
    print(f'{a.out}: сетка с ячейкой {cell} px, ячеек {cells}' + (f'; булавок {len(pins)}' if pins else ''))
    for p in off:
        print(f'  булавка {p[0]:g},{p[1]:g} стоит на прозрачном месте — мимо сетки')
    if a.pose and not found:
        print('  человека на картинке не нашлось (рисованных фигур системный поиск не видит): булавки — по листу, вручную')
    elif a.pose:
        print('  суставы (l — тот, что на картинке левее); в документ:\n    pins:')
        for name, (x, y) in found.items():
            print(f'      {name}: {{at: [{x}, {y}]}}')
    return 0


def cmd_cuts(a):
    """where a clip jumps from one shot to another (kadr/track.py)"""
    from . import track
    if not os.path.exists(a.clip):
        print(f'kadr cuts: нет файла {a.clip}', file=sys.stderr)
        return 1
    try:
        cuts, fps, n = track.find_cuts(a.clip)
    except (ValueError, OSError) as e:
        print(f'kadr cuts: {e}', file=sys.stderr)
        return 1
    print(f'{a.clip}: {n} кадров, {n / fps:.2f} с; склеек {len(cuts)}' + (' — один план' if not cuts else ''))
    edges = [0.0] + cuts + [n / fps]
    for i, (t0, t1) in enumerate(zip(edges, edges[1:])):
        print(f'  план {i + 1}: {t0:.2f} – {t1:.2f} с ({t1 - t0:.2f} с)')
    return 0


def cmd_beats(a):
    """the pulse of a piece of music (kadr/sound.py): its tempo and the second of every beat"""
    from . import sound
    if not os.path.exists(a.file):
        print(f'kadr beats: нет файла {a.file}', file=sys.stderr)
        return 1
    try:
        tempo, bt = sound.find_beats(sound.track(os.path.abspath(a.file)))
    except ValueError as e:
        print(f'kadr beats: {e}', file=sys.stderr)
        return 1
    if len(bt) < 2:
        print(f'{a.file}: ровного бита не слышно')
        return 0
    print(f'{a.file}: {tempo:.1f} ударов в минуту, {len(bt)} битов (в документе: beat:1 … beat:{len(bt)})')
    print(' '.join(f'{t:.2f}' for t in bt))
    return 0


def cmd_scene(a):
    """a 3D scene from six sides, and what cannot be right in it (kadr/scenecheck.py)"""
    from . import scenecheck
    try:
        pic, words = scenecheck.look(a.scene, a.layer, a.at, a.node, a.tile, not a.no_check, a.whole)
    except DocError as e:
        print(e, file=sys.stderr)
        return 1
    pic.save(a.out, quality=90)
    print(f'{a.out}: шесть видов')
    print('\n'.join(words))
    return 0


def cmd_depth(a):
    """depth maps for the parallax effect (kadr/depth.py): made once beside the photo, read later as a picture"""
    from . import depth
    try:
        if a.download:
            print('модель на месте:', depth.download())
        if not a.files:
            return 0 if a.download else print('kadr depth фото.jpg [ещё.png …] [-o карта.png]', file=sys.stderr) or 1
        miss = [f for f in a.files if not os.path.exists(f)]
        if miss:
            print('нет файла: ' + ', '.join(miss), file=sys.stderr)
            return 1
        regions = [depth.region(t) for t in a.smooth or []]
        if regions and len(a.files) != 1:
            print('--smooth задаётся для одного фото: области — в его пикселях', file=sys.stderr)
            return 1
        rep = depth.run(a.files, [a.out] if a.out and len(a.files) == 1 else None, edge=a.edge, regions=regions)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    for m in rep['maps']:
        print(f"{m['out']}  {m['size'][0]}×{m['size'][1]}  {m['seconds']} с")
        for r, share in zip(regions, m.get('agreed') or []):
            print(f"  область от ({r['pts'][0][0]:g}, {r['pts'][0][1]:g}): гладкая поверхность найдена по {share * 100:.0f} % её площади"
                  + ('' if share >= 0.3 else ' — мало: область, похоже, шире самой поверхности'))
    print(f"модель загрузилась за {rep['load_seconds']} с, считала на {rep['device']}", file=sys.stderr)
    return 0


def cmd_check(a, film):
    from .check import run, report
    issues = run(film, strict=a.strict)
    print(report(issues))
    return 1 if any(x.level == 'error' for x in issues) else 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):                    # Kadr speaks Russian; a console or a pipe in another code page must not break on it
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog='kadr', description='Видеодвижок для нейросети: документ фильма → видео без дрожи + отчёт о браке.')
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('render', help='документ → mp4')
    r.add_argument('film'); r.add_argument('-o', '--out', default='film.mp4')
    r.add_argument('--from', dest='start'); r.add_argument('--to', dest='end')
    r.add_argument('--alpha', action='store_true', help='без фона фильма, с прозрачностью — графика, которую кладут поверх чужого монтажа: -o x.mov (ProRes 4444), x.webm (VP9) или папка/%%05d.png')
    r.add_argument('--jobs', type=int, default=None); r.add_argument('--crf', type=int, default=18)
    r.add_argument('--preset', default='medium', help='x264: medium для сдачи; fast на 16 %% дешевле почти без потерь')
    r.add_argument('--hw', action='store_true', help='аппаратный кодер этой машины (на Mac — HEVC, q 85: качество как у x264 medium и ровнее от кадра к кадру, кодирование почти бесплатно, файл вдвое больше); где Kadr такого не знает — x264')
    r.add_argument('--draft', action='store_true', help='черновик: x264 veryfast, кодирование втрое дешевле (PSNR 45,8 против 48)')
    r.add_argument('--gpu', action='store_true', help='пиксели на видеокарте (kadr_core)')
    r.add_argument('--fg', action='store_true', help='обычный приоритет процессора вместо nice 10 (по умолчанию рендер уступает плееру и звуку)')
    r.add_argument('--lite', action='store_true', help='с --gpu: один процесс в 2 потока — на треть меньше памяти, примерно на 20 %% медленнее')
    f = sub.add_parser('frame', help='один кадр в PNG')
    f.add_argument('film'); f.add_argument('--at', required=True); f.add_argument('-o', '--out', default='frame.png')
    f.add_argument('--gpu', action='store_true', help='на видеокарте (kadr_core)')
    f.add_argument('--alpha', action='store_true', help='без фона фильма, с прозрачностью (PNG)')
    s = sub.add_parser('sheet', help='лист раскадровки')
    s.add_argument('film'); s.add_argument('-o', '--out', default='sheet.png')
    s.add_argument('--every', type=float, default=1.0); s.add_argument('--at', nargs='*')
    s.add_argument('--gpu', action='store_true', help='на видеокарте (kadr_core)')
    c = sub.add_parser('check', help='отчёт о браке')
    c.add_argument('film')
    c.add_argument('--strict', action='store_true', help='судить и слои, нарисованные стилем канала (для автора стиля)')
    t = sub.add_parser('srt', help='подписи в .srt (для загрузки на YouTube)')
    t.add_argument('film'); t.add_argument('-o', '--out', default='film.srt')
    sub.add_parser('selftest', help='проверить гарантию «без дрожи» на этой машине')
    sub.add_parser('serve', help='MCP-сервер для агента (stdio): кадры с тёплой видеокарты за доли секунды')
    g = sub.add_parser('catalog', help='что можно писать: виды кадров стиля и их параметры; `kadr catalog effects` — эффекты слоя; `kadr catalog figure` — блочная фигурка 3D-сцены')
    g.add_argument('style', nargs='?'); g.add_argument('--plugins', nargs='*')
    tr_ = sub.add_parser('trust', help='разрешить фильмам из этой папки (и вложенных) подключать свой код — plugins; без папки: показать список')
    tr_.add_argument('folder', nargs='?')
    sc = sub.add_parser('scene', help='3D-сцена с шести сторон в ровном свете и её проверка: что проходит сквозь что, что спрятано, что висит в воздухе')
    sc.add_argument('scene', help='файл сцены (.yaml словами Kadr или выгруженный .json) либо фильм со слоем scene3d')
    sc.add_argument('-o', '--out', default='scene.jpg')
    sc.add_argument('--layer', help='в фильме: id слоя scene3d (по умолчанию первый)')
    sc.add_argument('--at', default='0', help='в фильме: момент, в позе которого показать (секунды или word:…)')
    sc.add_argument('--node', help='показать и проверить только эту часть сцены со всем, что на ней висит (её id)')
    sc.add_argument('--tile', type=int, default=480, help='сторона одного вида, px')
    sc.add_argument('--whole', action='store_true', help='кадрировать по всей сцене, а не по основной массе тел (стена и пол целиком)')
    sc.add_argument('--no-check', action='store_true', help='только лист, без проверок')
    d = sub.add_parser('depth', help='карта глубины фотографии (белое — близко) для эффекта parallax: фото.depth.png рядом с фото')
    d.add_argument('files', nargs='*'); d.add_argument('-o', '--out', help='куда сохранить (для одного фото)')
    d.add_argument('--smooth', action='append', metavar='"x0,y0,x1,y1" | "x,y x,y x,y …"',
                   help='гладкая поверхность фото (грань коробки, постер, экран, стенка банки), px: глубина в ней станет гладкой, рисунок на ней не будет выпирать; можно несколько раз')
    d.add_argument('--edge', type=float, default=4.0, help='на сколько расширить близкое за кромку предмета, тысячные ширины фото (0 — карта как есть)')
    d.add_argument('--download', action='store_true', help='скачать модель Depth Anything V2 Small (99 МБ, Apache-2.0) с Hugging Face')
    tr = sub.add_parser('track', help='проследить предмет в клипе: клип.track.json рядом с клипом, в документе — формула track("слой", "имя")')
    tr.add_argument('clip'); tr.add_argument('--name', required=True, help='имя трека')
    tr.add_argument('--at', type=float, required=True, help='секунда клипа, на которой названа область')
    tr.add_argument('--region', required=True, metavar='"x0,y0,x1,y1" | "x,y x,y x,y x,y"', help='прямоугольник или четыре угла (левый верхний, правый верхний, правый нижний, левый нижний), px клипа')
    tr.add_argument('--model', choices=['point', 'similar', 'affine', 'plane'], help='point — место; similar — место, размер, поворот; plane — плоскость под любым углом (для четырёх углов по умолчанию)')
    tr.add_argument('--from', dest='start', type=float, help='с какой секунды клипа'); tr.add_argument('--to', type=float, help='по какую')
    tr.add_argument('--surface', metavar='"x,y x,y x,y x,y"', help='четыре угла, на которые ляжет прибитый слой (corner_pin), если это не сама область: область — экран вместе с рамкой, поверхность — стекло внутри')
    tr.add_argument('--sheet', help='лист кадров с найденным контуром, .jpg — посмотреть, что трек держится')
    sb = sub.add_parser('stabilize', help='путь камеры в клипе: клип.stab.json рядом с клипом, в документе — stabilize: true у видео-слоя')
    sb.add_argument('clip')
    sb.add_argument('--region', metavar='"x0,y0,x1,y1"', help='судить о камере только по этой части первого кадра (берег, стена), px клипа: для сцены, где половина кадра движется сама')
    mt = sub.add_parser('matte', help='вырезать человека (или главный предмет) из клипа или фото: файл.matte.mp4 / .png рядом, в документе — cutout: true у слоя')
    mt.add_argument('files', nargs='*'); mt.add_argument('-o', '--out', help='куда сохранить (для одного файла)')
    mt.add_argument('--what', default='person', choices=['person', 'subject', 'soft'], help='person — люди (по умолчанию); subject — всё, что выделяется (товар, зверь, предмет в руке); soft — люди с мягким краем (волосы, смаз)')
    mt.add_argument('--point', metavar='x,y', help='с --what subject: только предмет под этой точкой первого кадра, px клипа')
    mt.add_argument('--sheet', help='лист кадров с вырезанным на зелёном, .jpg')
    rfp = sub.add_parser('reframe', help='где в клипе человек: клип.frame.json рядом, в документе — reframe: true у видео-слоя (горизонтальное видео в вертикальный кадр)')
    rfp.add_argument('clip'); rfp.add_argument('--what', default='person', choices=['person', 'subject'], help='person — люди и лица; subject — то, что выделяется (товар, зверь)')
    rfp.add_argument('--sheet', help='лист кадров с окном 9:16, .jpg')
    pp = sub.add_parser('puppet', help='картинка под линейкой с сеткой и булавками — посмотреть, куда их ставить (слой type: puppet гнёт картинку за булавки)')
    pp.add_argument('picture'); pp.add_argument('-o', '--out', required=True, help='лист, .jpg или .png')
    pp.add_argument('--pins', nargs='*', help='булавки на пробу: 330,250 210,60'); pp.add_argument('--mesh', type=int, help='ячейка сетки, px')
    pp.add_argument('--cutout', action='store_true', help='картинка — фотография с вырезкой рядом (kadr matte)')
    pp.add_argument('--pose', action='store_true', help='найти суставы человека на фото и напечатать их булавками (на macOS — встроено; на других системах нужна своя программа вырезания, AGENTS.md)')
    ct = sub.add_parser('cuts', help='склейки в клипе: с какой секунды начинается каждый план')
    ct.add_argument('clip')
    bt = sub.add_parser('beats', help='пульс музыки: темп и секунда каждого бита (в документе время пишется как beat:5)')
    bt.add_argument('file')
    a = p.parse_args(argv)
    if a.cmd == 'beats':
        return cmd_beats(a)
    if a.cmd == 'cuts':
        return cmd_cuts(a)
    if a.cmd == 'reframe':
        return cmd_reframe(a)
    if a.cmd == 'puppet':
        return cmd_puppet(a)
    if a.cmd == 'matte':
        return cmd_matte(a)
    if a.cmd == 'track':
        return cmd_track(a)
    if a.cmd == 'stabilize':
        return cmd_stabilize(a)
    if a.cmd == 'catalog':
        return cmd_catalog(a)
    if a.cmd == 'trust':
        from . import plugins
        if a.folder:
            try:
                print(f'доверено: {plugins.trust(a.folder)} — фильмы оттуда могут подключать свой код (plugins)')
            except ValueError as e:
                print(e)
                return 1
        else:
            roots = plugins.trust_roots()
            print('\n'.join(roots) if roots else 'доверенных папок нет: код из plugins фильма не запускается (стили самого Kadr и ~/.kadr/plugins работают всегда)')
        return 0
    if a.cmd == 'scene':
        return cmd_scene(a)
    if a.cmd == 'depth':
        return cmd_depth(a)
    if a.cmd == 'serve':
        from .serve import main as serve
        return serve()
    if a.cmd == 'selftest':
        from .selftest import run
        return run()
    try:
        film = Film(a.film)
    except DocError as e:
        print(e, file=sys.stderr)
        return 1
    return {'render': cmd_render, 'frame': cmd_frame, 'sheet': cmd_sheet, 'check': cmd_check, 'srt': cmd_srt}[a.cmd](a, film) or 0


if __name__ == '__main__':
    sys.exit(main())
