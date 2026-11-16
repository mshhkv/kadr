"""Footage: probing and frame-exact sequential reading through ffmpeg (no extra dependencies)."""
import functools, json, math, queue, subprocess, threading
import numpy as np


@functools.lru_cache(None)
def probe(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-show_streams', '-show_format', '-of', 'json', path], capture_output=True, text=True, encoding='utf-8', errors='replace')
    d = json.loads(r.stdout or '{}')
    v = next((s for s in d.get('streams', []) if s['codec_type'] == 'video'), None)
    if v is None:
        raise ValueError(f'в {path} нет видеодорожки')
    num, den = map(int, v.get('avg_frame_rate', v['r_frame_rate']).split('/'))
    if not num:
        num, den = map(int, v['r_frame_rate'].split('/'))
    dur = float(v.get('duration') or d['format']['duration'])
    w, h = int(v['width']), int(v['height'])
    # a phone films upright into a frame stored on its side, with a note to turn it; ffmpeg turns it when it reads, so the
    # frames that come are h × w of what is stored. Without this a portrait clip was read as a sheared mess, silently
    turn = next((float(sd['rotation']) for sd in v.get('side_data_list') or [] if 'rotation' in sd), float((v.get('tags') or {}).get('rotate', 0) or 0))
    if round(abs(turn)) % 180 == 90:
        w, h = h, w
    return dict(w=w, h=h, fps=num / den, duration=dur,
                audio=any(s['codec_type'] == 'audio' for s in d.get('streams', [])))


class VideoReader:
    """Frame shown at source time t = frame floor(t·fps). Reads forward without re-seeking; seeks (frame-exact) only when asked to jump."""

    def __init__(self, path):
        self.path = path
        p = probe(path)
        self.w, self.h, self.fps = p['w'], p['h'], p['fps']
        self.last = max(0, int(p['duration'] * self.fps) - 1)
        self.proc, self.next, self.frame, self.idx = None, None, None, None

    def _start(self, n):
        self.close()
        ss = max(0.0, (n - 0.5) / self.fps)                  # frames with pts ≥ ss → the first one out is frame n
        self.proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{ss:.6f}', '-protocol_whitelist', 'file,pipe', '-i', self.path, '-an', '-f', 'rawvideo',
                                      '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     bufsize=self.w * self.h * 3 * 2)
        self.next = n
        # a reader thread keeps up to AHEAD frames waiting: the pipe holds only 64 KB, so without it ffmpeg stalls until
        # each frame is asked for and decoding never overlaps the drawing (the read lets go of Python's lock)
        self.q = queue.Queue(AHEAD)
        self.thread = threading.Thread(target=self._pump, args=(self.proc, self.q), daemon=True)
        self.thread.start()

    def _pump(self, proc, q):
        size = self.w * self.h * 3
        while True:
            try:
                buf = proc.stdout.read(size)
            except (ValueError, OSError):                     # closed under us (a jump): stop
                buf = b''
            q.put(np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3) if len(buf) == size else None)
            if len(buf) < size:
                return

    def _read(self):
        f = self.q.get()
        if f is None:
            self.q.put(None)                                   # stays at the end for any later read
            return None
        self.next += 1
        return f

    def at(self, t):
        n = min(self.last, max(0, int(math.floor(t * self.fps + 1e-4))))
        if self.idx == n:
            return self.frame
        if self.proc is None or n < self.next or n > self.next + 2 * self.fps:
            self._start(n)
        f = self.frame
        while self.next <= n:
            g = self._read()
            if g is None:                                     # ran out early (duration rounding): hold the last frame
                break
            f = g
        self.frame, self.idx = f, n
        return f

    def pair(self, t):
        """the two source frames around time t and how far t is from the first to the second (frame blending); the
        second is None at a frame's own moment and at the clip's end. Read forward: the earlier one is kept"""
        x = t * self.fps
        n = min(self.last, max(0, int(math.floor(x + 1e-4))))
        w = min(1.0, max(0.0, x - n))
        if n >= self.last or w < 1e-3:
            return self.at(t), None, 0.0
        if not (self.idx == n + 1 and getattr(self, '_prev_idx', None) == n):
            a = self.at((n + 0.5) / self.fps)
            b = self.at((n + 1.5) / self.fps)
            self._prev, self._prev_idx = a, n
        return self._prev, self.frame, w

    def close(self):
        if self.proc:
            self.proc.kill()
            while True:                                        # let the reader thread finish (it may wait on a full queue)
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    if not self.thread.is_alive():
                        break
                    self.thread.join(0.01)
            self.proc.stdout.close()
            self.proc.wait()
            self.proc = None


AHEAD = 2               # decoded frames waiting per open clip (6 MB each at 1080p, 25 MB at 4K)

_local = threading.local()   # each render thread its own decoders: two threads in one clip would seek each other back
OPEN = 3                # decoders kept open per thread: each is an ffmpeg (≈ 100 MB) — a whole film has hundreds of clips


def _readers():
    d = getattr(_local, 'readers', None)
    if d is None:
        d = _local.readers = {}
    return d


def reader(path):
    """the reader of a clip, most recently used last; the oldest is closed once more than OPEN are open"""
    rs = _readers()
    r = rs.pop(path, None) or VideoReader(path)
    rs[path] = r
    while len(rs) > OPEN:
        rs.pop(next(iter(rs))).close()
    return r


def close_all():
    rs = _readers()
    while rs:
        rs.popitem()[1].close()
