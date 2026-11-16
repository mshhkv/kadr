"""Captions from the voice-over words: phrases (by punctuation, pauses and length), line breaks, display times, SRT."""

END = ('.', '!', '?', '…', ':', ';')
PAUSE = 0.45            # s of silence that starts a new phrase
MIN_SHOW = 0.8          # s a phrase stays at least (unless the next one comes first)


def phrases(words, max_chars=42, lines=2, t0=0.0, t1=1e9):
    """words: [(norm, start, end, raw)] → [dict(text, words=[(raw, start, end)], start, end)]; text already broken into lines"""
    out, cur = [], []
    limit = max_chars * lines
    for i, (_, s, e, raw) in enumerate(words):
        if not (t0 <= s < t1):
            continue
        if cur:
            prev = cur[-1]
            joined = len(' '.join(w for w, _, _ in cur)) + 1 + len(raw)
            if prev[0].endswith(END) or s - prev[2] > PAUSE or joined > limit:
                out.append(cur)
                cur = []
        cur.append((raw, s, e))
    if cur:
        out.append(cur)
    res = []
    for j, ws in enumerate(out):
        start, end = ws[0][1], ws[-1][2]
        nxt = out[j + 1][0][1] if j + 1 < len(out) else 1e9
        end = min(max(end + 0.15, start + MIN_SHOW), nxt)
        res.append(dict(words=ws, text=break_lines([w for w, _, _ in ws], max_chars, lines), start=start, end=end))
    return res


def break_lines(ws, max_chars, lines):
    """one line if it fits, else the most balanced break (lines ≤ 2 here; longer phrases were already split)"""
    text = ' '.join(ws)
    if len(text) <= max_chars or lines < 2 or len(ws) < 2:
        return text
    best, cut = None, 1
    for k in range(1, len(ws)):
        a, b = len(' '.join(ws[:k])), len(' '.join(ws[k:]))
        score = max(a, b) + (0 if a <= max_chars and b <= max_chars else 1000)
        if best is None or score < best:
            best, cut = score, k
    return ' '.join(ws[:cut]) + '\n' + ' '.join(ws[cut:])


def srt(ph):
    def ts(t):
        ms = int(round(t * 1000))
        return f'{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}'
    return '\n'.join(f'{i}\n{ts(p["start"])} --> {ts(p["end"])}\n{p["text"]}\n' for i, p in enumerate(ph, 1))
