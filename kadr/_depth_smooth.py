"""A smooth surface in a depth map (kadr depth --smooth): numpy and Pillow only, so it is tested without the model."""
import numpy as np
from PIL import Image


def smooth(n, pts, order, rng):
    """the depth inside the polygon `pts` replaced by the smooth surface most of it lies on (order 2: flat or gently
    curved — a box's face, a poster, a screen, a can; the model itself does not hold a flat face flat, so a strict
    plane left a step at the region's edge). A picture printed on a flat thing reads as
    relief to the model; the relief is the minority there, so the surface is found the way outliers are thrown out:
    many surfaces through a few random points, the one most pixels agree with (within 2 % of the depth range), then
    fitted to all of those. The thing's own rim (the region's edge, less the background it spills over) counts three
    times: a surface reaches its edges, a print rarely covers them all — without it a face whose lower half is under a printed bowl was continued from its top half
    and tilted away. What lies far behind the surface and reaches the region's edge (the background past the thing's
    edge, if the region spills over it) is left alone. → the share of the region that agreed"""
    from PIL import ImageDraw, ImageFilter
    h, w = n.shape
    m = Image.new('L', (w, h), 0)
    ImageDraw.Draw(m).polygon([tuple(q) for q in pts], fill=255)
    mask = np.asarray(m) > 0
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return 0.0
    cx, cy, sx, sy = xs.mean(), ys.mean(), max(1.0, np.ptp(xs) / 2), max(1.0, np.ptp(ys) / 2)

    def design(x, y):
        u, v = (x - cx) / sx, (y - cy) / sy
        cols = [np.ones_like(u), u, v] + ([u * u, u * v, v * v] if order == 2 else [])
        return np.stack(cols, -1)
    def agree(A, z, wgt, tries):
        """the most popular surface: its coefficients and the points it passes through (within 2 % of the depth
        range). A surface through a few random points is a rough guess — six points out of a half-covered face are
        all on the face once in fifty tries, and even then they pin it loosely, so the answer changed with the dice
        (the same box's lower edge came out anywhere from 129 to 163 of 255). So the best few guesses are each
        polished — fitted to everything that agrees, then asked again who agrees, until that settles — and the
        winner is chosen among the polished ones."""
        k, found = A.shape[1], []
        for _ in range(tries):
            i = rng.choice(len(z), k, replace=False)
            c, *_ = np.linalg.lstsq(A[i], z[i], rcond=None)
            found.append((float(((np.abs(A @ c - z) < 0.02) * wgt).sum()), c))
        found.sort(key=lambda q: -q[0])
        best, bc, score = None, None, -1.0
        for _, c in found[:8]:
            ok = np.abs(A @ c - z) < 0.02
            for _ in range(12):
                if ok.sum() < k:
                    break
                c, *_ = np.linalg.lstsq(A[ok] * wgt[ok, None], z[ok] * wgt[ok], rcond=None)
                ok2 = np.abs(A @ c - z) < 0.02
                if (ok2 == ok).all():
                    break
                ok = ok2
            sc = float((ok * wgt).sum())
            if sc > score:
                best, bc, score = ok, c, sc
        return bc, best

    def patches(far):
        """far-behind pixels that are background: those the region's own edge reaches, with all that hangs on them
        (the region spilling over the thing's edge). A far-behind island inside is not background — the region was
        named as one surface, so that is a print the model read as a recess (the inside of a printed bowl left a
        hole in a can)"""
        fa = np.zeros((h, w), bool)
        fa[ys, xs] = far
        seed = fa & mask & ~(np.asarray(m.filter(ImageFilter.MinFilter(3))) > 0)
        for _ in range(max(h, w)):                             # grow the seeds through the far pixels they touch
            g = (np.asarray(Image.fromarray(seed.astype(np.uint8) * 255, 'L').filter(ImageFilter.MaxFilter(9))) > 0) & fa
            if (g == seed).all():
                break
            seed = g
        return seed

    pick = rng.choice(len(xs), min(len(xs), 5000), replace=False)
    was = n[ys, xs].astype(float)
    # first the surface most of the region lies on, with no thought of the rim: it tells the thing from the background
    # the region spills over (the region's own edge may run wholly outside the thing — then «the rim» was all
    # background, counted three times, and the background won)
    A0, z0 = design(xs[pick].astype(float), ys[pick].astype(float)), was[pick]
    c0, _ = agree(A0, z0, np.ones(len(pick)), 300)
    back = patches(was < design(xs.astype(float), ys.astype(float)) @ c0 - 0.2)
    thing = mask & ~back
    r = max(3, int(round(0.02 * max(np.ptp(xs), np.ptp(ys)))))
    e = max(2, r // 2)                                         # next to the background the model blurs the edge:
    t = Image.fromarray(thing.astype(np.uint8) * 255, 'L')     # the rim keeps that far from it, and only from it
    near = np.asarray(Image.fromarray(back.astype(np.uint8) * 255, 'L').filter(ImageFilter.MaxFilter(2 * e + 1))) > 0
    ry, rx = np.nonzero(thing & ~near & ~(np.asarray(t.filter(ImageFilter.MinFilter(2 * r + 1))) > 0))   # the thing's own rim
    rpick = rng.choice(len(rx), min(len(rx), 2500), replace=False) if len(rx) else np.array([], int)
    X = np.concatenate([xs[pick], rx[rpick]]).astype(float)
    Y = np.concatenate([ys[pick], ry[rpick]]).astype(float)
    A, z = design(X, Y), n[Y.astype(int), X.astype(int)].astype(float)
    wgt = np.concatenate([np.ones(len(pick)), np.full(len(rpick), 3.0)])
    c, best = agree(A, z, wgt, 600)
    fit = design(xs.astype(float), ys.astype(float)) @ c
    # the surface must meet what surrounds it: what the rim really is, less the surface there, is carried inwards as a
    # smooth correction (rim pixels the print reaches — off by more than 10 % — say nothing and are filled from the rest)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    res, val = np.zeros((y1 - y0, x1 - x0), np.float32), np.zeros((y1 - y0, x1 - x0), np.float32)
    rfit = design(rx.astype(float), ry.astype(float)) @ c
    good = np.abs(n[ry, rx] - rfit) < 0.1
    res[ry[good] - y0, rx[good] - x0] = (n[ry, rx] - rfit)[good]
    val[ry[good] - y0, rx[good] - x0] = 1.0
    if val.sum() > 20:
        small = (max(8, min(96, x1 - x0)), max(8, min(96, y1 - y0)))
        a_, b_ = (np.asarray(Image.fromarray(v, 'F').resize(small, Image.BOX), np.float32) for v in (res, val))
        g = np.exp(-0.5 * (np.arange(-48, 49) / 12.0) ** 2)

        def blur(v):
            v = np.apply_along_axis(lambda q: np.convolve(q, g, 'same'), 0, v)
            return np.apply_along_axis(lambda q: np.convolve(q, g, 'same'), 1, v)
        corr = blur(a_) / np.maximum(blur(b_), 1e-6)
        corr = np.asarray(Image.fromarray(corr.astype(np.float32), 'F').resize((x1 - x0, y1 - y0), Image.BICUBIC), np.float32)
        fit = fit + corr[ys - y0, xs - x0]
    fit = np.clip(fit, 0, 1)
    far = patches(was < fit - 0.2)                             # far behind the surface: background, not print
    n[ys, xs] = np.where(far[ys, xs], was, fit)
    return float(best[:len(pick)].mean())
