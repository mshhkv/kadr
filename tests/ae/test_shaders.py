"""Known answers for the built-in shader effects (kadr/shaders): what must not change does not, what must move moves
by exactly that. python tests/ae/test_shaders.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import kadr_core as kc
from kadr import shader
from kadr.render import Fx

SH = shader.builtin()
rng = np.random.default_rng(7)
H, W = 48, 64
pic = np.zeros((H, W, 4), np.float32)
pic[8:40, 12:52, :3] = rng.random((32, 40, 3)) * 255
pic[8:40, 12:52, 3] = 1.0
bad = []


def run(name, pv, im=pic, mp=None, scale=1.0, t=0.0):
    fx = Fx(scale, t, t, 1, (0, 0), None)
    fx.frame, fx.map = (im.shape[1], im.shape[0]), (kc.upload(np.ascontiguousarray(mp, np.float32)) if mp is not None else None)
    out, g = shader.run(SH[name], kc.upload(np.ascontiguousarray(im)), pv, fx)
    return out.download(), g


def pm(a):
    return np.concatenate([a[..., :3] * a[..., 3:], a[..., 3:] * 255], -1)


def check(what, got, want, tol=0.3):
    d = float(np.abs(pm(got) - pm(want)).max())
    print(f'{"OK  " if d <= tol else "FAIL"} {what}: max {d:.3f} (допуск {tol})')
    if d > tol:
        bad.append(what)


def grown(a, g):
    return np.pad(a, ((g, g), (g, g), (0, 0)))


def shifted(a, dx, dy, g):
    out = np.zeros_like(grown(a, g))
    out[g + dy:g + dy + a.shape[0], g + dx:g + dx + a.shape[1]] = a
    return out


# ---- nothing asked, nothing changes
for name, pv in (('displace', {'amount': (0.0, 0.0)}), ('wave', {'amplitude': 0.0}), ('ripple', {'amplitude': 0.0}),
                 ('bulge', {'strength': 0.0}), ('rgb_split', {'amount': 0.0}), ('glitch', {'amount': 0.0, 'split': 0.0, 'lines': 0.0}),
                 ('radial_blur', {'zoom': 0.0, 'spin': 0.0}), ('alpha_levels', {'low': 0.0, 'high': 1.0}), ('curves', {})):
    got, g = run(name, pv)
    check(f'{name}: без силы картинка та же', got, grown(pic, g))

# ---- displace by a map: red 1 → +amount px along x, grey 0.5 → stays
mp = np.zeros((H, W, 4), np.float32); mp[...] = [255, 127.5, 0, 1]
got, g = run('displace', {'amount': (10.0, 6.0)}, mp=mp)
check('displace: красная карта сдвигает ровно на amount по x', got, shifted(pic, 10, 0, g))
mp[...] = [127.5, 127.5, 0, 1]
got, g = run('displace', {'amount': (10.0, 6.0)}, mp=mp)
check('displace: серая карта 0.5 ничего не двигает', got, grown(pic, g))
got2, g2 = run('displace', {'amount': (10.0, 6.0)}, mp=mp, scale=2.0)
print('OK  ' if g2 == 2 * g else 'FAIL', f'displace: запас растёт с масштабом ({g} → {g2})')
if g2 != 2 * g:
    bad.append('grow scale')

# ---- rgb split: red right by amount, blue left, green stays
got, g = run('rgb_split', {'amount': 4.0, 'angle': 0.0})
want = np.zeros_like(grown(pic, g))
for ch, dx in ((0, 4), (1, 0), (2, -4)):
    want[..., ch] = shifted(pic, dx, 0, g)[..., ch] * shifted(pic, dx, 0, g)[..., 3]
al = np.maximum.reduce([shifted(pic, d, 0, g)[..., 3] for d in (4, 0, -4)])
gp = pm(got)
d = float(np.abs(gp[..., :3] - want[..., :3]).max()); d2 = float(np.abs(got[..., 3] - al).max())
print(f'{"OK  " if d <= 0.3 and d2 < 1e-3 else "FAIL"} rgb_split: каналы разъехались ровно на amount: цвет {d:.3f}, альфа {d2:.4f}')
if d > 0.3 or d2 >= 1e-3:
    bad.append('rgb_split')

# ---- mosaic: one colour inside a cell
flat = np.zeros((H, W, 4), np.float32); flat[..., :3] = rng.random((H, W, 3)) * 255; flat[..., 3] = 1
got, g = run('mosaic', {'cell': 8.0}, im=flat)
cells = got[8:40, 8:56].reshape(4, 8, 6, 8, 4)
d = float(np.abs(cells - cells[:, :1, :, :1]).max())
print(f'{"OK  " if d <= 0.3 else "FAIL"} mosaic: внутри клетки один цвет: {d:.3f}')
if d > 0.3:
    bad.append('mosaic')

# ---- alpha levels: half stays half, below low nothing, above high solid
ramp = np.zeros((4, 64, 4), np.float32); ramp[..., :3] = 200; ramp[..., 3] = np.linspace(0, 1, 64)[None, :]
got, g = run('alpha_levels', {'low': 0.4, 'high': 0.6}, im=ramp)
a = got[1, :, 3]; x = np.linspace(0, 1, 64)
ok = a[x < 0.39].max() < 1e-3 and a[x > 0.61].min() > 0.999 and abs(np.interp(0.5, x, a) - 0.5) < 0.02 and (np.diff(a) >= -1e-4).all()
print('OK  ' if ok else 'FAIL', 'alpha_levels: ниже low пусто, выше high плотно, середина 0.5, монотонно')
if not ok:
    bad.append('alpha_levels')

# ---- curves: a flat master makes one grey
got, g = run('curves', {'master': (0.5, 0.5, 0.5), 'black': 0.5, 'white': 0.5})
inside = got[8:40, 12:52, :3]
d = float(np.abs(inside - 127.5).max())
print(f'{"OK  " if d <= 0.5 else "FAIL"} curves: плоская кривая даёт один серый: {d:.3f}')
if d > 0.5:
    bad.append('curves')

# ---- wave: a whole wavelength of shift cancels over a period; the picture keeps its mass
got, g = run('wave', {'amplitude': 5.0, 'wavelength': 16.0, 'angle': 0.0, 'speed': 0.0})
m0, m1 = float(pic[..., 3].sum()), float(got[..., 3].sum())
print(f'{"OK  " if abs(m0 - m1) / m0 < 0.02 else "FAIL"} wave: площадь картинки сохранилась ({m0:.0f} → {m1:.0f}), запас {g}')
if abs(m0 - m1) / m0 >= 0.02:
    bad.append('wave')

# ---- halftone: darker → more ink
for lum, name in ((40, 'тёмное'), (220, 'светлое')):
    f2 = np.zeros((H, W, 4), np.float32); f2[..., :3] = lum; f2[..., 3] = 1
    got, g = run('halftone', {'cell': 8.0, 'angle': 0.0}, im=f2)
    print(f'     halftone {name}: покрытие {got[8:40, 8:56, 3].mean():.3f}')
dark = run('halftone', {'cell': 8.0, 'angle': 0.0}, im=np.dstack([np.full((H, W, 3), 40, np.float32), np.ones((H, W, 1), np.float32)]))[0][8:40, 8:56, 3].mean()
light = run('halftone', {'cell': 8.0, 'angle': 0.0}, im=np.dstack([np.full((H, W, 3), 220, np.float32), np.ones((H, W, 1), np.float32)]))[0][8:40, 8:56, 3].mean()
print('OK  ' if dark > light * 2 else 'FAIL', 'halftone: на тёмном краски больше')
if not dark > light * 2:
    bad.append('halftone')

# ================= the second batch: many-pass effects, tables, shader layers =================
from kadr import plugins
plugins.load()
FX = plugins.EFFECTS


def run_fx(name, pv, im=pic, mp=None, origin=(0, 0)):
    """an effect of plugins.EFFECTS (GPU kind) with its defaults"""
    fn, defaults, _ = FX[name]
    fx = Fx(1.0, 0.0, 0.0, 1, origin, None)
    fx.frame, fx.map = (im.shape[1], im.shape[0]), (kc.upload(np.ascontiguousarray(mp, np.float32)) if mp is not None else None)
    got = fn(kc.upload(np.ascontiguousarray(im)), dict(defaults, **pv), fx)
    out, g = got if isinstance(got, tuple) else (got, 0)
    return out.download(), g


# ---- bloom: nothing bright enough → nothing changes; a bright spot lights its surroundings and never darkens
got, g = run_fx('bloom', {'threshold': 2.0})
check('bloom: порог выше всего — картинка та же', got, grown(pic, g))
got, g = run_fx('bloom', {'strength': 0.0})
check('bloom: сила 0 — картинка та же', got, grown(pic, g))
spot = np.zeros((H, W, 4), np.float32); spot[..., 3] = 1; spot[..., :3] = 20; spot[20:28, 28:36, :3] = 255
got, g = run_fx('bloom', {'radius': 4.0, 'threshold': 0.5, 'strength': 1.0}, im=spot)
base = grown(spot, g)
near, far = got[g + 24, g + 38, 0], got[g + 2, g + 2, 0]      # 2 px beside the spot; a corner 30 px away
ok = (pm(got)[..., :3] >= pm(base)[..., :3] - 0.3).all() and near > 30 and abs(far - 20) < 1.0
print('OK  ' if ok else 'FAIL', f'bloom: яркое пятно светит вокруг себя ({near:.0f} рядом, {far:.0f} вдали при фоне 20), ничего не темнит')
if not ok:
    bad.append('bloom')

# ---- LUT: the identity table changes nothing, the negative one inverts
here = os.path.dirname(os.path.abspath(__file__))
got, g = run_fx('lut', {'file': os.path.join(here, 'luts', 'identity.cube')})
check('lut: тождественная таблица не меняет картинку', got, pic, 0.6)
got, g = run_fx('lut', {'file': os.path.join(here, 'luts', 'invert.cube')})
want = pic.copy(); want[8:40, 12:52, :3] = 255 - pic[8:40, 12:52, :3]
check('lut: таблица-негатив переворачивает цвет', got, want, 0.6)
got, g = run_fx('lut', {'file': os.path.join(here, 'luts', 'invert.cube'), 'amount': 0.0})
check('lut: amount 0 — картинка та же', got, pic, 0.6)

# ---- chroma key: the screen goes, the subject stays, its green cast is pulled out
shot = np.zeros((H, W, 4), np.float32); shot[..., :3] = [0, 208, 0]; shot[..., 3] = 1
shot[16:32, 20:44, :3] = [200, 40, 40]
shot[16:32, 44:48, :3] = [150, 190, 150]                       # a greenish fringe
got, g = run_fx('chroma_key', {}, im=shot)
ok = got[4, 4, 3] < 0.01 and got[24, 30, 3] > 0.99 and abs(got[24, 30, 0] - 200) < 3 and abs(got[24, 46, 1] - got[24, 46, 0]) <= 1.5
print('OK  ' if ok else 'FAIL', f'chroma_key: фон прозрачен ({got[4, 4, 3]:.2f}), предмет цел ({got[24, 30, 3]:.2f}), зелёный отсвет снят ({got[24, 46, 0]:.0f}, {got[24, 46, 1]:.0f})')
if not ok:
    bad.append('chroma_key')

# ---- light wrap: no strength, no change; a white background lightens the rim and not the middle
card = np.zeros((H, W, 4), np.float32); card[12:36, 16:48] = [40, 40, 40, 1]
bg = np.zeros((H, W, 4), np.float32); bg[...] = [255, 255, 255, 1]
got, g = run_fx('light_wrap', {'strength': 0.0}, im=card, mp=bg)
check('light_wrap: сила 0 — картинка та же', got, card)
got, g = run_fx('light_wrap', {'radius': 3.0, 'strength': 1.0}, im=card, mp=bg)
ok = got[24, 17, 0] > 60 and abs(got[24, 32, 0] - 40) < 2 and abs(got[24, 17, 3] - 1) < 1e-3 and got[4, 4, 3] < 1e-3
print('OK  ' if ok else 'FAIL', f'light_wrap: кромка светлее ({got[24, 17, 0]:.0f}), середина та же ({got[24, 32, 0]:.0f}), прозрачность не тронута')
if not ok:
    bad.append('light_wrap')

# ---- parallax: a flat depth at the focus moves nothing; all-near moves by the whole shift
dm = np.zeros((H, W, 4), np.float32); dm[...] = [127.5, 127.5, 127.5, 1]
got, g = run('parallax', {'shift': (10.0, 0.0), 'focus': 0.5}, mp=dm)
check('parallax: ровная глубина в фокусе ничего не двигает', got, pic)
dm[...] = [255, 255, 255, 1]
wide = np.zeros((H, W, 4), np.float32); wide[8:40, 12:40] = pic[8:40, 12:40]
got, g = run('parallax', {'shift': (10.0, 0.0), 'focus': 0.0}, im=wide, mp=dm)
want = np.zeros_like(wide); want[:, 10:] = wide[:, :-10]
check('parallax: всё близкое сдвигается ровно на shift', got[8:40, 24:50], want[8:40, 24:50])

# ---- a layer a shader draws: the flare is brightest at its source, a shader layer ends at its box
GEN = shader.generators()
fxg = Fx(1.0, 0.0, 0.0, 1, (0, 0), np.eye(3)); fxg.frame = (W, H)
im = shader.run(GEN['flare'], None, {'at': (0.25, 0.5), 'radius': 10.0}, fxg, size=(W, H), box=(W, H))[0].download()
lum = im[..., 3] * im[..., :3].sum(-1)
ok = lum[H // 2, W // 4] >= lum.max() - 1e-3 and im[2, W - 3, 3] < 0.2 and lum[H // 2, W // 4 + 12] < lum.max() * 0.6
print('OK  ' if ok else 'FAIL', f'flare: ярче всего в источнике ({lum[H // 2, W // 4]:.0f} из {lum.max():.0f}), в стороне слабее, вдали темно')
if not ok:
    bad.append('flare')
full = shader.parse('fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> { return vec4<f32>(1.0); }', 'white')
inv = np.array([[1, 0, -10.5], [0, 1, -8.0], [0, 0, 1.0]])    # the box 20 × 12 sits at (10.5, 8) of the picture
fxg = Fx(1.0, 0.0, 0.0, 1, (0, 0), inv); fxg.frame = (W, H)
im = shader.run(full, None, {}, fxg, size=(W, H), box=(20.0, 12.0))[0].download()[..., 3]
ok = abs(im.sum() - 240) < 0.5 and abs(im[10, 10] - 0.5) < 0.02 and im[10, 15] > 0.999 and im[4, 15] < 1e-3
print('OK  ' if ok else 'FAIL', f'слой-шейдер кончается на своей рамке: площадь {im.sum():.1f} из 240, полпикселя на краю {im[10, 10]:.2f}')
if not ok:
    bad.append('shader layer box')

# ---- a mistake in a shader is a message with the author's own line
e = shader.parse('fn effect(xy: vec2<f32>, p: FxP) -> vec4<f32> {\n    return src_at(xy) * nope;\n}', 'bad').compile_error()
ok = e is not None and 'строка 2' in e and 'nope' in e
print('OK  ' if ok else 'FAIL', 'ошибка в шейдере приходит текстом со своей строкой')
if not ok:
    bad.append('error text')

print('ВСЁ СХОДИТСЯ' if not bad else f'ЕСТЬ РАСХОЖДЕНИЯ: {bad}')
sys.exit(1 if bad else 0)
