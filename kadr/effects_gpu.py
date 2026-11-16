"""Effects of several GPU passes: each is a function of the layer's picture on the GPU (a kadr_core Img: rgb 0..255,
alpha) that calls shader passes (kadr/shaders/_*.wgsl) and the blur kernels — fn(im, pv, fx) → (Img, px it grew by).
The CPU renderer calls the same function (shader.apply_numpy), so both renderers show the same picture."""
import copy, math, os, threading
import numpy as np
from . import shader
from .plugins import effect

_tl = threading.local()                                        # tables on the GPU belong to the thread's device


def gpu_effect(name, params, grow=None, takes_map=False, timed=False):
    def deco(fn):
        fn.kadr_gpu, fn.takes_map, fn.kadr_timed = True, takes_map, timed
        return effect(name, params=params, grow=grow)(fn)
    return deco


def _with(fx, **kw):
    out = copy.copy(fx)
    for k, v in kw.items():
        setattr(out, k, v)
    return out


@gpu_effect('bloom', {'radius': 24.0, 'threshold': 0.6, 'knee': 0.2, 'strength': 1.0, 'tint': (255, 255, 255)},
            grow=lambda p, s: 6 * p['radius'] * s)
def bloom(im, pv, fx):
    """Свечение ярких мест: всё ярче threshold (плавно на ширине knee) размывается в три радиуса (radius/2, radius, 2·radius px слоя) и ложится экраном поверх. AE: Glow с порогом, Deep Glow, Optical Glow"""
    import kadr_core as kc
    sig = max(pv['radius'] * fx.scale, 0.3)
    R = int(math.ceil(6 * sig)) + 1
    pre, _ = shader.run(shader.internal('bloom_pre'), im, {'threshold': pv['threshold'], 'knee': pv['knee'], 'tint': pv['tint']}, fx, grow=0)
    pm = kc.padded(kc.pm(pre, 0), im.w + 2 * R, im.h + 2 * R, R, R)
    acc = kc.new_img(im.w + 2 * R, im.h + 2 * R, [0.0, 0, 0, 0])
    for m in (0.5, 1.0, 2.0):
        kc.accumulate(acc, kc.gauss(pm, sig * m, sig * m), 0, 0, True)
    fx2 = _with(fx, map=acc, map_origin=(fx.origin[0] - R, fx.origin[1] - R))
    return shader.run(shader.internal('bloom_mix'), im, {'strength': pv['strength']}, fx2, grow=R)


def _cube(path):
    """a .cube 3D LUT → (size, its table on this thread's GPU), read once"""
    import kadr_core as kc
    cache = _tl.__dict__.setdefault('luts', {})
    key = (path, os.path.getmtime(path))
    if key not in cache:
        n, rows = 0, []
        for line in open(path, encoding='utf-8', errors='replace'):
            t = line.split()
            if not t or t[0].startswith('#'):
                continue
            if t[0] == 'LUT_3D_SIZE':
                n = int(t[1])
            elif t[0] == 'LUT_1D_SIZE':
                raise shader.ShaderError(f'{path}: это 1D LUT, нужен 3D (.cube с LUT_3D_SIZE)')
            elif t[0][0].isdigit() or t[0][0] in '-.':
                rows.append([float(v) for v in t[:3]])
        if n < 2 or len(rows) != n ** 3:
            raise shader.ShaderError(f'{path}: в файле LUT {len(rows)} строк, для размера {n} нужно {n ** 3}')
        cache.clear()
        cache[key] = (n, kc.upload_plane(np.ascontiguousarray(np.array(rows, np.float32).reshape(n * n * n, 3))))
    return cache[key]


@gpu_effect('lut', {'file': '', 'amount': 1.0})
def lut(im, pv, fx):
    """Цветовая таблица из файла .cube (3D LUT): file — путь рядом с фильмом, amount — сила от 0 до 1. AE: Apply Color LUT, Lumetri"""
    if not pv['file']:
        return im, 0
    n, table = _cube(pv['file'])
    return shader.run(shader.internal('lut'), im, {'size': float(n), 'amount': pv['amount']}, fx, data=table, grow=0)


@gpu_effect('light_wrap', {'radius': 12.0, 'strength': 0.6}, takes_map=True)
def light_wrap(im, pv, fx):
    """Свет фона по краю вырезанного слоя: map — слой фона; его размытый цвет ложится экраном на кромку шириной radius px слоя. AE: Light Wrap после кеинга"""
    import kadr_core as kc
    bg = getattr(fx, 'map', None)
    if bg is None or pv['strength'] <= 0:
        return im, 0
    sig = max(pv['radius'] * fx.scale, 0.3)
    soft = kc.gauss(kc.pm(im, 0), sig, sig)                    # the layer's blurred alpha: 1 inside, falling to the edge
    wrap, _ = shader.run(shader.internal('wrap_edge'), soft, {}, _with(fx, map=kc.gauss(bg, sig, sig), map_origin=(0.0, 0.0)), grow=0, half=False)
    return shader.run(shader.internal('wrap_mix'), im, {'strength': pv['strength']}, _with(fx, map=wrap, map_origin=fx.origin), grow=0)
