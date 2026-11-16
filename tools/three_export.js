// A three.js scene → data for Kadr's scene3d layer (kadr/scene3d.py). Paste into the page's module (it needs the page's own
// `scene`, `renderer`, `camera`, and — for the record of moves — whatever sets the scene to a time: here worldAt(t),
// poseHero(t), camState(t, t) and the list of shots SH of the film; TL.total is the film's length), then
// from Node: page.evaluate(() => window.dumpScene(120)) and write json / bin / clip / env / pngs to files
// (the series' dump.mjs does exactly that). Keep the PMREM render target in scene.userData.pmrem for the environment.
// What comes out: scene.json, scene.bin (models), clip.bin (what moves, 120 a second), env.bin (CubeUV, half float), tex_N.png.
/* ---------- выгрузка сцены для Kadr: модели, материалы, свет, текстуры, запись движений (один раз) ---------- */
window.dumpScene = async (rate = 120) => {
  const b64 = a => { const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
  worldAt(0); poseHero(0); scene.updateMatrixWorld(true);
  const nodes = [], ids = new Map(), geos = new Map(), mats = new Map(), texs = new Map(), chunks = []; let off = 0;
  const push = a => { const f = a instanceof Float32Array || a instanceof Uint32Array ? a : null; chunks.push(f); const o = off; off += f.byteLength; return o; };
  const tex = t => { if (!t) return -1; if (!texs.has(t.uuid)) texs.set(t.uuid, { id: texs.size, png: t.image.toDataURL('image/png'), w: t.image.width, h: t.image.height, srgb: t.colorSpace === THREE.SRGBColorSpace,
    repeat: t.wrapS === THREE.RepeatWrapping, rep: [t.repeat.x, t.repeat.y], flipY: t.flipY, aniso: t.anisotropy }); return texs.get(t.uuid).id; };
  const mat = m => { if (!mats.has(m.uuid)) mats.set(m.uuid, { id: mats.size, name: m.name || '', uuid: m.uuid, kind: m.isMeshBasicMaterial ? 'basic' : m.isSpriteMaterial ? 'sprite' : 'standard',
    color: m.color.toArray(), roughness: m.roughness ?? 1, metalness: m.metalness ?? 0, emissive: m.emissive ? m.emissive.toArray() : [0, 0, 0], emissiveIntensity: m.emissiveIntensity ?? 0,
    opacity: m.opacity, transparent: m.transparent, side: m.side, map: tex(m.map), envMapIntensity: m.envMapIntensity ?? 1, clearcoat: m.clearcoat ?? 0, clearcoatRoughness: m.clearcoatRoughness ?? 0,
    additive: m.blending === THREE.AdditiveBlending, depthWrite: m.depthWrite, depthTest: m.depthTest }); return mats.get(m.uuid).id; };
  const geo = g => { if (!geos.has(g.uuid)) {
      const p = g.attributes.position, n = g.attributes.normal, u = g.attributes.uv, cnt = p.count;
      const v = new Float32Array(cnt * 8);
      for (let i = 0; i < cnt; i++) { v[i * 8] = p.getX(i); v[i * 8 + 1] = p.getY(i); v[i * 8 + 2] = p.getZ(i); v[i * 8 + 3] = n ? n.getX(i) : 0; v[i * 8 + 4] = n ? n.getY(i) : 0; v[i * 8 + 5] = n ? n.getZ(i) : 1; v[i * 8 + 6] = u ? u.getX(i) : 0; v[i * 8 + 7] = u ? u.getY(i) : 0; }
      const ix = g.index ? Uint32Array.from(g.index.array) : Uint32Array.from({ length: cnt }, (_, i) => i);
      geos.set(g.uuid, { id: geos.size, verts: cnt, v: push(v), tris: ix.length / 3, i: push(ix), kind: g.type });
    } return geos.get(g.uuid).id; };
  scene.traverse(o => { ids.set(o.uuid, nodes.length); nodes.push({ id: nodes.length, name: o.name || '', parent: o.parent ? ids.get(o.parent.uuid) ?? -1 : -1, kind: o.type, p: o.position.toArray(), q: o.quaternion.toArray(), s: o.scale.toArray(),
    visible: o.visible, cast: !!o.castShadow, receive: !!o.receiveShadow, geo: o.isMesh ? geo(o.geometry) : -1, mat: (o.isMesh || o.isSprite) ? mat(o.material) : -1 }); });
  const wp = o => o.getWorldPosition(new THREE.Vector3()).toArray();
  const lights = [];
  scene.traverse(o => { if (!o.isLight) return; const l = { kind: o.type, name: o.name || '', node: ids.get(o.uuid), color: o.color.toArray(), intensity: o.intensity, pos: wp(o) };
    if (o.isSpotLight) Object.assign(l, { target: wp(o.target), distance: o.distance, decay: o.decay, angle: o.angle, penumbra: o.penumbra, cast: o.castShadow,
      shadow: { size: o.shadow.mapSize.x, bias: o.shadow.bias, normalBias: o.shadow.normalBias, near: o.shadow.camera.near, far: o.shadow.camera.far, radius: o.shadow.radius, focus: o.shadow.focus } });
    if (o.isPointLight) Object.assign(l, { distance: o.distance, decay: o.decay });
    if (o.isHemisphereLight) Object.assign(l, { ground: o.groundColor.toArray() });
    if (o.isDirectionalLight) Object.assign(l, { target: wp(o.target) });
    if (o.isRectAreaLight) Object.assign(l, { width: o.width, height: o.height, quat: o.getWorldQuaternion(new THREE.Quaternion()).toArray() });
    lights.push(l); });
  // окружение: развёртка PMREM как есть (CubeUV, half float)
  const pm = scene.userData.pmrem, pw = pm.width, ph = pm.height, env16 = new Uint16Array(pw * ph * 4);
  renderer.readRenderTargetPixels(pm, 0, 0, pw, ph, env16);
  const env = { w: pw, h: ph, data: b64(env16), intensity: scene.environmentIntensity, mapping: pm.texture.mapping };
  const ltc = [THREE.UniformsLib.LTC_FLOAT_1, THREE.UniformsLib.LTC_FLOAT_2].map(t => ({ w: t.image.width, h: t.image.height, data: b64(new Float32Array(t.image.data)) }));
  // запись движений: что меняется во времени — узлы (положение, поворот, масштаб, видимость), цвета и прозрачность материалов, свет, камера
  const N = Math.ceil(TL.total * rate) + 2, state = () => { const a = []; scene.traverse(o => a.push(...o.position.toArray(), ...o.quaternion.toArray(), ...o.scale.toArray(), o.visible ? 1 : 0)); return a; };
  const mstate = () => [...mats.values()].flatMap(m => { const o = m.ref; return [...o.color.toArray(), o.opacity, o.emissiveIntensity ?? 0]; });
  scene.traverse(o => { if ((o.isMesh || o.isSprite) && mats.has(o.material.uuid)) mats.get(o.material.uuid).ref = o.material; });
  const lstate = () => { const a = []; scene.traverse(o => { if (o.isLight) a.push(...o.color.toArray(), o.intensity, ...wp(o)); }); return a; };
  const probe = [0, .37, 1.9, 7.3, 15.6, 18.2, 30.7, 34.1, 49.9, 53.3, 63.2, 66.4].map(t => { worldAt(t); poseHero(t); return [state(), mstate(), lstate()]; });
  const moving = (k, w0) => { const a = probe[0][k], set = new Set(); probe.forEach(p => p[k].forEach((v, i) => { if (Math.abs(v - a[i]) > 1e-7) set.add(Math.floor(i / w0)); })); return [...set].sort((x, y) => x - y); };
  const mn = moving(0, 11), mm = moving(1, 5), ml = moving(2, 7);
  const W0 = mn.length * 11 + mm.length * 5 + ml.length * 7 + 9, clip = new Float32Array(N * W0);
  for (let k = 0; k < N; k++) {
    const t = k / rate; worldAt(t); poseHero(t);
    const s = state(), m = mstate(), l = lstate(); let q = k * W0;
    mn.forEach(i => { for (let j = 0; j < 11; j++) clip[q++] = s[i * 11 + j]; });
    mm.forEach(i => { for (let j = 0; j < 5; j++) clip[q++] = m[i * 5 + j]; });
    ml.forEach(i => { for (let j = 0; j < 7; j++) clip[q++] = l[i * 7 + j]; });
    const c = camState(t, t); clip[q++] = c.pos.x; clip[q++] = c.pos.y; clip[q++] = c.pos.z; clip[q++] = c.at.x; clip[q++] = c.at.y; clip[q++] = c.at.z; clip[q++] = c.fov; clip[q++] = c.lens; clip[q++] = c.f;
  }
  worldAt(0); poseHero(0);
  const lightIdx = []; scene.traverse(o => { if (o.isLight) lightIdx.push(ids.get(o.uuid)); });
  const bin = new Uint8Array(off); let o2 = 0; chunks.forEach(c => { bin.set(new Uint8Array(c.buffer, c.byteOffset, c.byteLength), o2); o2 += c.byteLength; });
  return { json: { units: 'cm', nodes, geos: [...geos.values()], mats: [...mats.values()].map(({ ref, ...m }) => m), texs: [...texs.values()].map(({ png, ...t }) => t), lights,
      env: { w: env.w, h: env.h, intensity: env.intensity }, camera: { near: camera.near, far: camera.far, aspect: W / H },
      look: { exposure: L.exposure, bloom: .42, tint: [1.02, .99, .93], aces: .82, vignette: [.42, .86, .46], grain: .008, lampJitter, spp: SPP, shutter: SHUTTER * FPS },
      clip: { rate, n: N, width: W0, nodes: mn, mats: mm, lights: ml.map(i => lightIdx[i]), cuts: SH.map(x => x.t).filter(t => t > 0), total: TL.total } },
    bin: b64(bin), clip: b64(clip), env: env.data, ltc, pngs: [...texs.values()].map(t => t.png) };
};
