// 同一座島・八種風格
// One small Tidesong-style island (a detailed fishing cabin, jetty, rowboat, trees, a lagoon with swimming fish) rendered
// eight ways, one per look found in the reference posts. A style = camera + light + background + material family + post.
import * as THREE from 'three/webgpu';
import {
  Fn, uniform, float, vec2, vec3, vec4, mix, sin, cos, max, min, clamp, pow, exp, dot, normalize, length, smoothstep,
  step, fwidth, positionWorld, positionLocal, positionView, normalView, cameraPosition, screenUV, mx_noise_float,
  mx_fractal_noise_float, mx_worley_noise_vec2, pass, luminance, hash, instanceIndex, agxToneMapping,
  neutralToneMapping, acesFilmicToneMapping, sRGBTransferOETF, saturate, fog, uv, texture, perspectiveDepthToViewZ,
  orthographicDepthToViewZ, viewportSize, abs, If, attribute, dFdx, dFdy, cross, sign, floor, fract,
  viewportSharedTexture, refract,
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const DATA = await (await fetch('./styles.json')).json();

// bump from a procedural height: Mikkelsen's surface-gradient perturbation (bumpMap() only takes textures)
function bump(Hn, scale) {
  const sx = dFdx(positionView), sy = dFdy(positionView), n = normalView;
  const r1 = cross(sy, n), r2 = cross(n, sx), det = dot(sx, r1);
  return normalize(abs(det).mul(n).sub(sign(det).mul(r1.mul(dFdx(Hn)).add(r2.mul(dFdy(Hn)))).mul(scale)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Island (diorama-local metres, sea level 0). The block is 40 × 40 m.
// ─────────────────────────────────────────────────────────────────────────────
const S = 20, FLOOR = -2.4, BASE = -5;
const CAB = [3, -0.5];
const JETTY = { x: 4.2, z0: 7.0, z1: 16.5 };
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function smooth(a, b, x) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
function hash2(x, y) { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); }
function vnoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi, u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x, y, oct = 4) { let s = 0, a = 0.5, f = 1; for (let i = 0; i < oct; i++) { s += a * vnoise(x * f, y * f); a *= 0.5; f *= 2.03; } return s; }
function shoreR(x, z) { const g = Math.atan2(z, x); return 10.5 + (fbm(Math.cos(g) * 1.3 + 3, Math.sin(g) * 1.3 + 3, 3) - 0.5) * 5; }
function H(x, z) {
  const r = Math.hypot(x, z), inland = shoreR(x, z) - r;
  let h = Math.max(FLOOR, Math.min(0.7, inland * 0.28));
  h += 2.6 * Math.exp(-((x + 3) ** 2 + (z + 3) ** 2) / 30) * smooth(-1, 3, inland);
  h += (fbm(x * 0.15, z * 0.15, 3) - 0.5) * 0.5 * smooth(0, 3, inland);
  const dc = Math.hypot(x - CAB[0], z - CAB[1]);
  return h + (1.2 - h) * (1 - smooth(3.6, 5.4, dc));
}
function pathMask(x, z) {
  const ax = CAB[0] - 0.8, az = CAB[1] + 3, bx = JETTY.x, bz = JETTY.z0 + 0.5;
  const dx = bx - ax, dz = bz - az, t = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz)));
  return 1 - smooth(0.6, 1.3, Math.hypot(x - ax - dx * t, z - az - dz * t) + (vnoise(x * 2, z * 2) - 0.5) * 0.4);
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer, cameras, uniforms
// ─────────────────────────────────────────────────────────────────────────────
const canvas = document.getElementById('view');
const loaderEl = document.getElementById('loader');
if (!navigator.gpu) { loaderEl.innerHTML = '這個頁面需要 WebGPU<br><small>請用新版 Chrome / Edge</small>'; throw new Error('WebGPU unavailable'); }
const renderer = new THREE.WebGPURenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.NoToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
await renderer.init();

const scene = new THREE.Scene();
const world = new THREE.Group();
scene.add(world);
const persp = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.2, 40000);
const ortho = new THREE.OrthographicCamera(-30, 30, 20, -20, 0.1, 2000);
let camera = persp;
const controls = new OrbitControls(persp, canvas);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;

const U = {
  time: uniform(0), sunDir: uniform(new THREE.Vector3(0, 1, 0)), sunColor: uniform(new THREE.Vector3(1, 1, 1)), sunI: uniform(3),
  skyAmb: uniform(new THREE.Vector3(0.5, 0.6, 0.8)), glow: uniform(0.2), fogC: uniform(new THREE.Vector3()), fogSun: uniform(new THREE.Vector3()), fogDens: uniform(0),
  toon: uniform(0), exposure: uniform(1), tone: uniform(1), bloom: uniform(0.2), sat: uniform(1), split: uniform(0), curve: uniform(0),
  vig: uniform(0), grain: uniform(0), outline: uniform(0), outlineW: uniform(1), outlineC: uniform(new THREE.Vector3(0.12, 0.1, 0.1)),
  kuwa: uniform(0), water: uniform(0), paper: uniform(new THREE.Vector3(0.96, 0.93, 0.87)), tilt: uniform(0), focusY: uniform(0.5),
  ortho: uniform(0), near: uniform(0.2), far: uniform(40000), frame: uniform(0),
};
const viewZ = (d) => mix(perspectiveDepthToViewZ(d, U.near, U.far), orthographicDepthToViewZ(d, U.near, U.far), U.ortho).negate();

// ─────────────────────────────────────────────────────────────────────────────
// Sky, sun, environment, fog
// ─────────────────────────────────────────────────────────────────────────────
const sky = new SkyMesh();
sky.scale.setScalar(20000);
sky.mieDirectionalG.value = 0.8;
sky.cloudCoverage.value = 0.3;
scene.add(sky);
const sun = new THREE.DirectionalLight(0xffffff, 3);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
Object.assign(sun.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30, near: 1, far: 400 });
sun.shadow.bias = -0.0003;
sun.shadow.normalBias = 0.03;
scene.add(sun, sun.target);
const cubeRT = new THREE.CubeRenderTarget(128);
cubeRT.texture.type = THREE.HalfFloatType;
const cubeCam = new THREE.CubeCamera(0.5, 50000, cubeRT);
scene.environment = cubeRT.texture;
function captureSky() {
  const vis = sky.visible, bg = scene.background;
  world.visible = false; sky.visible = true; sky.showSunDisc.value = 0; scene.background = null;
  cubeCam.update(renderer, scene);
  scene.background = bg; world.visible = true; sky.visible = vis; sky.showSunDisc.value = 1;
}
const sunVec = (elev, az) => { const e = elev * Math.PI / 180; return new THREE.Vector3(Math.cos(az) * Math.cos(e), Math.sin(e), Math.sin(az) * Math.cos(e)).normalize(); };
let lightDir = new THREE.Vector3(0, 1, 0);
scene.fogNode = Fn(() => {
  const toP = positionWorld.sub(cameraPosition), dist = length(toP);
  const f = float(1).sub(exp(dist.mul(U.fogDens).mul(exp(max(positionWorld.y, 0).div(-30))).negate()));
  return fog(mix(U.fogC, U.fogSun, pow(saturate(dot(toP.div(dist), U.sunDir)), 6)), f);
})();

// ─────────────────────────────────────────────────────────────────────────────
// Material families. Every mesh carries a "real" PBR material and a "toon" one (3-step ramp, flat shaded).
// Surface detail (boards, shingles, stones) is procedural, in mesh-local space so it survives the turntable.
// ─────────────────────────────────────────────────────────────────────────────
const ramp = new THREE.DataTexture(new Uint8Array([70, 160, 255]), 3, 1, THREE.RedFormat);
ramp.minFilter = ramp.magFilter = THREE.NearestFilter; ramp.needsUpdate = true;
const swaps = [];
const C = (hex) => { const c = new THREE.Color(hex); return vec3(c.r, c.g, c.b); };
function mat({ color, toonColor = null, rough = 0.8, metal = 0, normal = null, emissive = null, physical = null, side = THREE.FrontSide, flat = true, alphaTest = 0, opacity = null }) {
  const real = physical ? new THREE.MeshPhysicalNodeMaterial({ roughness: rough, metalness: metal, side, ...physical }) : new THREE.MeshStandardNodeMaterial({ roughness: rough, metalness: metal, side });
  real.colorNode = color;
  if (normal) real.normalNode = normal;
  const toon = new THREE.MeshToonNodeMaterial({ gradientMap: ramp, side });
  toon.colorNode = toonColor ?? color;
  toon.flatShading = flat;
  for (const m of [real, toon]) {
    if (emissive) m.emissiveNode = emissive;
    if (alphaTest) { m.alphaTest = alphaTest; m.opacityNode = opacity; }
  }
  return { real, toon };
}
const plain = (hex, toonHex = hex, rough = 0.8) => mat({ color: C(hex).mul(mx_noise_float(positionLocal.mul(3.1)).mul(0.05).add(1)), toonColor: C(toonHex), rough });
function reg(mesh, p, shadow = true) { mesh.material = p.real; mesh.castShadow = mesh.receiveShadow = shadow; swaps.push({ mesh, p }); return mesh; }
function setToon(on) { for (const s of swaps) s.mesh.material = on ? s.p.toon : s.p.real; }

// horizontal clapboard siding: overlapping boards, a shadow line under each, slight per-board tint
function siding(hex, toonHex, board = 0.19) {
  const y = positionLocal.y.div(board), f = fract(y), id = floor(y);
  const tint = hash(id.add(17)).mul(0.08).add(0.96);
  const shade = mix(float(0.72), float(1), smoothstep(0.0, 0.22, f));
  const grain = mx_noise_float(positionLocal.mul(vec3(0.8, 18, 0.8))).mul(0.04).add(1);
  return mat({ color: C(hex).mul(shade).mul(tint).mul(grain), toonColor: C(toonHex).mul(mix(float(0.8), float(1), step(0.14, f))), rough: 0.75, normal: bump(f.mul(0.02), 1) });
}
// roof shingles on a slab whose local x runs along the ridge and local z down the slope
function shingles(hex, toonHex) {
  const row = positionLocal.z.div(0.24), r = floor(row), f = fract(row);
  const col = positionLocal.x.div(0.34).add(hash(r).mul(0.5)), c = floor(col), g = abs(fract(col).sub(0.5));
  const tint = hash(r.mul(13.1).add(c)).mul(0.22).add(0.86);
  const lip = mix(float(0.62), float(1), smoothstep(0.0, 0.3, f));
  const gap = mix(float(0.55), float(1), smoothstep(0.5, 0.44, g));
  const moss = smoothstep(0.55, 0.8, mx_fractal_noise_float(positionLocal.mul(0.9), 3)).mul(0.5);
  const col3 = mix(C(hex).mul(tint), vec3(0.28, 0.33, 0.14), moss).mul(lip).mul(gap);
  return mat({ color: col3, toonColor: C(toonHex).mul(mix(float(0.78), float(1), step(0.18, f))).mul(mix(float(0.85), float(1), step(g, 0.47))), rough: 0.85, normal: bump(f.mul(0.03), 1) });
}
// dry-stone blocks with mortar
function stones(scale = 1) {
  const w = mx_worley_noise_vec2(positionLocal.mul(vec3(2.2, 3.4, 2.2).mul(1 / scale)));
  const mortar = smoothstep(0.0, 0.08, w.y.sub(w.x));
  const tone = mx_noise_float(positionLocal.mul(7)).mul(0.08).add(hash(floor(w.x.mul(40))).mul(0.12)).add(0.85);
  return mat({ color: mix(vec3(0.62, 0.6, 0.55), vec3(0.5, 0.47, 0.42).mul(tone), mortar), toonColor: mix(C(0xd8d2c4), C(0xb0a898), mortar), rough: 0.92, normal: bump(mortar.mul(0.03), 1) });
}
// vertical planks (doors, crates)
function planks(hex, toonHex, width = 0.14) {
  const x = positionLocal.x.div(width), f = fract(x), id = floor(x);
  const shade = mix(float(0.6), float(1), smoothstep(0.0, 0.08, f).mul(smoothstep(1.0, 0.92, f)));
  const grain = mx_noise_float(positionLocal.mul(vec3(30, 1.5, 30))).mul(0.06).add(1);
  return mat({ color: C(hex).mul(shade).mul(hash(id).mul(0.1).add(0.95)).mul(grain), toonColor: C(toonHex).mul(mix(float(0.8), float(1), step(0.1, f))), rough: 0.8 });
}

// ─────────────────────────────────────────────────────────────────────────────
// Height texture (water bands), terrain top with caustics below the waterline, soil skirt
// ─────────────────────────────────────────────────────────────────────────────
const EXT = 24, HN = 256;
const heightTex = (() => {
  const d = new Uint16Array(HN * HN);
  for (let j = 0; j < HN; j++) for (let i = 0; i < HN; i++) {
    const x = (i + 0.5) / HN * 2 * EXT - EXT, z = (j + 0.5) / HN * 2 * EXT - EXT;
    d[j * HN + i] = THREE.DataUtils.toHalfFloat(Math.abs(x) > S || Math.abs(z) > S ? FLOOR : H(x, z));
  }
  const t = new THREE.DataTexture(d, HN, HN, THREE.RedFormat, THREE.HalfFloatType);
  t.magFilter = t.minFilter = THREE.LinearFilter; t.needsUpdate = true;
  return t;
})();
const heightAt = (xz) => texture(heightTex, xz.add(EXT).div(2 * EXT)).r;
function causticsAt(xz, depth) {
  const t = U.time;
  const warp = vec2(mx_noise_float(vec3(xz.mul(0.4), t.mul(0.2))), mx_noise_float(vec3(xz.mul(0.4).add(9), t.mul(0.2)))).mul(0.6);
  const wa = mx_worley_noise_vec2(vec3(xz.add(warp).mul(1.5), t.mul(0.5))), wb = mx_worley_noise_vec2(vec3(xz.add(warp).mul(2.0).add(3.3), t.mul(0.6).add(5)));
  const edge = (w) => float(1).sub(smoothstep(0, 0.07, w.y.sub(w.x)));
  return pow(edge(wa).mul(edge(wb).add(0.35)), 1.5).mul(smoothstep(0.05, 0.4, depth)).mul(exp(depth.mul(-0.3)));
}

const diorama = new THREE.Group();
world.add(diorama);
{
  const N = 220, pos = [], idx = [], path = [];
  for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) { const x = -S + i / N * 2 * S, z = -S + j / N * 2 * S; pos.push(x, H(x, z), z); path.push(pathMask(x, z)); }
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const a = j * (N + 1) + i, b = a + 1, c = a + N + 1, d = c + 1; idx.push(a, c, b, b, c, d); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('path', new THREE.Float32BufferAttribute(path, 1));
  g.setIndex(idx); g.computeVertexNormals();
  const p = positionLocal, h = p.y, pa = attribute('path', 'float');
  const n1 = mx_fractal_noise_float(p.mul(0.25), 3), n2 = mx_noise_float(p.mul(2.2));
  const grass = mix(vec3(0.1, 0.2, 0.04), vec3(0.24, 0.33, 0.07), saturate(n1.mul(0.8).add(0.5)));
  const sand = vec3(0.78, 0.68, 0.5).mul(n2.mul(0.05).add(1)).add(sin(p.x.mul(5).add(n2.mul(3))).mul(0.015));
  const beach = smoothstep(0.62, 0.4, h.add(n2.mul(0.06)));
  const wet = smoothstep(0.14, 0.02, h);
  const pebbles = smoothstep(0.85, 0.95, mx_worley_noise_vec2(p.xz.mul(6)).x.oneMinus()).mul(pa).mul(0.25);
  const under = smoothstep(0.05, -0.15, h);
  const caus = causticsAt(p.xz, h.negate()).mul(under);
  const sunWater = U.sunColor.mul(max(U.sunDir.y, 0)).mul(U.sunI).mul(0.18);
  const realCol = mix(mix(grass, sand.mul(mix(1, 0.62, wet)), beach), vec3(0.42, 0.33, 0.22).sub(pebbles), pa.mul(float(1).sub(beach)));
  const t = mat({ color: realCol, toonColor: mix(mix(vec3(0.36, 0.66, 0.22), vec3(0.98, 0.86, 0.6), step(h, 0.5)), vec3(0.78, 0.6, 0.4), step(0.5, pa).mul(step(0.5, h))), rough: 0.9, flat: false });
  t.real.roughnessNode = mix(float(0.92), float(0.35), wet.mul(beach));
  t.real.emissiveNode = caus.mul(sunWater);
  t.toon.emissiveNode = caus.mul(sunWater).mul(0.6);
  diorama.add(reg(new THREE.Mesh(g), t));

  const skirt = [];
  for (const f of [(k) => [-S + k * 2 * S, -S], (k) => [S, -S + k * 2 * S], (k) => [S - k * 2 * S, S], (k) => [-S, S - k * 2 * S]]) {
    const sp = [], si = [], M = 80;
    for (let k = 0; k <= M; k++) { const [x, z] = f(k / M); sp.push(x, H(x, z), z, x, BASE, z); }
    for (let k = 0; k < M; k++) { const a = k * 2; si.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3)); sg.setIndex(si); sg.computeVertexNormals();
    skirt.push(sg);
  }
  const ly = positionLocal.y;
  const layers = mix(mix(vec3(0.42, 0.3, 0.18), vec3(0.3, 0.2, 0.12), smoothstep(-2.9, -3.1, ly)), vec3(0.2, 0.18, 0.17), smoothstep(-4.1, -4.3, ly));
  const sk = reg(new THREE.Mesh(mergeGeometries(skirt)), mat({ color: layers.mul(mx_noise_float(positionLocal.mul(vec3(1.5, 6, 1.5))).mul(0.12).add(1)), toonColor: layers.mul(1.25), rough: 0.95, side: THREE.DoubleSide }));
  sk.name = 'skirt';
  diorama.add(sk);
}

// ─────────────────────────────────────────────────────────────────────────────
// Water: looks THROUGH the surface at what is really behind it (sea bed, fish, posts) via the shared viewport
// copy + depth, absorbs per RGB over the distance travelled, adds in-scatter, Fresnel sky, glints and shore foam.
// Toon mode: three depth bands tinting what lies beneath, and a crisp foam rim.
// ─────────────────────────────────────────────────────────────────────────────
// Minamo's clear-tropical coefficients ×2.5: a 2.4 m lagoon is too shallow to colour otherwise (art-directed)
const SIG_A = vec3(0.42, 0.065, 0.025).mul(2.5), SIG_S = vec3(0.02, 0.03, 0.045).mul(2.5), SIG_T = SIG_A.add(SIG_S);
const WAVES = [[0.3, 4.5, 0.03], [1.2, 2.7, 0.03], [-0.6, 1.6, 0.028], [2.1, 1.0, 0.024], [-1.4, 0.6, 0.02]];
const waterMat = new THREE.MeshBasicNodeMaterial({ transparent: true });
{
  const p = positionLocal;
  const fp = max(fwidth(p.x), fwidth(p.z));
  let sx = float(0), sz = float(0);
  for (const [a, l, st] of WAVES) {
    const k = 2 * Math.PI / l, w = Math.sqrt(9.81 * k), dx = Math.cos(a), dz = Math.sin(a);
    const s = cos(p.x.mul(dx * k).add(p.z.mul(dz * k)).sub(U.time.mul(w))).mul(st).mul(float(1).sub(smoothstep(l * 0.12, l * 0.5, fp)));
    sx = sx.add(s.mul(dx)); sz = sz.add(s.mul(dz));
  }
  const n = normalize(vec3(sx.negate(), 1, sz.negate()));
  const v = normalize(cameraPosition.sub(positionWorld));
  // what lies behind the surface (bed, fish, posts), nudged by the waves; the path length through the water comes
  // from the baked bed height along the refracted ray (the viewport depth copy is not reliable inside the pass)
  const depth = max(heightAt(p.xz).negate(), 0);
  const behind = viewportSharedTexture(screenUV.add(n.xz.mul(0.02).mul(smoothstep(0.0, 1.0, depth)))).rgb;
  const rr = refract(v.negate(), n, 1 / 1.333);
  const L = depth.div(max(rr.y.negate(), 0.25)).min(60);
  const T = exp(SIG_T.mul(L).negate());
  const light = U.sunColor.mul(U.sunI).mul(0.25).add(U.skyAmb);
  const under = behind.mul(T).add(SIG_S.div(SIG_T).mul(light).mul(float(1).sub(T)));
  const fres = float(0.02).add(pow(float(1).sub(saturate(dot(n, v))), 5).mul(0.98));
  const hv = normalize(U.sunDir.add(v));
  const spec = U.sunColor.mul(U.sunI).mul(pow(saturate(dot(n, hv)), 500).mul(20)).min(30);
  const fn = mx_fractal_noise_float(vec3(p.xz.mul(0.8), U.time.mul(0.25)), 2);
  const foam = saturate(smoothstep(0.55, 0.95, sin(depth.mul(9).sub(U.time.mul(1.5)).add(fn.mul(3)))).mul(smoothstep(0.7, 0.1, depth)).mul(smoothstep(-0.1, 0.4, fn)).add(smoothstep(0.18, 0.0, depth.add(fn.mul(0.05)))));
  const lightW = U.sunColor.mul(U.sunI).mul(max(U.sunDir.y, 0.1)).mul(0.8).add(U.skyAmb);
  const realCol = mix(mix(under, U.skyAmb.mul(1.3), fres).add(spec), lightW.mul(0.85), foam);
  const band = mix(mix(vec3(0.55, 0.95, 0.9), vec3(0.25, 0.75, 0.85), step(float(0.5), depth)), vec3(0.1, 0.48, 0.72), step(float(1.5), depth));
  const toonFoam = step(float(0.5), smoothstep(0.32, 0.05, depth.add(fn.mul(0.12)).add(sin(U.time.mul(1.6)).mul(0.03))));
  const toonCol = mix(mix(band.mul(lightW.mul(0.55)), behind.mul(band), 0.45), lightW.mul(0.95), toonFoam);
  waterMat.colorNode = mix(realCol, toonCol, U.toon);
}
const slabWater = new THREE.Mesh(new THREE.PlaneGeometry(2 * S, 2 * S).rotateX(-Math.PI / 2), waterMat);
slabWater.renderOrder = 10;
diorama.add(slabWater);
const seaWater = new THREE.Mesh(new THREE.PlaneGeometry(40000, 40000).rotateX(-Math.PI / 2), waterMat);
seaWater.renderOrder = 10;
world.add(seaWater);
const waterSideMat = new THREE.MeshBasicNodeMaterial({ transparent: true, side: THREE.DoubleSide, depthWrite: false });
{
  const d = saturate(positionLocal.y.negate().div(-FLOOR));
  waterSideMat.colorNode = mix(mix(vec3(0.25, 0.65, 0.68), vec3(0.03, 0.2, 0.32), d), mix(vec3(0.3, 0.82, 0.86), vec3(0.08, 0.42, 0.66), step(float(0.5), d)), U.toon)
    .mul(U.skyAmb.mul(0.6).add(U.sunColor.mul(U.sunI).mul(0.15)));
  waterSideMat.opacityNode = mix(0.7, 0.88, d);
}
const waterSides = new THREE.Group();
for (const [x, z, ry] of [[0, -S, 0], [S, 0, Math.PI / 2], [0, S, 0], [-S, 0, Math.PI / 2]]) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(2 * S, -FLOOR), waterSideMat);
  m.position.set(x, FLOOR / 2, z); m.rotation.y = ry; m.renderOrder = 11;
  waterSides.add(m);
}
diorama.add(waterSides);

// ─────────────────────────────────────────────────────────────────────────────
// The fishing cabin, in detail
// ─────────────────────────────────────────────────────────────────────────────
const glowWarm = vec3(1.0, 0.56, 0.22);
const M = {
  wall: siding(0xe9dfc9, 0xfff3dc), gable: siding(0xe9dfc9, 0xfff3dc), roof: shingles(0x9c3a26, 0xe0583c), stone: stones(), chimney: stones(0.7),
  trim: plain(0xf4f0e6, 0xffffff, 0.6), shutter: planks(0x3f7f86, 0x4fb3b8, 0.11), door: planks(0x2f6870, 0x3e9aa5, 0.15),
  wood: planks(0x6b4a2f, 0x9c6b43, 0.3), darkWood: plain(0x3a2a1c, 0x5a3e28), deck: planks(0xb08a5e, 0xe8bf86, 0.32),
  brass: mat({ color: C(0xc49a45), rough: 0.35, metal: 0.8 }), boat: plain(0x8d4f2c, 0xd9783f, 0.7), boatIn: planks(0xc69a6a, 0xf0c48a, 0.12),
  glass: mat({ color: C(0x1c2630), toonColor: C(0x2c3e50), rough: 0.08, metal: 0.3, emissive: glowWarm.mul(U.glow.mul(9)) }),
  soil: plain(0x4a3424, 0x7a5236), red: plain(0xc8352a, 0xff5a46, 0.5), white: plain(0xf2f0ea, 0xffffff, 0.5),
  leaf: plain(0x2f5a1a, 0x5fb83c, 0.85), leaf2: plain(0x3e6b1e, 0x88cc45, 0.85), pine: plain(0x16351f, 0x2f8a55, 0.85), bark: plain(0x4a3424, 0x7a5236, 0.9),
  rope: plain(0xb59a6a, 0xe0c48a, 0.9),
};
const flowerCols = [0xd8344a, 0xf2c230, 0xf5f0f5, 0xe86aa6].map((c) => plain(c, c, 0.6));
const net = mat({ color: C(0x9a8c6c), toonColor: C(0xd8c89a), rough: 0.9, side: THREE.DoubleSide, alphaTest: 0.5,
  opacity: smoothstep(0.38, 0.44, max(abs(fract(uv().x.mul(18)).sub(0.5)), abs(fract(uv().y.mul(14)).sub(0.5)))) });   // only the knotted twine survives the alpha test
const at = (x, z) => H(x, z);
const box = (w, h, d, p, r = 0.03) => reg(new THREE.Mesh(r ? new RoundedBoxGeometry(w, h, d, 2, r) : new THREE.BoxGeometry(w, h, d)), p);
const cabin = new THREE.Group();
{
  const g = cabin;
  g.position.set(CAB[0], at(...CAB) - 0.05, CAB[1]);
  g.rotation.y = 0.12;
  const w = 4.4, d = 3.4, h = 2.4, plinth = 0.45, pitch = 0.62;
  const plinthM = box(w + 0.25, plinth, d + 0.25, M.stone, 0.04); plinthM.position.y = plinth / 2;
  const body = box(w, h, d, M.wall, 0.05); body.position.y = plinth + h / 2;
  g.add(plinthM, body);
  // corner boards
  for (const [x, z] of [[w / 2, d / 2], [w / 2, -d / 2], [-w / 2, d / 2], [-w / 2, -d / 2]]) { const t = box(0.14, h, 0.14, M.trim, 0.02); t.position.set(x, plinth + h / 2, z); g.add(t); }
  // gable triangles (siding), roof slabs with overhang, ridge cap, barge boards
  const top = plinth + h, rise = (d / 2 + 0.35) * Math.tan(pitch), slope = (d / 2 + 0.55) / Math.cos(pitch);
  for (const s of [1, -1]) {
    const tri = new THREE.Shape([new THREE.Vector2(-d / 2, 0), new THREE.Vector2(d / 2, 0), new THREE.Vector2(0, (d / 2) * Math.tan(pitch))]);
    const gm = reg(new THREE.Mesh(new THREE.ShapeGeometry(tri)), M.gable);
    gm.rotation.y = Math.PI / 2 * s; gm.position.set(s * (w / 2), top, 0);
    g.add(gm);
    const slab = box(w + 0.7, 0.12, slope, M.roof, 0.03);
    slab.rotation.x = s * pitch;
    slab.position.set(0, top + rise / 2 - 0.05, s * (slope / 2) * Math.cos(pitch) - s * 0.2);
    g.add(slab);
    for (const e of [1, -1]) {
      const bb = box(0.08, 0.22, slope + 0.05, M.trim, 0.02);
      bb.rotation.x = s * pitch; bb.position.set(e * (w / 2 + 0.36), top + rise / 2 - 0.1, s * (slope / 2) * Math.cos(pitch) - s * 0.2);
      g.add(bb);
    }
  }
  const ridge = box(w + 0.75, 0.16, 0.26, M.darkWood, 0.05); ridge.position.set(0, top + rise + 0.02, 0); g.add(ridge);
  // windows: frame, glass, cross mullions, sill, louvred shutters, flower box
  const windowAt = (x, y, z, ry, box3 = true) => {
    const wg = new THREE.Group(); wg.position.set(x, y, z); wg.rotation.y = ry;
    const fr = box(1.0, 0.95, 0.08, M.trim, 0.02); const gl = box(0.8, 0.75, 0.04, M.glass, 0); gl.position.z = 0.03; gl.castShadow = false;
    const mv = box(0.05, 0.75, 0.06, M.trim, 0); mv.position.z = 0.05; const mh = box(0.8, 0.05, 0.06, M.trim, 0); mh.position.z = 0.05;
    const sill = box(1.15, 0.07, 0.22, M.trim, 0.02); sill.position.set(0, -0.5, 0.09);
    wg.add(fr, gl, mv, mh, sill);
    for (const s of [-1, 1]) { const sh = box(0.42, 0.95, 0.05, M.shutter, 0.015); sh.position.set(s * 0.74, 0, 0.02); sh.rotation.y = -s * 0.12; wg.add(sh); }
    if (box3) {
      const fb = box(1.0, 0.2, 0.25, M.wood, 0.02); fb.position.set(0, -0.68, 0.16); wg.add(fb);
      const soil = box(0.92, 0.04, 0.2, M.soil, 0); soil.position.set(0, -0.57, 0.16); wg.add(soil);
      const fr2 = mulberry32(Math.round(x * 10 + z * 7));
      for (let i = 0; i < 9; i++) {
        const f = reg(new THREE.Mesh(new THREE.IcosahedronGeometry(0.06 + fr2() * 0.03, 0)), flowerCols[Math.floor(fr2() * 4)], false);
        f.position.set(-0.4 + i * 0.1, -0.5 + fr2() * 0.08, 0.12 + fr2() * 0.1); wg.add(f);
        const lf = reg(new THREE.Mesh(new THREE.IcosahedronGeometry(0.07, 0)), M.leaf, false); lf.position.set(-0.36 + i * 0.09, -0.54, 0.2); wg.add(lf);
      }
    }
    g.add(wg);
  };
  windowAt(0.9, plinth + 1.35, d / 2 + 0.05, 0);
  windowAt(w / 2 + 0.05, plinth + 1.35, 0, Math.PI / 2, false);
  windowAt(-w / 2 - 0.05, plinth + 1.35, 0.3, -Math.PI / 2, false);
  windowAt(-0.9, plinth + 1.35, -d / 2 - 0.05, Math.PI, false);
  // door with frame, knob, little gabled hood on brackets, stone steps
  const doorG = new THREE.Group(); doorG.position.set(-1.0, plinth, d / 2 + 0.04);
  const df = box(1.05, 1.95, 0.08, M.trim, 0.02); df.position.y = 0.97;
  const dr = box(0.85, 1.8, 0.06, M.door, 0.015); dr.position.set(0, 0.92, 0.03);
  const kn = reg(new THREE.Mesh(new THREE.SphereGeometry(0.045, 10, 8)), M.brass); kn.position.set(0.3, 0.92, 0.09);
  doorG.add(df, dr, kn);
  for (const s of [1, -1]) { const hood = box(0.7, 0.05, 0.62, M.roof, 0.015); hood.rotation.z = s * 0.5; hood.position.set(s * 0.28, 2.2, 0.28); doorG.add(hood); }
  for (const s of [1, -1]) { const br = box(0.05, 0.05, 0.5, M.darkWood, 0); br.rotation.x = 0.7; br.position.set(s * 0.5, 1.95, 0.2); doorG.add(br); }
  g.add(doorG);
  for (let i = 0; i < 2; i++) { const st = box(1.2 - i * 0.2, 0.2, 0.45, M.stone, 0.03); st.position.set(-1.0, 0.1 + i * 0.2, d / 2 + 0.55 - i * 0.25); g.add(st); }
  // wall lantern by the door
  const lan = new THREE.Group(); lan.position.set(-1.75, plinth + 1.75, d / 2 + 0.15);
  const lanBody = reg(new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.24, 6)), M.glass); const lanCap = reg(new THREE.Mesh(new THREE.ConeGeometry(0.13, 0.12, 6)), M.darkWood); lanCap.position.y = 0.18;
  lan.add(lanBody, lanCap); g.add(lan);
  // chimney with cap
  const ch = box(0.6, 1.5, 0.6, M.chimney, 0.03); ch.position.set(1.3, top + rise - 0.1, -0.5);
  const cap = box(0.75, 0.1, 0.75, M.stone, 0.02); cap.position.set(1.3, top + rise + 0.68, -0.5);
  g.add(ch, cap);
  // fishing life: a net hung on the east wall with buoys, oars leaning, barrel, crates, a drying rack with fish
  const netM = reg(new THREE.Mesh(new THREE.PlaneGeometry(1.6, 1.3, 12, 8)), net, false);
  { const pp = netM.geometry.attributes.position; for (let i = 0; i < pp.count; i++) pp.setZ(i, Math.sin(pp.getX(i) * 4) * 0.05 + Math.abs(pp.getY(i)) * 0.04); netM.geometry.computeVertexNormals(); }
  netM.rotation.y = Math.PI / 2; netM.position.set(w / 2 + 0.12, plinth + 1.35, -1.1); g.add(netM);
  for (const [z, y, c] of [[-1.6, 1.0, M.red], [-0.6, 1.2, M.white], [-1.1, 0.8, M.red]]) { const b = reg(new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 10)), c); b.position.set(w / 2 + 0.2, plinth + y, z); g.add(b); }
  for (const s of [0, 1]) { const oar = box(0.07, 2.2, 0.05, M.deck, 0.02); oar.rotation.z = -0.25 - s * 0.06; oar.position.set(w / 2 + 0.3 + s * 0.14, 1.1, 1.0 + s * 0.15); const blade = box(0.18, 0.5, 0.03, M.deck, 0.02); blade.position.y = -0.95; oar.add(blade); g.add(oar); }
  const barrel = reg(new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.72, 16)), M.wood); barrel.position.set(1.9, plinth * 0 + 0.36, d / 2 + 0.55);
  for (const y of [-0.24, 0.24]) { const hoop = reg(new THREE.Mesh(new THREE.TorusGeometry(0.31, 0.025, 6, 20)), M.darkWood); hoop.rotation.x = Math.PI / 2; hoop.position.y = y; barrel.add(hoop); }
  const crate = box(0.65, 0.5, 0.65, M.wood, 0.03); crate.position.set(2.45, 0.25, d / 2 + 0.15); crate.rotation.y = 0.35;
  const crate2 = box(0.5, 0.4, 0.5, M.deck, 0.03); crate2.position.set(2.4, 0.7, d / 2 + 0.15); crate2.rotation.y = -0.2;
  g.add(barrel, crate, crate2);
  const rack = new THREE.Group(); rack.position.set(-3.2, 0, 1.6); rack.rotation.y = 0.5;
  for (const s of [-0.9, 0.9]) { const post = box(0.07, 1.5, 0.07, M.darkWood, 0.01); post.position.set(s, 0.75, 0); rack.add(post); }
  const bar = box(1.9, 0.05, 0.05, M.darkWood, 0); bar.position.y = 1.45; rack.add(bar);
  g.add(rack);
  cabin.userData.rack = rack;
  diorama.add(g);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fish: shaped bodies (tapered, laterally compressed), forked tail, dorsal + pectoral fins, eyes; scales,
// counter-shading and iridescence; the whole body undulates, more toward the tail
// ─────────────────────────────────────────────────────────────────────────────
function fishGeometry(L, Hh) {
  const SEG = 30, RAD = 16, pos = [], uvs = [], idx = [];
  for (let i = 0; i <= SEG; i++) {
    const t = i / SEG;                        // 0 tail … 1 snout
    const x = (t - 0.5) * L;
    let e = Math.pow(Math.sin(Math.PI * Math.min(1, t * 0.93 + 0.07)), 0.62);
    e = Math.max(e, t < 0.12 ? 0.14 : 0);     // caudal peduncle
    if (t > 0.92) e *= Math.sqrt(Math.max(0, (1 - t) / 0.08));
    const hh = Hh / 2 * e, hw = hh * 0.42, yc = -0.06 * Hh * e;
    for (let j = 0; j <= RAD; j++) {
      const a = j / RAD * Math.PI * 2;
      pos.push(x, yc + Math.cos(a) * hh, Math.sin(a) * hw);
      uvs.push(t, j / RAD);
    }
  }
  for (let i = 0; i < SEG; i++) for (let j = 0; j < RAD; j++) { const a = i * (RAD + 1) + j, b = a + RAD + 1; idx.push(a, a + 1, b, b, a + 1, b + 1); }   // outward-facing
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(idx); g.computeVertexNormals();
  return g;
}
function finGeometry(pts) { const g = new THREE.ShapeGeometry(new THREE.Shape(pts.map(([x, y]) => new THREE.Vector2(x, y)))); return g; }
const SPECIES = {
  bream: { L: 0.62, H: 0.26, back: 0xd2645a, belly: 0xf3e6e0, fin: 0xe08a7a, spots: true, stripes: false },
  mackerel: { L: 0.5, H: 0.13, back: 0x1f5f6a, belly: 0xdfe6ea, fin: 0x7c9aa0, spots: false, stripes: true },
};
const fishes = [];
function makeFish(kind, seed) {
  const sp = SPECIES[kind], L = sp.L, Hh = sp.H;
  const phase = uniform(seed * 7.3), speed = uniform(6 + (seed % 3));
  // undulation: lateral wave travelling tailward, amplitude growing toward the tail
  const und = (pl) => pl.add(vec3(0, 0, sin(pl.x.mul(9 / L).sub(U.time.mul(speed)).add(phase)).mul(L * 0.07).mul(pow(saturate(float(L * 0.3).sub(pl.x).div(L * 0.85)), 1.4))));
  const tU = uv().x, ring = uv().y;
  const up = positionLocal.y.div(Hh * 0.5);
  const counter = smoothstep(-0.35, 0.4, up);
  const sc = mx_worley_noise_vec2(vec2(tU.mul(L * 95), ring.mul(Hh * 120)));
  const scale = mix(float(0.84), float(1.0), smoothstep(0.0, 0.1, sc.y.sub(sc.x)));
  let col = mix(C(sp.belly), C(sp.back), counter).mul(scale);
  if (sp.spots) col = col.add(vec3(0.1, 0.35, 0.6).mul(step(0.86, mx_worley_noise_vec2(vec2(tU.mul(26), ring.mul(9))).x.oneMinus()).mul(counter)));
  if (sp.stripes) col = mix(col, vec3(0.03, 0.12, 0.14), step(0.62, sin(tU.mul(60).add(sin(ring.mul(40)).mul(1.5))).mul(0.5).add(0.5)).mul(smoothstep(0.35, 0.7, up)).mul(0.85));
  const gill = smoothstep(0.012, 0.0, abs(tU.sub(0.8))).mul(0.25);
  col = col.mul(float(1).sub(gill));
  const body = mat({ color: col, toonColor: mix(C(sp.belly), C(sp.back), step(0.1, up)), rough: 0.3, metal: 0.25, flat: false,
    physical: { iridescence: 0.6, iridescenceIOR: 1.35, clearcoat: 0.4, clearcoatRoughness: 0.2 } });
  const finMat = mat({ color: C(sp.fin).mul(mix(float(0.75), float(1.05), sin(uv().x.mul(40)).mul(0.5).add(0.5))), toonColor: C(sp.fin), rough: 0.5, side: THREE.DoubleSide, flat: false });
  for (const m of [body.real, body.toon, finMat.real, finMat.toon]) m.positionNode = und(positionLocal);
  const g = new THREE.Group();
  g.add(reg(new THREE.Mesh(fishGeometry(L, Hh)), body));
  const tail = finGeometry([[-L * 0.47, 0], [-L * 0.72, Hh * 0.62], [-L * 0.64, 0], [-L * 0.72, -Hh * 0.62]]);
  const dors = finGeometry([[-L * 0.12, Hh * 0.42], [L * 0.05, Hh * 0.75], [L * 0.2, Hh * 0.44]]);
  const anal = finGeometry([[-L * 0.22, -Hh * 0.38], [-L * 0.1, -Hh * 0.62], [0, -Hh * 0.4]]);
  for (const fg of [tail, dors, anal]) g.add(reg(new THREE.Mesh(fg), finMat));
  for (const s of [1, -1]) {
    const pec = reg(new THREE.Mesh(finGeometry([[L * 0.22, -Hh * 0.12], [L * 0.05, -Hh * 0.35], [L * 0.12, -Hh * 0.08]])), finMat);
    pec.position.z = s * Hh * 0.18; pec.rotation.x = s * 0.6; g.add(pec);
    const eye = reg(new THREE.Mesh(new THREE.SphereGeometry(Hh * 0.08, 10, 8)), mat({ color: vec3(0.02, 0.02, 0.025), rough: 0.05, metal: 0.2 }));
    eye.position.set(L * 0.38, Hh * 0.06, s * Hh * 0.13); g.add(eye);
  }
  // half the school circles the jetty (where the close-ups look), the rest roams round the island
  const nearJetty = seed % 2 === 0;
  g.userData = { cx: nearJetty ? JETTY.x - 3.6 : 0, cz: nearJetty ? 14 : 0, r: nearJetty ? 1.4 + (seed % 3) * 1.0 : 12.5 + (seed * 1.37 % 4.5), a: seed * 1.9, w: (nearJetty ? 0.32 : 0.16) * (seed % 3 ? 1 : -1), bob: seed * 0.7 };
  g.scale.setScalar(1.6);
  diorama.add(g);
  fishes.push(g);
  return g;
}
for (let i = 0; i < 14; i++) makeFish(i % 3 === 0 ? 'mackerel' : 'bream', i + 1);
// two fish hanging on the drying rack
for (let i = 0; i < 3; i++) {
  const f = makeFish('mackerel', 40 + i);
  fishes.pop();
  diorama.remove(f);
  f.scale.setScalar(0.9); f.rotation.z = -Math.PI / 2; f.position.set(-0.6 + i * 0.6, 1.1, 0);
  cabin.userData.rack.add(f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Jetty, lantern, rowboat, trees, rocks, grass, flowers
// ─────────────────────────────────────────────────────────────────────────────
{
  const g = new THREE.Group();
  const len = JETTY.z1 - JETTY.z0, n = Math.round(len / 0.36);
  for (let i = 0; i < n; i++) {
    const pl = box(1.8, 0.08, 0.3, i % 3 ? M.wood : M.deck, 0.03);
    pl.position.set(JETTY.x + (hash2(i, 1) - 0.5) * 0.06, 0.62, JETTY.z0 + 0.18 + i * len / n);
    pl.rotation.y = (hash2(i, 3) - 0.5) * 0.04;
    g.add(pl);
  }
  for (const s of [-0.7, 0.7]) { const str = box(0.12, 0.16, len, M.darkWood, 0.02); str.position.set(JETTY.x + s, 0.5, JETTY.z0 + len / 2); g.add(str); }
  for (let i = 0; i <= 4; i++) for (const s of [-0.85, 0.85]) {
    const z = JETTY.z0 + i * len / 4, y0 = Math.min(at(JETTY.x + s, z), 0), hp = 1.1 - y0;
    const post = reg(new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.1, hp, 8)), M.darkWood);
    post.position.set(JETTY.x + s, 1.1 - hp / 2, z); g.add(post);
    if (i === 4) { const rope = reg(new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.025, 6, 14)), M.rope); rope.rotation.x = Math.PI / 2; rope.position.set(JETTY.x + s, 0.9, z); g.add(rope); }
  }
  diorama.add(g);
  const lx = JETTY.x - 1.1, ly = at(lx, JETTY.z0);
  const lp = reg(new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 2.2, 8)), M.darkWood); lp.position.set(lx, ly + 1.1, JETTY.z0);
  const arm = box(0.5, 0.05, 0.05, M.darkWood, 0); arm.position.set(lx + 0.22, ly + 2.15, JETTY.z0);
  const lampMat = new THREE.MeshBasicNodeMaterial(); lampMat.colorNode = vec3(0.9, 0.85, 0.75).add(glowWarm.mul(U.glow.mul(18)));
  const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 8), lampMat); lamp.position.set(lx + 0.42, ly + 1.95, JETTY.z0);
  const hoodL = reg(new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.14, 8)), M.darkWood); hoodL.position.set(lx + 0.42, ly + 2.1, JETTY.z0);
  diorama.add(lp, arm, lamp, hoodL);
}
const lanternLight = new THREE.PointLight(0xffa25a, 0, 16, 2);
lanternLight.position.set(JETTY.x - 0.7, H(JETTY.x - 1.1, JETTY.z0) + 1.8, JETTY.z0);
diorama.add(lanternLight);
const windowLight = new THREE.PointLight(0xffa25a, 0, 10, 2);
windowLight.position.set(CAB[0] + 0.9, 2.3, CAB[1] + 2.7);
diorama.add(windowLight);
const rowboat = new THREE.Group();
{
  const outline = (s) => { const sh = new THREE.Shape(); sh.moveTo(0, -1.6 * s); sh.bezierCurveTo(0.75 * s, -1.0 * s, 0.68 * s, 1.0 * s, 0.42 * s, 1.5 * s); sh.lineTo(-0.42 * s, 1.5 * s); sh.bezierCurveTo(-0.68 * s, 1.0 * s, -0.75 * s, -1.0 * s, 0, -1.6 * s); return sh; };
  const hull = reg(new THREE.Mesh(new THREE.ExtrudeGeometry(outline(1), { depth: 0.42, bevelEnabled: true, bevelThickness: 0.06, bevelSize: 0.06, bevelSegments: 3, curveSegments: 16 }).rotateX(-Math.PI / 2).translate(0, -0.22, 0)), M.boat);
  const fl = reg(new THREE.Mesh(new THREE.ExtrudeGeometry(outline(0.84), { depth: 0.03, bevelEnabled: false, curveSegments: 16 }).rotateX(-Math.PI / 2).translate(0, 0.25, 0)), M.boatIn);
  const seat = box(1.1, 0.06, 0.28, M.trim, 0.01); seat.position.y = 0.36;
  const oar = box(0.06, 0.04, 2.6, M.deck, 0.01); oar.position.set(0.5, 0.4, 0.2); oar.rotation.y = 0.25;
  rowboat.add(hull, fl, seat, oar);
  rowboat.position.set(JETTY.x + 2.0, 0.12, JETTY.z1 - 2.2);
  rowboat.rotation.y = 0.2;
  diorama.add(rowboat);
}
// ─────────────────────────────────────────────────────────────────────────────
// Trees grown branch by branch, every leaf / needle its own instance (Pond Line style):
// broadleaf = trunk → 5 limbs → 4 twigs each, clusters of shaped, folded, veined leaves on the twigs;
// pine = straight trunk with drooping whorls, needles radiating along every branch.
// Leaves glow when the sun is behind them and flutter on top of the whole tree's sway.
// ─────────────────────────────────────────────────────────────────────────────
const trees = [];
const V3 = THREE.Vector3;
function leafGeometry(l, w, fold) {
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.bezierCurveTo(0.55 * w, 0.22 * l, 0.48 * w, 0.78 * l, 0, l);
  s.bezierCurveTo(-0.48 * w, 0.78 * l, -0.55 * w, 0.22 * l, 0, 0);
  const g = new THREE.ShapeGeometry(s, w < 0.05 ? 2 : 4);
  const p = g.attributes.position, uvA = g.attributes.uv;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i);
    p.setZ(i, Math.abs(x) * fold - Math.sin(y / l * Math.PI) * l * 0.08);   // folded along the midrib, gently arched
    uvA.setXY(i, x / w + 0.5, y / l);
  }
  g.computeVertexNormals();
  return g;
}
function leafMaterials(dark, light, autumn = 0) {
  const tip = uv().y, side = uv().x, vary = hash(float(instanceIndex));
  const rib = smoothstep(0.035, 0.0, abs(side.sub(0.5))).mul(smoothstep(0.98, 0.6, tip));
  const veins = smoothstep(0.82, 1.0, sin(tip.mul(26).sub(abs(side.sub(0.5)).mul(22))).mul(0.5).add(0.5)).mul(0.5);
  let col = mix(C(dark), C(light), vary.mul(0.8).add(tip.mul(0.2)));
  if (autumn) col = mix(col, vec3(0.75, 0.55, 0.12), step(1 - autumn, hash(float(instanceIndex).add(9))));
  col = col.mul(float(1).add(rib.mul(0.35)).add(veins.mul(0.12)));
  const v = normalize(cameraPosition.sub(positionWorld));
  const backlit = pow(saturate(dot(v.negate(), U.sunDir)), 4).mul(U.sunI).mul(0.35);
  const p = mat({ color: col, toonColor: C(light).mul(mix(float(0.8), float(1.2), step(0.5, vary))), rough: 0.6, side: THREE.DoubleSide, flat: false,
    emissive: col.mul(U.sunColor).mul(backlit).mul(float(1).sub(U.toon)) });
  // whole-tree sway is done on the group; here each leaf flutters about its own stem
  const flutter = sin(U.time.mul(7).add(vary.mul(40))).mul(0.03).mul(tip);
  for (const m of [p.real, p.toon]) m.positionNode = positionLocal.add(vec3(flutter, flutter.mul(0.5), flutter.negate()));
  return p;
}
function growTree(kind, seed) {
  const rnd = mulberry32(seed);
  const tubes = [], leaves = [];
  const add = (pts, r) => tubes.push(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), Math.max(3, pts.length * 2), r, 6, false));
  const bend = (d, k, droop = 0) => d.add(new V3((rnd() - 0.5) * k, (rnd() - 0.5) * k * 0.5 - droop, (rnd() - 0.5) * k)).normalize();
  const m4 = new THREE.Matrix4();
  const leafAt = (p, out, size, droop) => {
    const t = out.clone().add(new V3((rnd() - 0.5) * 0.9, (rnd() - 0.5) * 0.6 - droop, (rnd() - 0.5) * 0.9)).normalize();
    const side = new V3().crossVectors(t, new V3(rnd() - 0.5, 1, rnd() - 0.5)).normalize();
    const n = new V3().crossVectors(side, t);
    m4.makeBasis(side, t, n).scale(new V3(size, size, size)).setPosition(p);
    leaves.push(m4.clone());
  };
  if (kind === 'round') {
    const trunk = [new V3(0, 0, 0)]; let p = new V3(), d = new V3(0, 1, 0);
    for (let i = 0; i < 5; i++) { d = bend(d.clone(), 0.18); p = p.clone().addScaledVector(d, 0.55); trunk.push(p); }
    add(trunk, 0.17);
    const top = trunk[trunk.length - 1];
    for (let b = 0; b < 6; b++) {
      const a = b / 6 * Math.PI * 2 + rnd() * 0.6, start = trunk[3 + (b % 2)].clone();
      let bd = new V3(Math.cos(a), 0.9 + rnd() * 0.5, Math.sin(a)).normalize(), bp = start.clone();
      const limb = [bp.clone()];
      for (let i = 0; i < 4; i++) { bd = bend(bd, 0.3); bp = bp.clone().addScaledVector(bd, 0.42); limb.push(bp); }
      add(limb, 0.075);
      for (const q of limb.slice(3)) for (let k = 0; k < 40; k++) {
        const lp = q.clone().add(new V3((rnd() - 0.5) * 0.8, (rnd() - 0.4) * 0.6, (rnd() - 0.5) * 0.8));
        leafAt(lp, lp.clone().sub(new V3(top.x, top.y - 0.2, top.z)).normalize(), 1.0 + rnd() * 0.5, 0.25);
      }
      for (let t = 0; t < 6; t++) {
        const from = limb[1 + (t % 4)].clone();
        let td = bend(bd.clone().add(new V3(rnd() - 0.5, 0.3, rnd() - 0.5)), 0.6), tp = from.clone();
        const twig = [tp.clone()];
        for (let i = 0; i < 3; i++) { td = bend(td, 0.4, 0.05); tp = tp.clone().addScaledVector(td, 0.32); twig.push(tp); }
        add(twig, 0.03);
        for (const q of twig.slice(1)) for (let k = 0; k < 70; k++) {
          const lp = q.clone().add(new V3((rnd() - 0.5) * 0.75, (rnd() - 0.4) * 0.6, (rnd() - 0.5) * 0.75));
          const out = lp.clone().sub(new V3(top.x, top.y - 0.2, top.z)).normalize();
          leafAt(lp, out, 1.0 + rnd() * 0.5, 0.25);
        }
      }
    }
  } else {
    const H0 = 5.6;
    add([new V3(0, 0, 0), new V3(0, H0 * 0.5, 0), new V3(0.02, H0, 0)], 0.14);
    add([new V3(0.02, H0 - 0.6, 0), new V3(0.02, H0 + 0.35, 0)], 0.03);
    for (let y = 1.0; y < H0 - 0.3; y += 0.42) {
      const k = 1 - (y - 1) / (H0 - 1.2), n = 5 + Math.floor(rnd() * 2), len = 0.4 + 1.6 * k;
      for (let b = 0; b < n; b++) {
        const a = b / n * Math.PI * 2 + y * 2.3;
        let bd = new V3(Math.cos(a), 0.05, Math.sin(a)).normalize(), bp = new V3(0, y, 0);
        const br = [bp.clone()];
        for (let i = 0; i < 3; i++) { bd = bend(bd, 0.15, 0.12); bp = bp.clone().addScaledVector(bd, len / 3); br.push(bp); }
        add(br, 0.014 + 0.012 * k);
        for (let i = 1; i < br.length; i++) for (let j = 0; j < 120; j++) {
          const q = br[i - 1].clone().lerp(br[i], rnd()).add(new V3((rnd() - 0.5) * 0.22, (rnd() - 0.3) * 0.16, (rnd() - 0.5) * 0.22));
          const out = new V3((rnd() - 0.5) * 2, rnd() * 0.8 - 0.1, (rnd() - 0.5) * 2).normalize().add(bd.clone().multiplyScalar(0.6)).normalize();
          leafAt(q, out, 1.7 + rnd() * 0.6, 0.12);
        }
      }
    }
    for (let j = 0; j < 160; j++) leafAt(new V3((rnd() - 0.5) * 0.3, H0 - 0.4 + rnd() * 0.7, (rnd() - 0.5) * 0.3), new V3(rnd() - 0.5, 1, rnd() - 0.5).normalize(), 0.8, 0);
  }
  return { wood: mergeGeometries(tubes.map((t) => t.toNonIndexed())), leaves };
}
const LEAF = { round: leafGeometry(0.17, 0.1, 0.35), pine: leafGeometry(0.2, 0.018, 0.1) };
const LEAF_MATS = { round: [leafMaterials(0x1f4710, 0x5a8a22), leafMaterials(0x2a5212, 0x7a9a2a, 0.08)], pine: [leafMaterials(0x12331b, 0x3c6e34)] };
{
  const tr = mulberry32(9);
  [[-6.5, 1.5, 'round'], [-2.5, -7.2, 'round'], [5.8, -5.5, 'round'], [-8.3, -3.8, 'round'], [-4.2, -1.2, 'pine'], [-0.8, -4.6, 'pine'], [0.6, -8.4, 'pine'], [-7.2, 4.6, 'pine'], [-5.4, -6.0, 'pine']].forEach(([x, z, kind], i) => {
    const g = new THREE.Group(); g.position.set(x, at(x, z) - 0.1, z); g.scale.setScalar(0.85 + tr() * 0.35); g.rotation.y = tr() * 6.28;
    const { wood, leaves } = growTree(kind, 100 + i * 17);
    g.add(reg(new THREE.Mesh(wood), M.bark));
    const lm = LEAF_MATS[kind][i % LEAF_MATS[kind].length];
    const im = new THREE.InstancedMesh(LEAF[kind], lm.real, leaves.length);
    leaves.forEach((m, k) => im.setMatrixAt(k, m));
    reg(im, lm);
    g.add(im);
    g.userData.ph = tr() * 6.28;
    trees.push(g); diorama.add(g);
  });
}
{
  const rr = mulberry32(31);
  for (let i = 0; i < 16; i++) {
    const a = rr() * Math.PI * 2, R = shoreR(Math.cos(a), Math.sin(a)) - 0.4 + rr() * 1.2, x = Math.cos(a) * R, z = Math.sin(a) * R;
    if (Math.abs(x - JETTY.x) < 2 && z > 5) continue;
    const g = new THREE.IcosahedronGeometry(0.3 + rr() * 0.6, 1);
    const p = g.attributes.position; for (let k = 0; k < p.count; k++) p.setY(k, p.getY(k) * 0.65);
    g.computeVertexNormals();
    const m = reg(new THREE.Mesh(g), M.stone); m.position.set(x, at(x, z) + 0.05, z); m.rotation.y = rr() * 6; diorama.add(m);
  }
}
{
  const bg = new THREE.BufferGeometry(); const w = 0.035, h = 0.32;
  bg.setAttribute('position', new THREE.Float32BufferAttribute([-w, 0, 0, w, 0, 0, -w * 0.6, h * 0.5, 0.01, w * 0.6, h * 0.5, 0.01, 0, h, 0.03], 3));
  bg.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 0.5, 1, 0.5, 0.5, 1], 2));
  bg.setAttribute('normal', new THREE.Float32BufferAttribute([0, 0.5, 1, 0, 0.5, 1, 0, 0.6, 1, 0, 0.6, 1, 0, 0.8, 1], 3));
  bg.setIndex([0, 1, 2, 1, 3, 2, 2, 3, 4]);
  const tip = uv().y, vary = hash(float(instanceIndex));
  const sway = sin(positionLocal.x.mul(0.5).add(positionLocal.z.mul(0.2)).sub(U.time.mul(1.6))).mul(0.5).add(0.5).mul(0.7).add(sin(U.time.mul(3).add(vary.mul(30))).mul(0.1));
  const gp = mat({ color: mix(vec3(0.06, 0.12, 0.03), mix(vec3(0.24, 0.36, 0.07), vec3(0.4, 0.4, 0.13), vary.mul(vary)), tip), toonColor: mix(vec3(0.3, 0.6, 0.2), vec3(0.55, 0.85, 0.35), tip), side: THREE.DoubleSide, flat: false });
  for (const m of [gp.real, gp.toon]) m.positionNode = positionLocal.add(vec3(sway.mul(0.12), 0, sway.mul(0.04)).mul(tip.mul(tip)));
  const gr = mulberry32(5), mats = [];
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), pv = new THREE.Vector3(), yA = new THREE.Vector3(0, 1, 0);
  for (let t = 0; t < 60000 && mats.length < 9000; t++) {
    const x = (gr() * 2 - 1) * 13, z = (gr() * 2 - 1) * 13, hh = H(x, z);
    if (hh < 0.6 || pathMask(x, z) > 0.3 || Math.hypot(x - CAB[0], z - CAB[1]) < 3.4 || Math.hypot(x + 3.2 - CAB[0] + CAB[0], z - 1.1) < 1) continue;
    q.setFromAxisAngle(yA, gr() * 6.28); const k = 0.7 + gr() * 0.8;
    mats.push(m4.compose(pv.set(x, hh - 0.03, z), q, s.set(k, k * (0.8 + gr() * 0.6), k)).clone());
  }
  const im = new THREE.InstancedMesh(bg, gp.real, mats.length);
  mats.forEach((m, i) => im.setMatrixAt(i, m));
  im.receiveShadow = true;
  swaps.push({ mesh: im, p: gp });
  diorama.add(im);
  const fr = mulberry32(77), fm = [];
  for (let t = 0; t < 20000 && fm.length < 260; t++) {
    const x = (fr() * 2 - 1) * 12, z = (fr() * 2 - 1) * 12, hh = H(x, z);
    if (hh < 0.7 || pathMask(x, z) > 0.3 || Math.hypot(x - CAB[0], z - CAB[1]) < 3.6) continue;
    if (fbm(x * 0.3 + 5, z * 0.3) < 0.5) continue;
    fm.push(new THREE.Matrix4().makeTranslation(x, hh + 0.28, z));
  }
  const fv = hash(float(instanceIndex).add(3));
  const fc = mix(mix(vec3(0.95, 0.93, 0.85), vec3(0.98, 0.8, 0.2), step(float(0.45), fv)), vec3(0.95, 0.45, 0.6), step(float(0.8), fv));
  const fl = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.07, 0), null, fm.length);
  reg(fl, mat({ color: fc, rough: 0.6 }), false);
  fm.forEach((m, i) => fl.setMatrixAt(i, m));
  diorama.add(fl);
}

// table (miniature): graph paper; shadow catcher for the flat-background styles
const table = new THREE.Mesh(new THREE.PlaneGeometry(400, 400).rotateX(-Math.PI / 2), (() => {
  const m = new THREE.MeshStandardNodeMaterial({ roughness: 0.9 });
  const p = positionWorld.xz;
  const lineF = float(1).sub(smoothstep(0.0, 0.035, min(abs(p.x.sub(p.x.round())), abs(p.y.sub(p.y.round())))));
  const q = p.div(5);
  const lineC = float(1).sub(smoothstep(0.0, 0.012, min(abs(q.x.sub(q.x.round())), abs(q.y.sub(q.y.round())))));
  m.colorNode = vec3(0.93, 0.92, 0.88).mul(float(1).sub(lineF.mul(0.1))).mul(vec3(1).sub(vec3(0.05, 0.12, 0.16).mul(lineC)));
  return m;
})());
table.position.y = BASE - 0.02;
table.receiveShadow = true;
world.add(table);
const catcher = new THREE.Mesh(new THREE.PlaneGeometry(600, 600).rotateX(-Math.PI / 2), new THREE.ShadowNodeMaterial({ opacity: 0.22 }));
catcher.position.y = BASE - 1.2;
catcher.receiveShadow = true;
world.add(catcher);

// ─────────────────────────────────────────────────────────────────────────────
// Post-processing
// ─────────────────────────────────────────────────────────────────────────────
const pipeline = new THREE.RenderPipeline(renderer);
function buildPipeline() {
  const scenePass = pass(scene, camera);
  const hdr = scenePass.getTextureNode();
  const depthTex = scenePass.getTextureNode('depth');
  const glow = bloom(hdr, 0.2, 0.5, 1.0);
  glow.strength = U.bloom;
  const px = vec2(1).div(viewportSize);
  pipeline.outputNode = Fn(() => {
    const uv0 = screenUV;
    const wob = vec2(mx_noise_float(vec3(uv0.mul(vec2(14, 9)), 1)), mx_noise_float(vec3(uv0.mul(vec2(14, 9)), 7))).mul(0.0035).mul(U.water);
    const uvw = uv0.add(wob);
    const col = hdr.sample(uvw).rgb.toVar();
    If(U.kuwa.greaterThan(0.5), () => {
      let acc = vec3(0), ws = float(0);
      const st = px.mul(2.4);
      for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
        let s = vec3(0), s2 = vec3(0);
        for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { const c = hdr.sample(uvw.add(vec2(sx * i, sy * j).mul(st))).rgb.min(6); s = s.add(c); s2 = s2.add(c.mul(c)); }
        const m = s.div(16), v = dot(s2.div(16).sub(m.mul(m)), vec3(1)).max(0);
        const w = float(1).div(pow(v.div(luminance(m).mul(luminance(m)).add(0.02)).mul(60).add(1), 4));
        acc = acc.add(m.mul(w)); ws = ws.add(w);
      }
      col.assign(acc.div(ws));
    });
    If(U.tilt.greaterThan(0.5), () => {
      const b = smoothstep(0.05, 0.36, abs(uv0.y.sub(U.focusY)));
      let acc = vec3(0);
      for (let i = 0; i < 16; i++) { const a = i * 2.39996, r = Math.sqrt((i + 0.5) / 16); acc = acc.add(hdr.sample(uv0.add(vec2(Math.cos(a) * r, Math.sin(a) * r).mul(px).mul(b.mul(14)))).rgb); }
      col.assign(mix(col, acc.div(16), smoothstep(0, 0.15, b)));
    });
    const lit = col.add(glow.rgb.mul(vec3(1.0, 0.82, 0.6))).mul(U.exposure);
    const one = float(1);
    const w = (k) => max(float(1).sub(U.tone.sub(k).abs()), 0);
    const tm = agxToneMapping(lit, one).mul(w(1)).add(neutralToneMapping(lit, one).mul(w(2))).add(acesFilmicToneMapping(lit, one).mul(w(3))).add(lit.mul(w(0)));
    let o = sRGBTransferOETF(tm);
    const l = luminance(o);
    o = mix(vec3(l), o, U.sat);
    o = o.add(vec3(0.05, 0.025, -0.03).mul(smoothstep(0.35, 1.0, l)).mul(U.split));
    o = o.add(vec3(-0.015, 0.01, 0.028).mul(pow(float(1).sub(l), 2)).mul(U.split));
    o = mix(o, smoothstep(0.0, 1.0, o), U.curve);
    const vz = (off) => viewZ(depthTex.sample(uv0.add(off)).r);
    const ow = px.mul(U.outlineW);
    const c0 = vz(vec2(0));
    const lap = vz(vec2(ow.x, 0)).add(vz(vec2(ow.x.negate(), 0))).add(vz(vec2(0, ow.y))).add(vz(vec2(0, ow.y.negate()))).sub(c0.mul(4));
    const edgeK = abs(lap).div(c0.max(0.01));
    o = mix(o, U.outlineC, smoothstep(0.004, 0.012, edgeK).mul(U.outline).mul(0.9));
    const uvc = uv0.sub(0.5);
    const paperN = mx_fractal_noise_float(vec3(uv0.mul(vec2(viewportSize.x.div(viewportSize.y), 1)).mul(6), 3), 4).mul(0.05).add(mx_noise_float(vec3(uv0.mul(viewportSize).mul(0.35), 9)).mul(0.03));
    const pool = smoothstep(0.003, 0.02, edgeK).mul(0.4);
    let wc = mix(vec3(luminance(o)), o, 0.8).mul(float(1).sub(pool)).mul(paperN.mul(1.6).add(0.95));
    wc = mix(wc, vec3(1), 0.08);
    const border = smoothstep(0.56, 0.8, length(uvc.mul(vec2(1.05, 1.5))).add(mx_fractal_noise_float(vec3(uv0.mul(5), 2), 3).mul(0.06)));
    wc = mix(wc, U.paper.mul(paperN.add(0.98)), border);
    o = mix(o, wc, U.water);
    o = o.mul(float(1).sub(pow(length(uvc.mul(vec2(1.0, 0.85))).mul(1.35), 2.6).mul(U.vig)));
    o = o.add(hash(uv0.x.mul(1973.1).add(uv0.y.mul(9277.3)).add(U.frame.mul(26.699))).sub(0.5).mul(U.grain).mul(2));
    return vec4(o, 1);
  })();
  pipeline.outputColorTransform = false;
  pipeline.needsUpdate = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Styles. `close` = a detail framing (cabin + jetty + fish) for checking the craft.
// ─────────────────────────────────────────────────────────────────────────────
const TONE = { none: 0, agx: 1, neutral: 2, aces: 3 };
const STYLE = {
  cinematic: { sun: [6, 2.4, [1, 0.68, 0.42], 3.2], sky: [3.5, 2.2], bg: 'sky', slab: false, cam: 'p', fov: 38, pos: [21, 3.2, 27], at: [0, 1.2, 0], close: { pos: [9.5, 2.4, 15], at: [3, 1.0, 3] },
    exp: 0.8, tone: 'agx', bloom: 0.12, sat: 1.05, split: 0, curve: 0.1, vig: 0.25, grain: 0.012, fog: 0.004, fogC: [0.32, 0.36, 0.46], fogS: [0.85, 0.55, 0.35], glow: 0.5 },
  miniature: { sun: [52, 2.0, [1, 0.97, 0.92], 3.0], sky: [2, 1], bg: [0.9, 0.89, 0.85], slab: true, table: true, cam: 'p', fov: 28, pos: [44, 36, 50], at: [0, -1.5, 1], close: { pos: [16, 12, 20], at: [3, 0.5, 3], focusY: 0.5 },
    exp: 0.42, tone: 'neutral', bloom: 0.05, sat: 1.32, split: 0.3, curve: 0.25, vig: 0.2, grain: 0.008, tilt: 1, focusY: 0.47, glow: 0.12 },
  isometric: { sun: [45, 2.2, [1, 0.97, 0.92], 2.6], sky: [2, 1], bg: [0.94, 0.92, 0.88], slab: true, catcher: true, cam: 'o', zoom: 31, close: { zoom: 9, target: [3, 1, 2] },
    exp: 0.5, tone: 'neutral', bloom: 0, sat: 1.1, split: 0, curve: 0.12, outline: 1, outlineW: 1, outlineC: [0.2, 0.17, 0.15], glow: 0.12 },
  watercolor: { sun: [40, 2.3, [1, 0.95, 0.86], 2.8], sky: [2, 1], bg: [0.97, 0.95, 0.9], slab: true, catcher: true, cam: 'p', fov: 32, pos: [40, 32, 44], at: [0, -1, 0], close: { pos: [14, 9, 17], at: [2.5, 1, 2] },
    exp: 0.5, tone: 'neutral', bloom: 0, sat: 0.85, split: 0.2, curve: 0, kuwa: 1, water: 1, outlineW: 2, glow: 0.12 },
  cozy: { sun: [50, 2.1, [1, 0.96, 0.9], 2.6], sky: [2, 1], bg: [0.62, 0.84, 0.95], slab: true, catcher: true, toon: true, cam: 'p', fov: 30, pos: [30, 44, 40], at: [0, -1, 1], close: { pos: [12, 14, 17], at: [3, 0.5, 3] },
    exp: 0.52, tone: 'neutral', bloom: 0.04, sat: 1.25, split: 0, curve: 0.1, outline: 1, outlineW: 2, outlineC: [0.16, 0.14, 0.2], glow: 0.12 },
  dreamy: { sun: [4.5, 2.4, [1, 0.62, 0.36], 3.4], sky: [4, 3], bg: 'sky', slab: false, cam: 'p', fov: 42, pos: [13, 3.6, -20], at: [-1, 1.8, 3], close: { pos: [12, 2, 2], at: [3, 1.4, -0.5] },
    exp: 0.85, tone: 'neutral', bloom: 0.45, sat: 1.25, split: 1, curve: 0.35, vig: 0.22, grain: 0.01, fog: 0.008, fogC: [0.5, 0.42, 0.45], fogS: [1.0, 0.62, 0.38], glow: 0.6 },
  moody: { sun: [-1.5, 2.4, [0.5, 0.6, 1.0], 0.3], moon: true, sky: [3, 2.6], bg: 'sky', slab: false, cam: 'p', fov: 40, pos: [19, 4.2, 22], at: [2, 1.5, 3], close: { pos: [8.5, 2.2, 13], at: [2, 1.4, 2] },
    exp: 2.2, tone: 'aces', bloom: 0.35, sat: 0.95, split: 0.4, curve: 0.2, vig: 0.4, grain: 0.02, fog: 0.022, fogC: [0.04, 0.06, 0.1], fogS: [0.1, 0.08, 0.1], glow: 1.0 },
  editorial: { sun: [38, 2.5, [1, 0.97, 0.93], 2.4], sky: [2, 1], bg: [0.95, 0.93, 0.89], slab: true, catcher: true, cam: 'p', fov: 22, pos: [96, 62, 104], at: [0, -2, 0], close: { pos: [20, 11, 24], at: [3, 1.2, 2] },
    exp: 0.5, tone: 'neutral', bloom: 0, sat: 1.05, split: 0, curve: 0.1, glow: 0.12, offset: -0.24, spin: true },
};
let current = null, viewOffset = 0, orthoHalf = 30, closeUp = false;
function placeCamera() {
  const s = STYLE[current], c = closeUp ? s.close : null;
  if (s.cam === 'o') {
    const t = c?.target ?? [0, -1.5, 0], d = 120;
    ortho.position.set(t[0] + d * 0.7071 * Math.cos(0.6155), t[1] + d * Math.sin(0.6155), t[2] + d * 0.7071 * Math.cos(0.6155));
    orthoHalf = c?.zoom ?? s.zoom; ortho.zoom = 1;
    controls.target.set(...t);
  } else {
    persp.fov = s.fov; persp.position.set(...(c?.pos ?? s.pos)); controls.target.set(...(c?.at ?? s.at));
  }
  U.focusY.value = c?.focusY ?? s.focusY ?? 0.5;
  viewOffset = closeUp ? 0 : (s.offset ?? 0);
  resize();
  controls.update();
}
function applyStyle(id, close = closeUp) {
  const s = STYLE[id];
  current = id; closeUp = close;
  document.body.className = document.body.className.replace(/style-\w+/g, '').trim() + (closeUp ? '' : ` style-${id}`);
  const sv = sunVec(s.sun[0], s.sun[1]);
  sky.sunPosition.value.copy(sv); sky.turbidity.value = s.sky[0]; sky.rayleigh.value = s.sky[1];
  lightDir = s.moon ? sunVec(35, s.sun[1] + 2.0) : sv.clone();
  U.sunDir.value.copy(lightDir); U.sunColor.value.set(...s.sun[2]); U.sunI.value = s.sun[3];
  sun.color.setRGB(...s.sun[2]); sun.intensity = s.sun[3];
  captureSky();
  sky.visible = s.bg === 'sky';
  // flat backdrops are pre-divided by the exposure so they come out of the post chain at their own colour
  scene.background = s.bg === 'sky' ? null : new THREE.Color().setRGB(...s.bg, THREE.SRGBColorSpace).multiplyScalar(1 / s.exp);
  U.skyAmb.value.set(...(s.fogC ?? [0.55, 0.65, 0.8])).multiplyScalar(s.moon ? 0.6 : 1);
  U.fogDens.value = closeUp ? (s.fog ?? 0) * 0.4 : (s.fog ?? 0);
  U.fogC.value.set(...(s.fogC ?? [0.6, 0.7, 0.85])); U.fogSun.value.set(...(s.fogS ?? [0.9, 0.85, 0.8]));
  U.glow.value = s.glow; lanternLight.intensity = 40 * s.glow; windowLight.intensity = 18 * s.glow;
  slabWater.visible = waterSides.visible = s.slab;
  diorama.getObjectByName('skirt').visible = s.slab;
  seaWater.visible = !s.slab;
  table.visible = !!s.table;
  catcher.visible = !!s.catcher;
  setToon(!!s.toon);
  U.toon.value = s.toon ? 1 : 0;
  const want = s.cam === 'o' ? ortho : persp;
  if (want !== camera) { camera = want; controls.object = camera; buildPipeline(); }
  U.ortho.value = s.cam === 'o' ? 1 : 0;
  diorama.rotation.y = 0;
  placeCamera();
  U.exposure.value = s.exp; U.tone.value = TONE[s.tone]; U.bloom.value = s.bloom; U.sat.value = s.sat; U.split.value = s.split; U.curve.value = s.curve;
  U.vig.value = s.vig ?? 0; U.grain.value = s.grain ?? 0; U.outline.value = s.outline ?? 0; U.outlineW.value = s.outlineW ?? 1;
  U.outlineC.value.set(...(s.outlineC ?? [0.12, 0.1, 0.1])); U.kuwa.value = s.kuwa ?? 0; U.water.value = s.water ?? 0; U.tilt.value = s.tilt ?? 0;
  syncUI();
}

// ─────────────────────────────────────────────────────────────────────────────
// UI
// ─────────────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const stylesEl = $('styles'), infoEl = $('info');
for (const st of DATA.styles) {
  const b = document.createElement('button');
  b.className = 'chip'; b.textContent = st.name; b.dataset.id = st.id;
  b.onclick = () => applyStyle(st.id);
  stylesEl.append(b);
}
const closeBtn = document.createElement('button');
closeBtn.className = 'chip'; closeBtn.textContent = '近看細節'; closeBtn.dataset.id = 'close';
closeBtn.onclick = () => applyStyle(current, !closeUp);
stylesEl.append(closeBtn);
const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
function syncUI() {
  for (const b of stylesEl.children) b.classList.toggle('on', b.dataset.id === current || (b.dataset.id === 'close' && closeUp));
  const st = DATA.styles.find((x) => x.id === current);
  infoEl.innerHTML = `<h2>${esc(st.name)}</h2><div class="refs">參考：${st.refs.map(esc).join('、')}</div><p class="look">${esc(st.look)}</p>`
    + `<div class="label">關鍵技巧</div><ul>${st.keys.map((k) => `<li>${esc(k)}</li>`).join('')}</ul>`
    + `<div class="label">下次這樣提示我</div><p class="prompt">${esc(st.prompt)}</p>`;
}
$('toggle').onclick = () => $('panel').classList.remove('hidden');
addEventListener('keydown', (e) => { if (e.key === 'h' || e.key === 'H') $('panel').classList.toggle('hidden'); });

// ─────────────────────────────────────────────────────────────────────────────
// Loop, resize, capture
// ─────────────────────────────────────────────────────────────────────────────
const timer = new THREE.Timer();
let simT = 12, frozen = false, fpsAcc = 0, fpsN = 0, captureSize = null, pendingShot = null;
const hud = $('hud');
function frame(now) {
  timer.update(now);
  const dt = Math.min(timer.getDelta(), 0.1);
  if (!frozen) simT += dt;
  U.time.value = simT; U.frame.value = (U.frame.value + 1) % 1000;
  rowboat.position.y = 0.14 + Math.sin(simT * 1.1) * 0.04; rowboat.rotation.z = Math.sin(simT * 0.9) * 0.03;
  for (const t of trees) t.rotation.z = Math.sin(simT * 1.3 + t.userData.ph) * 0.015 + 0.01;
  for (const f of fishes) {
    const u = f.userData, a = u.a + simT * u.w, r = u.r + Math.sin(simT * 0.3 + u.bob) * 0.8;
    const x = u.cx + Math.cos(a) * r, z = u.cz + Math.sin(a) * r;
    const dx = -Math.sin(a) * Math.sign(u.w), dz = Math.cos(a) * Math.sign(u.w);
    f.position.set(x, Math.min(-0.3, H(x, z) * 0.45) + Math.sin(simT + u.bob) * 0.05, z);
    f.rotation.set(0, Math.atan2(-dz, dx), Math.sin(simT * 0.8 + u.bob) * 0.06);
  }
  if (STYLE[current]?.spin && !frozen && !closeUp) diorama.rotation.y += dt * 0.12;
  controls.update();
  sun.target.position.set(0, 0, 0);
  sun.position.copy(lightDir).multiplyScalar(120);
  pipeline.render();
  if (pendingShot && --pendingShot.wait <= 0) { const s = pendingShot; pendingShot = null; s.resolve(canvas.toDataURL('image/jpeg', 0.9)); }
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { hud.textContent = `${Math.round(fpsN / fpsAcc)} FPS　·　拖曳旋轉、滾輪縮放、H 隱藏面板`; fpsAcc = 0; fpsN = 0; }
}
function resize() {
  const [w, h] = captureSize ?? [innerWidth, innerHeight];
  if (!captureSize) renderer.setSize(w, h);
  const aspect = w / h;
  persp.aspect = aspect;
  if (viewOffset) persp.setViewOffset(w, h, viewOffset * w, 0, w, h); else persp.clearViewOffset();
  persp.updateProjectionMatrix();
  ortho.left = -orthoHalf * aspect; ortho.right = orthoHalf * aspect; ortho.top = orthoHalf; ortho.bottom = -orthoHalf;
  ortho.updateProjectionMatrix();
  U.near.value = camera.near; U.far.value = camera.far;
}
addEventListener('resize', resize);
window.styles = {
  apply: applyStyle, STYLE, U,
  async captureBegin(w = 1600, h = 900) { captureSize = [w, h]; frozen = true; simT = 12; document.body.classList.add('capture'); renderer.setPixelRatio(1); renderer.setSize(w, h, false); resize(); },
  captureEnd() { captureSize = null; frozen = false; document.body.classList.remove('capture'); renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); resize(); },
  async shot(name, wait = 16) {
    const url = await new Promise((resolve) => { pendingShot = { wait, resolve }; });
    return (await fetch(`/save?name=${name}`, { method: 'POST', body: url })).ok;
  },
  // a low look over the jetty at the fish, in whatever style is active
  fishCam() { persp.fov = 26; persp.position.set(JETTY.x - 0.6, 3.4, 19.5); controls.target.set(JETTY.x - 3.6, -0.5, 14); viewOffset = 0; resize(); controls.update(); },
  // the broadleaf by the cabin, close enough to read single leaves
  treeCam() { persp.fov = 30; persp.position.set(-1.2, 3.4, 8.5); controls.target.set(-6.2, 3.0, 1.2); viewOffset = 0; resize(); controls.update(); },
  async all(prefix = 'st_', ids = DATA.styles.map((s) => s.id)) {
    const done = [];
    for (const id of ids) {
      applyStyle(id, false); if (await this.shot(`${prefix}${id}`)) done.push(id);
      applyStyle(id, true); if (await this.shot(`${prefix}${id}_close`)) done.push(id + '_close');
    }
    applyStyle(ids[0], false);
    return done;
  },
};

buildPipeline();
applyStyle('cinematic', false);
renderer.setAnimationLoop(frame);
setTimeout(() => loaderEl.classList.add('gone'), 400);
