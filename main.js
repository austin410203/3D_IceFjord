// 冰峽灣・渲染技巧實驗室
// A blue-hour Norwegian fjord (snowy mountains, icy sea, drifting floes, a fishing village with warm windows) built
// to demonstrate, one switch at a time, the rendering techniques studied in Minamo and Pond Line.
// Every switch is a uniform or a visibility flag, so toggling never recompiles a pipeline.
import * as THREE from 'three/webgpu';
import {
  Fn, uniform, float, vec2, vec3, vec4, mix, sin, cos, max, clamp, pow, exp, dot, normalize, length, smoothstep,
  select, fwidth, reflector, positionWorld, positionLocal, normalWorld, cameraPosition, screenUV, mx_noise_float,
  mx_fractal_noise_float, pass, luminance, hash, instanceIndex, agxToneMapping, neutralToneMapping,
  acesFilmicToneMapping, sRGBTransferOETF, saturate, mod, fog, uv, dFdx, dFdy, cross, sign, abs, positionView, normalView,
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// bump from a procedural height (three's bumpMap() only takes textures): Mikkelsen's surface-gradient perturbation
function proceduralBump(H, scale) {
  const sx = dFdx(positionView), sy = dFdy(positionView), n = normalView;
  const r1 = cross(sy, n), r2 = cross(n, sx);
  const det = dot(sx, r1);
  const grad = sign(det).mul(r1.mul(dFdx(H)).add(r2.mul(dFdy(H)))).mul(scale);
  return normalize(abs(det).mul(n).sub(grad));
}

const DATA = await (await fetch('./techniques.json')).json();

// ─────────────────────────────────────────────────────────────────────────────
// Parameters (metres). Water at y = 0; the fjord runs toward −z and opens to the sea far away.
// ─────────────────────────────────────────────────────────────────────────────
const W = 230;                                    // channel half-width near the village
const SUN_ELEV = 3.4 * Math.PI / 180;
const SUN_DIR = new THREE.Vector3(0.32, Math.tan(SUN_ELEV), -1).normalize();   // towards the sun
const SUN_COLOR = new THREE.Color(1.0, 0.66, 0.42);
const SEED = 11;

function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
const rng = mulberry32(SEED);

// fjord centreline meanders; it widens into open sea beyond z ≈ −2600
const cx = (z) => 140 * Math.sin(z * 0.0016) + 60 * Math.sin(z * 0.0041 + 1.3);
const halfW = (z) => W + 1600 * smooth(-2300, -3600, z);
function smooth(a, b, x) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
function hash2(x, y) { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); }
function vnoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x, y, oct = 5) { let s = 0, a = 0.5, f = 1; for (let i = 0; i < oct; i++) { s += a * vnoise(x * f, y * f); a *= 0.5; f *= 2.03; } return s; }
function ridged(x, y) { let s = 0, a = 0.55, f = 1; for (let i = 0; i < 6; i++) { const n = 1 - Math.abs(vnoise(x * f, y * f) * 2 - 1); s += a * n * n; a *= 0.5; f *= 2.1; } return s; }

function terrainHeight(x, z) {
  const d = Math.abs(x - cx(z)), hw = halfW(z);
  const n = fbm(x * 0.004, z * 0.004);
  // underwater floor → steep shore → a low rocky shelf the village sits on
  const shore = -40 + 46 * smooth(hw - 70, hw + 12, d + (n - 0.5) * 70);
  // mountain walls, fading out where the fjord opens to the sea
  const rise = smooth(hw + 40, hw + 560, d + (n - 0.5) * 140);
  const open = smooth(-3500, -2300, z);
  const mtn = (360 + 620 * ridged(x * 0.0021 + 3.1, z * 0.0021)) * rise * (0.25 + 0.75 * open);
  // rocky detail on the shelf and slopes
  const detail = (fbm(x * 0.035, z * 0.035, 4) - 0.5) * 9 * smooth(hw - 10, hw + 60, d);
  // far islands at the mouth
  const isl = Math.max(0, fbm(x * 0.003 + 7, z * 0.003) - 0.58) * 900 * (1 - open);
  return shore + mtn + detail + isl;
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer
// ─────────────────────────────────────────────────────────────────────────────
const canvas = document.getElementById('view');
const loaderEl = document.getElementById('loader');
if (!navigator.gpu) { loaderEl.innerHTML = '這個場景需要 WebGPU<br><small>請用新版 Chrome / Edge</small>'; throw new Error('WebGPU unavailable'); }
const renderer = new THREE.WebGPURenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.NoToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;      // WebGPU's PCF is already soft (PCFSoft was folded into it)
await renderer.init();

const scene = new THREE.Scene();
const world = new THREE.Group();          // everything except the sky (hidden while the sky probe is captured)
scene.add(world);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.5, 60000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;

// ─────────────────────────────────────────────────────────────────────────────
// Switch uniforms
// ─────────────────────────────────────────────────────────────────────────────
const U = {
  time: uniform(0),
  sunDir: uniform(SUN_DIR.clone()),
  sunColor: uniform(new THREE.Vector3(SUN_COLOR.r, SUN_COLOR.g, SUN_COLOR.b)),
  skyAmb: uniform(1),         // brightness of the sky light that tints the water body / snow particles
  snow: uniform(1), ice: uniform(1), lights: uniform(1), fog: uniform(1),
  bloom: uniform(0.3), tone: uniform(1), grade: uniform(1), vig: uniform(0.25), grain: uniform(0.012), ca: uniform(0.0016),
  exposure: uniform(1), frame: uniform(0),
};

// ─────────────────────────────────────────────────────────────────────────────
// Sky, sun and environment
// ─────────────────────────────────────────────────────────────────────────────
const sky = new SkyMesh();
sky.scale.setScalar(40000);
sky.turbidity.value = 3.5;
sky.rayleigh.value = 2.6;
sky.mieCoefficient.value = 0.006;
sky.mieDirectionalG.value = 0.82;
sky.cloudCoverage.value = 0.32;
sky.cloudDensity.value = 0.45;
sky.sunPosition.value.copy(SUN_DIR);
scene.add(sky);
const FLAT_BG = new THREE.Color(0x8d99a6);

const sun = new THREE.DirectionalLight(SUN_COLOR, 3.2);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
const sc = sun.shadow.camera;
sc.left = -800; sc.right = 800; sc.top = 800; sc.bottom = -800; sc.near = 10; sc.far = 6000;
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.6;
scene.add(sun, sun.target);
const ambient = new THREE.AmbientLight(0xffffff, 0);    // the "naive" fill used when the sky light is off
scene.add(ambient);

// sky probe: the sun disc is cut out first so it is not counted twice (the directional light carries it)
const cubeRT = new THREE.CubeRenderTarget(256);
cubeRT.texture.type = THREE.HalfFloatType;
const cubeCam = new THREE.CubeCamera(1, 100000, cubeRT);
function captureSky() {
  world.visible = false; sky.showSunDisc.value = 0;
  cubeCam.update(renderer, scene);
  world.visible = true; sky.showSunDisc.value = 1;
}
captureSky();
scene.environment = cubeRT.texture;

// ─────────────────────────────────────────────────────────────────────────────
// Fog: aerial perspective, denser low down, warm toward the sun
// ─────────────────────────────────────────────────────────────────────────────
const fogBlue = uniform(new THREE.Vector3(0.11, 0.16, 0.25));
const fogSun = uniform(new THREE.Vector3(0.62, 0.36, 0.22));
scene.fogNode = Fn(() => {
  const toP = positionWorld.sub(cameraPosition);
  const dist = length(toP);
  const dir = toP.div(dist);
  const hAvg = max(positionWorld.y.add(cameraPosition.y).mul(0.5), 0);
  const dens = float(0.00042).mul(exp(hAvg.div(-350)));
  const f = float(1).sub(exp(dist.mul(dens).negate())).mul(U.fog);
  const sunAmt = pow(saturate(dot(dir, U.sunDir)), 5);
  return fog(mix(fogBlue, fogSun, sunAmt), f);
})();

// ─────────────────────────────────────────────────────────────────────────────
// Terrain: one warped grid, dense around the village, coarse toward the mountains
// ─────────────────────────────────────────────────────────────────────────────
const VZ = 160;                                     // z of the village centre
const VX = cx(VZ) - W;                              // left shoreline there
function buildTerrain() {
  const N = 440;
  const pos = new Float32Array((N + 1) * (N + 1) * 3);
  const idx = [];
  const map = (t, a, b) => (t < 0 ? -a : b) * t * t;
  for (let j = 0; j <= N; j++) {
    const tz = j / N * 2 - 1;
    const z = VZ - map(tz, 1100, 4600);
    for (let i = 0; i <= N; i++) {
      const tx = i / N * 2 - 1;
      const x = VX + 60 + map(tx, 1900, 2300);
      const k = (j * (N + 1) + i) * 3;
      pos[k] = x; pos[k + 1] = terrainHeight(x, z); pos[k + 2] = z;
    }
  }
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const a = j * (N + 1) + i, b = a + 1, c = a + N + 1, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  // make the winding face up
  const n = g.attributes.normal;
  if (n.getY(Math.floor(n.count / 2)) < 0) { g.setIndex(idx.map((_, i) => idx[i - (i % 3) + [0, 2, 1][i % 3]])); g.computeVertexNormals(); }
  return g;
}

const terrainMat = new THREE.MeshStandardNodeMaterial();
{
  const p = positionWorld, up = normalWorld.y, h = p.y;
  const n1 = mx_fractal_noise_float(p.mul(0.012), 4);
  const n2 = mx_noise_float(p.mul(0.35));
  const snowMask = smoothstep(0.58, 0.78, up.add(n1.mul(0.2)).add(n2.mul(0.05))).mul(smoothstep(1.4, 2.8, h.add(n2.mul(0.5))));
  const rock = mix(vec3(0.05, 0.048, 0.047), vec3(0.15, 0.14, 0.13), saturate(n1.mul(0.6).add(0.5)));
  const wet = smoothstep(1.7, 0.2, h.add(n2.mul(0.35)));          // splash band above the waterline
  const rockWet = rock.mul(mix(1, 0.55, wet));
  const snowCol = vec3(0.82, 0.86, 0.92).mul(n2.mul(0.04).add(1));
  terrainMat.colorNode = mix(vec3(0.42, 0.42, 0.42), mix(rockWet, snowCol, snowMask), U.snow);
  terrainMat.roughnessNode = mix(float(0.9), mix(mix(float(0.92), float(0.32), wet), float(0.78), snowMask), U.snow);
  // rock strata + wind-carved snow: a bump that breaks up the smooth heightfield shading
  // each band fades out once a pixel covers its wavelength, or the derivatives alias into blocks
  const fpT = length(fwidth(p));
  const strata = mx_fractal_noise_float(p.mul(vec3(0.05, 0.16, 0.05)), 3).mul(1.6).mul(float(1).sub(smoothstep(0.6, 2.5, fpT)))
    .add(mx_noise_float(p.mul(0.45)).mul(0.18).mul(float(1).sub(smoothstep(0.15, 0.6, fpT))));
  terrainMat.normalNode = proceduralBump(strata.mul(mix(float(1), float(0.3), snowMask)), U.snow);
}
const terrain = new THREE.Mesh(buildTerrain(), terrainMat);
terrain.castShadow = true; terrain.receiveShadow = true;
world.add(terrain);

// ─────────────────────────────────────────────────────────────────────────────
// Water: planar reflection distorted by analytic waves, Fresnel, sun glint, footprint-filtered
// ─────────────────────────────────────────────────────────────────────────────
// [direction (rad), wavelength (m), steepness] — spread directions so no single stripe pattern dominates
const WAVES = [[0.25, 13, 0.035], [0.9, 7.7, 0.032], [-0.45, 5.1, 0.034], [1.9, 3.3, 0.03], [-1.2, 2.3, 0.03],
  [0.55, 1.45, 0.028], [2.6, 1.0, 0.026], [-2.1, 0.62, 0.024], [1.3, 0.41, 0.02]];
const waterGeo = new THREE.PlaneGeometry(80000, 80000).rotateX(-Math.PI / 2);
const refl = reflector({ resolutionScale: 0.5 });
refl.target.rotateX(-Math.PI / 2);
world.add(refl.target);
const waterMat = new THREE.MeshBasicNodeMaterial();
{
  const p = positionWorld;
  const fp = max(fwidth(p.x), fwidth(p.z));
  let sx = float(0), sz = float(0);
  for (const [a, lambda, steep] of WAVES) {
    const k = 2 * Math.PI / lambda, w = Math.sqrt(9.81 * k), dx = Math.cos(a), dz = Math.sin(a);
    const phase = p.x.mul(dx * k).add(p.z.mul(dz * k)).sub(U.time.mul(w));
    const fade = float(1).sub(smoothstep(lambda * 0.12, lambda * 0.5, fp));
    const s = cos(phase).mul(steep).mul(fade);
    sx = sx.add(s.mul(dx)); sz = sz.add(s.mul(dz));
  }
  // cat's-paw wind patches: calm, glassy slicks between rougher gusts (the fjord look)
  const gust = smoothstep(-0.35, 0.45, mx_fractal_noise_float(vec3(p.x.mul(0.006), U.time.mul(0.015), p.z.mul(0.01)), 3));
  const amp = mix(float(0.18), float(1.3), gust);
  const n = normalize(vec3(sx.mul(amp).negate(), 1, sz.mul(amp).negate()));
  const v = normalize(cameraPosition.sub(p));
  refl.uvNode = refl.uvNode.add(clamp(n.xz.mul(0.05), -0.012, 0.012));
  const fres = float(0.02).add(pow(float(1).sub(saturate(dot(n, v))), 5).mul(0.98));
  const deep = vec3(0.006, 0.024, 0.032).mul(U.skyAmb);
  const hv = normalize(U.sunDir.add(v));
  const glint = pow(saturate(dot(n, hv)), 900).mul(70).add(pow(saturate(dot(n, hv)), 60).mul(0.5));
  const spec = U.sunColor.mul(glint).min(40);                  // HDR safety: never let a highlight reach Inf
  waterMat.colorNode = mix(deep, refl.rgb, fres).add(spec);
}
const water = new THREE.Mesh(waterGeo, waterMat);
world.add(water);
const waterBasic = new THREE.Mesh(waterGeo, new THREE.MeshStandardNodeMaterial({ color: 0x2b4a63, roughness: 0.3 }));
waterBasic.receiveShadow = true;
waterBasic.visible = false;
world.add(waterBasic);

// ─────────────────────────────────────────────────────────────────────────────
// Ice: floes (extruded, bevelled or not) and a few icebergs
// ─────────────────────────────────────────────────────────────────────────────
const iceMat = new THREE.MeshStandardNodeMaterial();
{
  const p = positionWorld;
  const top = smoothstep(0.5, 0.85, normalWorld.y);
  const striae = mx_noise_float(p.mul(vec3(0.25, 2.2, 0.25))).mul(0.06);
  const sideCol = mix(vec3(0.07, 0.33, 0.44), vec3(0.6, 0.76, 0.84), float(1).sub(exp(max(p.y, 0).mul(-0.25)))).add(striae);
  const drift = mx_fractal_noise_float(p.mul(0.45), 3);
  const topCol = mix(vec3(0.62, 0.74, 0.84), vec3(0.88, 0.92, 0.96), smoothstep(-0.4, 0.3, drift));
  const v = normalize(cameraPosition.sub(p));
  const back = pow(saturate(dot(v.negate(), U.sunDir)), 6);
  const sss = vec3(0.08, 0.42, 0.52).mul(back.mul(0.55).add(0.02)).mul(float(1).sub(top)).mul(U.sunColor);
  iceMat.colorNode = mix(vec3(0.8, 0.8, 0.8), mix(sideCol, topCol, top), U.ice);
  iceMat.roughnessNode = mix(float(0.55), mix(float(0.12), float(0.62), top), U.ice);
  iceMat.emissiveNode = sss.mul(U.ice);
}

const floes = [];
function floeShape(r) {
  const n = 9 + Math.floor(rng() * 6), pts = [];
  for (let i = 0; i < n; i++) {
    const a = i / n * Math.PI * 2 + (rng() - 0.5) * 0.35;
    const rr = r * (0.65 + rng() * 0.5);
    pts.push(new THREE.Vector2(Math.cos(a) * rr, Math.sin(a) * rr));
  }
  return new THREE.Shape(pts);
}
function floeGeo(shape, th, bevel) {
  const g = new THREE.ExtrudeGeometry(shape, bevel
    ? { depth: th, bevelEnabled: true, bevelThickness: 0.22, bevelSize: 0.35, bevelSegments: 3, curveSegments: 1 }
    : { depth: th, bevelEnabled: false, curveSegments: 1 });
  g.rotateX(-Math.PI / 2);          // extrusion now runs along +y (0 … th)
  g.translate(0, -th, 0);           // top at y = 0
  return g;
}
for (let i = 0; i < 95; i++) {
  // more floes near the camera paths, thinning out down the fjord
  const z = 380 - Math.pow(rng(), 1.6) * 1500;
  const d = (rng() * 2 - 1) * (halfW(z) - 45);
  const x = cx(z) + d;
  // keep clear of the fixed camera positions (a floe right under the lens reads as a giant slab)
  if (Math.hypot(x - (VX + 120), z - (VZ + 150)) < 45 || Math.hypot(x - (VX + 95), z - (VZ + 120)) < 40) continue;
  const r = 2 + Math.pow(rng(), 2) * 11;
  const th = 0.8 + rng() * 1.2;
  const shape = floeShape(r);
  const gB = floeGeo(shape, th, true), gS = floeGeo(shape, th, false);
  const m = new THREE.Mesh(gB, iceMat);
  m.castShadow = true; m.receiveShadow = true;
  m.userData = { x, z, gB, gS, free: 0.16 + rng() * 0.12, ph: rng() * 6.28, rot: rng() * 6.28, drift: 0.12 + rng() * 0.15 };
  world.add(m);
  floes.push(m);
}
// a cluster close to the "ice" and "water" viewpoints so they have something to look at
for (let i = 0; i < 14; i++) {
  const z = 225 - rng() * 140, x = VX + 95 + rng() * 120;
  const r = 1.2 + rng() * 4, th = 0.6 + rng() * 0.8, shape = floeShape(r);
  const gB = floeGeo(shape, th, true), gS = floeGeo(shape, th, false);
  const m = new THREE.Mesh(gB, iceMat);
  m.castShadow = true; m.receiveShadow = true;
  m.userData = { x, z, gB, gS, free: 0.14 + rng() * 0.1, ph: rng() * 6.28, rot: rng() * 6.28, drift: 0.06 + rng() * 0.08 };
  world.add(m);
  floes.push(m);
}

// icebergs: a coarse, non-indexed icosphere pushed around by noise, then cut into flat-shaded facets and a flat top
function icebergGeo(rx, ry, rz) {
  const g = new THREE.IcosahedronGeometry(1, 2);
  const p = g.attributes.position, s = rng() * 100;
  const cache = new Map();
  for (let i = 0; i < p.count; i++) {
    const key = `${p.getX(i).toFixed(4)},${p.getY(i).toFixed(4)},${p.getZ(i).toFixed(4)}`;
    if (!cache.has(key)) {
      let x = p.getX(i), y = p.getY(i), z = p.getZ(i);
      const n = fbm(x * 1.4 + s, z * 1.4 + y * 1.1, 4);
      const k = 0.55 + n * 1.0 + (hash2(x * 9 + s, z * 9 + y) - 0.5) * 0.25;
      x *= rx * k; z *= rz * k;
      y = y > 0 ? Math.min(y * ry * (0.8 + n * 0.8), ry * (0.75 + 0.3 * fbm(x * 0.05 + s, z * 0.05))) : y * ry * 1.2;
      cache.set(key, [x, y, z]);
    }
    p.setXYZ(i, ...cache.get(key));
  }
  g.computeVertexNormals();          // non-indexed → one normal per face: crisp facets
  return g;
}
for (const [dz, dd, rx, ry, rz] of [[-260, 90, 24, 18, 17], [-520, -120, 32, 26, 22], [-900, 60, 22, 14, 30], [40, 150, 9, 7, 7]]) {
  const z = VZ + dz, x = cx(z) + dd;
  const m = new THREE.Mesh(icebergGeo(rx, ry, rz), iceMat);
  m.position.set(x, 0, z);
  m.rotation.y = rng() * 6.28;
  m.castShadow = true; m.receiveShadow = true;
  world.add(m);
}

// ─────────────────────────────────────────────────────────────────────────────
// Village: rorbu cabins with gable ends to the water, warm windows, stilts, a jetty
// ─────────────────────────────────────────────────────────────────────────────
const bevelSwaps = [];           // { mesh, gB, gS }
const winMat = new THREE.MeshStandardNodeMaterial({ roughness: 0.2 });
winMat.colorNode = mix(vec3(0.04, 0.05, 0.06), vec3(0.06, 0.04, 0.02), U.lights);
winMat.emissiveNode = vec3(1.0, 0.52, 0.2).mul(U.lights.mul(7));
const trimMat = new THREE.MeshStandardNodeMaterial({ color: 0xe8e4dc, roughness: 0.6 });
const snowRoofMat = new THREE.MeshStandardNodeMaterial({ color: 0xdfe6ee, roughness: 0.8 });
const woodMat = new THREE.MeshStandardNodeMaterial({ color: 0x2b2018, roughness: 0.9 });
const wallMats = [0x5c0d08, 0x6b1209, 0x8a5a17, 0x540c0a, 0x6e6152].map((c) => new THREE.MeshStandardNodeMaterial({ color: c, roughness: 0.78 }));
const pointLights = [];

function swapMesh(gB, gS, mat) {
  const m = new THREE.Mesh(gB, mat);
  m.castShadow = true; m.receiveShadow = true;
  bevelSwaps.push({ mesh: m, gB, gS });
  return m;
}
function roofGeo(w, d, h, bevel) {
  // gable roof: triangle in (z, y), extruded along x (the ridge runs toward the water)
  const s = new THREE.Shape([new THREE.Vector2(-d / 2, 0), new THREE.Vector2(d / 2, 0), new THREE.Vector2(0, h)]);
  const g = new THREE.ExtrudeGeometry(s, bevel
    ? { depth: w, bevelEnabled: true, bevelThickness: 0.08, bevelSize: 0.08, bevelSegments: 2 }
    : { depth: w, bevelEnabled: false });
  g.rotateY(Math.PI / 2);           // extrude along +x
  g.translate(-w / 2, 0, 0);
  return g;
}

function cabin(x, z, yaw, wallMat) {
  const ground = terrainHeight(x, z);
  const floorY = Math.max(ground, 0) + 1.4;
  const w = 8.5, d = 5.6, h = 4.2;          // w along x (toward the water), d across
  const g = new THREE.Group();
  g.position.set(x, floorY, z);
  g.rotation.y = yaw;
  const body = swapMesh(new RoundedBoxGeometry(w, h, d, 3, 0.14), new THREE.BoxGeometry(w, h, d), wallMat);
  body.position.y = h / 2;
  g.add(body);
  const rB = roofGeo(w + 0.9, d + 1.0, 2.6, true), rS = roofGeo(w + 0.9, d + 1.0, 2.6, false);
  const roof = swapMesh(rB, rS, woodMat);
  roof.position.y = h;
  g.add(roof);
  const sB = roofGeo(w + 0.95, d + 1.1, 2.62, true), sS = roofGeo(w + 0.95, d + 1.1, 2.62, false);
  const snowCap = swapMesh(sB, sS, snowRoofMat);
  snowCap.position.y = h + 0.12;
  g.add(snowCap);
  // gable end facing the water (+x): two windows below, one in the gable, white frames
  const face = w / 2 + 0.03;
  for (const [wz, wy, ww, wh] of [[-1.3, 1.6, 1.0, 1.25], [1.3, 1.6, 1.0, 1.25], [0, 4.7, 0.9, 1.0]]) {
    const frame = new THREE.Mesh(new THREE.BoxGeometry(0.06, wh + 0.24, ww + 0.24), trimMat);
    frame.position.set(face, wy, wz);
    const pane = new THREE.Mesh(new THREE.BoxGeometry(0.08, wh, ww), winMat);
    pane.position.set(face + 0.01, wy, wz);
    g.add(frame, pane);
  }
  // side windows
  for (const sx of [-2.2, 2.2]) {
    const pane = new THREE.Mesh(new THREE.BoxGeometry(1.0, 1.1, 0.08), winMat);
    pane.position.set(sx, 2.2, d / 2 + 0.02);
    g.add(pane);
  }
  // corner trims
  for (const [tx, tz] of [[w / 2, d / 2], [w / 2, -d / 2], [-w / 2, d / 2], [-w / 2, -d / 2]]) {
    const t = new THREE.Mesh(new THREE.BoxGeometry(0.16, h, 0.16), trimMat);
    t.position.set(tx, h / 2, tz);
    g.add(t);
  }
  // stilts down to the rock / sea bed
  const legH = floorY - Math.min(ground, -2) + 0.2;
  for (const lx of [-w / 2 + 0.4, 0, w / 2 - 0.4]) for (const lz of [-d / 2 + 0.4, d / 2 - 0.4]) {
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.16, legH, 8), woodMat);
    leg.position.set(lx, -legH / 2 + 0.1, lz);
    leg.castShadow = true;
    g.add(leg);
  }
  world.add(g);
  return g;
}

{
  const vr = mulberry32(5);
  const spots = [];
  for (let i = 0; i < 9; i++) {
    const z = VZ - 120 + i * 30 + (vr() - 0.5) * 8;
    const x = cx(z) - halfW(z) - 6 - vr() * 22 - (i % 3 === 1 ? 26 : 0);
    spots.push([x, z]);
  }
  spots.forEach(([x, z], i) => {
    const g = cabin(x, z, (vr() - 0.5) * 0.25, wallMats[i % wallMats.length]);
    if (i % 2 === 0) {
      const L = new THREE.PointLight(0xffa45c, 0, 40, 2);
      L.position.set(x + 6.5, g.position.y + 2.2, z);
      world.add(L);
      pointLights.push(L);
    }
  });
  // a jetty running out into the water
  const jz = VZ + 12, jx = cx(jz) - halfW(jz) + 2;
  const deck = swapMesh(new RoundedBoxGeometry(26, 0.25, 2.4, 2, 0.06), new THREE.BoxGeometry(26, 0.25, 2.4), woodMat);
  deck.position.set(jx + 13, 1.3, jz);
  world.add(deck);
  for (let i = 0; i < 7; i++) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 6, 8), woodMat);
    post.position.set(jx + 1 + i * 4, -1.6, jz + (i % 2 ? 1.1 : -1.1));
    post.castShadow = true;
    world.add(post);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spruce forest on the gentler lower slopes (instanced; snow settles on the upward-facing tiers)
// ─────────────────────────────────────────────────────────────────────────────
{
  const tiers = [[1.9, 3.2, 1.6], [1.45, 2.8, 3.3], [1.0, 2.5, 4.9], [0.55, 2.0, 6.3]].map(([r, h, y]) => new THREE.ConeGeometry(r, h, 7, 1).translate(0, y, 0));
  const trunk = new THREE.CylinderGeometry(0.16, 0.22, 1.8, 6).translate(0, 0.9, 0);
  const treeGeo = mergeGeometries([trunk, ...tiers].map((g) => g.toNonIndexed()));
  treeGeo.computeVertexNormals();
  const treeMat = new THREE.MeshStandardNodeMaterial({ roughness: 0.85 });
  const nz = mx_noise_float(positionWorld.mul(1.7));
  const snowy = smoothstep(0.25, 0.7, normalWorld.y.add(nz.mul(0.25)));
  treeMat.colorNode = mix(vec3(0.08, 0.22, 0.1), mix(vec3(0.018, 0.04, 0.03), vec3(0.78, 0.83, 0.9), snowy), U.snow);
  const tr = mulberry32(23), mats = [];
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sv = new THREE.Vector3(), pv = new THREE.Vector3(), yAxis = new THREE.Vector3(0, 1, 0);
  for (let tries = 0; tries < 30000 && mats.length < 1600; tries++) {
    const z = VZ + 700 - tr() * 2600;
    const side = tr() < 0.5 ? -1 : 1;
    const d = halfW(z) + 18 + Math.pow(tr(), 1.5) * 420;
    const x = cx(z) + side * d;
    if (Math.abs(z - VZ) < 170 && side < 0 && d < halfW(z) + 70) continue;    // keep the village clearing open
    const h = terrainHeight(x, z);
    if (h < 3 || h > 260) continue;
    const e = 2;
    const slope = Math.hypot(terrainHeight(x + e, z) - terrainHeight(x - e, z), terrainHeight(x, z + e) - terrainHeight(x, z - e)) / (2 * e);
    if (slope > 0.55 || tr() > 1.1 - h / 300) continue;
    const s = 0.6 + tr() * 0.9;
    q.setFromAxisAngle(yAxis, tr() * 6.28);
    mats.push(m4.compose(pv.set(x, h - 0.4, z), q, sv.set(s, s * (0.85 + tr() * 0.4), s)).clone());
  }
  const forest = new THREE.InstancedMesh(treeGeo, treeMat, mats.length);
  mats.forEach((m, i) => forest.setMatrixAt(i, m));
  forest.castShadow = true; forest.receiveShadow = true;
  world.add(forest);
}

// ─────────────────────────────────────────────────────────────────────────────
// Snowfall: instanced quads wrapped in a box that follows the camera
// ─────────────────────────────────────────────────────────────────────────────
const SNOW_N = 6000, BOX = 60;
const camRight = uniform(new THREE.Vector3(1, 0, 0)), camUp = uniform(new THREE.Vector3(0, 1, 0)), camPos = uniform(new THREE.Vector3());
const snowGeo = new THREE.InstancedBufferGeometry().copy(new THREE.PlaneGeometry(1, 1));
snowGeo.instanceCount = SNOW_N;
const snowMat = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
{
  const id = float(instanceIndex);
  const seed = vec3(hash(id), hash(id.add(17.31)), hash(id.add(91.7)));
  const t = U.time;
  const drift = vec3(sin(t.mul(0.4).add(seed.x.mul(40))).mul(0.8).add(t.mul(0.5)), t.mul(seed.y.mul(0.5).add(0.9)).negate(), cos(t.mul(0.33).add(seed.z.mul(40))).mul(0.8));
  const p = mod(seed.mul(BOX).add(drift).sub(camPos), BOX).add(camPos).sub(BOX / 2);
  const size = seed.x.mul(0.035).add(0.025);
  snowMat.positionNode = p.add(camRight.mul(positionLocal.x.mul(size))).add(camUp.mul(positionLocal.y.mul(size)));
  const r = length(uv().sub(0.5));
  snowMat.colorNode = vec3(0.75, 0.82, 0.92).mul(U.skyAmb).mul(0.9);
  snowMat.opacityNode = float(1).sub(smoothstep(0.2, 0.5, r)).mul(0.85);
}
const snowfall = new THREE.Mesh(snowGeo, snowMat);
snowfall.frustumCulled = false;
world.add(snowfall);

// ─────────────────────────────────────────────────────────────────────────────
// Post: chromatic aberration → warm bloom → exposure / white balance → tone map → grade → vignette, grain
// ─────────────────────────────────────────────────────────────────────────────
const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera, { samples: 4 });
const hdr = scenePass.getTextureNode();
const glow = bloom(hdr, 0.3, 0.55, 1.05);
glow.strength = U.bloom;
const WB = vec3(1.08, 0.98, 0.9);
pipeline.outputNode = Fn(() => {
  const uvc = screenUV.sub(0.5);
  const ca = uvc.mul(U.ca);
  const col = vec3(hdr.sample(screenUV.sub(ca)).r, hdr.sample(screenUV).g, hdr.sample(screenUV.add(ca)).b);
  const lit = col.add(glow.rgb.mul(vec3(1.0, 0.82, 0.6))).mul(U.exposure).mul(mix(vec3(1), WB, U.grade));
  const one = float(1);
  // operator picked by weights rather than nested select() (that chain came out black for "none");
  // "none" passes HDR through and lets the canvas clip it
  const w = (k) => max(float(1).sub(U.tone.sub(k).abs()), 0);
  const tm = agxToneMapping(lit, one).mul(w(1)).add(neutralToneMapping(lit, one).mul(w(2)))
    .add(acesFilmicToneMapping(lit, one).mul(w(3))).add(lit.mul(w(0)));
  let o = sRGBTransferOETF(tm);
  // grade (Pond Line): saturation back, golden highlights / teal shade, soft S-curve
  const l = luminance(o);
  let g = mix(vec3(l), o, 1.22);
  g = g.add(vec3(0.05, 0.025, -0.03).mul(smoothstep(0.35, 1.0, l)));
  g = g.add(vec3(-0.015, 0.01, 0.028).mul(pow(float(1).sub(l), 2)));
  g = mix(g, smoothstep(0.0, 1.0, g), 0.3);
  o = mix(o, g, U.grade);
  // lens (Minamo): vignette and per-frame grain
  o = o.mul(float(1).sub(pow(length(uvc.mul(vec2(1.0, 0.85))).mul(1.35), 2.6).mul(U.vig)));
  const nz = hash(screenUV.x.mul(1973.1).add(screenUV.y.mul(9277.3)).add(U.frame.mul(26.699))).sub(0.5);
  o = o.add(nz.mul(U.grain).mul(2));
  return vec4(o, 1);
})();
pipeline.outputColorTransform = false;

// ─────────────────────────────────────────────────────────────────────────────
// State & switches
// ─────────────────────────────────────────────────────────────────────────────
const TONES = { none: 0, agx: 1, neutral: 2, aces: 3 };
const state = { ...DATA.presets.find((p) => p.id === 'full').state };
const extras = { bloomStrength: 0.3, ca: 0.0016 };

function apply() {
  const s = state;
  // sky light vs. flat ambient
  sky.visible = s.sky;
  scene.background = s.sky ? null : FLAT_BG;
  scene.environmentIntensity = s.sky ? 1 : 0;
  ambient.intensity = s.sky ? 0 : 1.1;
  sun.intensity = s.sky ? 3.2 : 2.2;
  U.skyAmb.value = s.sky ? 1 : 2.5;
  // shadows: switching castShadow would rebuild every material, so the shadow map is kept and simply emptied
  sun.shadow.intensity = s.shadow ? 1 : 0;
  U.snow.value = s.snow ? 1 : 0;
  water.visible = s.water; waterBasic.visible = !s.water;
  U.ice.value = s.ice ? 1 : 0;
  for (const b of bevelSwaps) b.mesh.geometry = s.bevel ? b.gB : b.gS;
  for (const f of floes) f.geometry = s.bevel ? f.userData.gB : f.userData.gS;
  U.lights.value = s.lights ? 1 : 0;
  for (const L of pointLights) L.intensity = s.lights ? 450 : 0;
  U.fog.value = s.fog ? 1 : 0;
  snowfall.visible = s.motion;
  U.bloom.value = s.bloom ? (s.bloomStrength ?? extras.bloomStrength) : 0;
  U.tone.value = s.tonemap ? TONES[s.toneMode || 'agx'] : 0;
  U.exposure.value = s.tonemap ? 0.82 : 0.7;
  U.grade.value = s.grade ? 1 : 0;
  U.vig.value = s.lens ? 0.25 : 0;
  U.grain.value = s.lens ? 0.012 : 0;
  U.ca.value = s.lens ? (s.ca ?? extras.ca) : 0;
  syncUI();
}

// ─────────────────────────────────────────────────────────────────────────────
// Views
// ─────────────────────────────────────────────────────────────────────────────
const VIEWS = {
  village: { pos: [VX + 95, 7, VZ + 120], at: [VX - 8, 5, VZ + 10] },
  fjord: { pos: [VX + 120, 70, VZ + 420], at: [cx(-1400), 120, -1400] },
  water: { pos: [VX + 120, 1.6, VZ + 150], at: [VX + 160, 0, VZ - 400] },
  ice: { pos: [VX + 112, 3.2, VZ + 98], at: [VX + 125, -0.5, VZ + 60] },
};
let currentView = 'village';
function setView(id) {
  const v = VIEWS[id]; currentView = id;
  camera.position.set(...v.pos);
  controls.target.set(...v.at);
  controls.update();
  syncUI();
}

// ─────────────────────────────────────────────────────────────────────────────
// UI
// ─────────────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const presetsEl = $('presets'), viewsEl = $('views'), techsEl = $('techs'), infoEl = $('info');
let focusId = null, currentPreset = 'full';

for (const p of DATA.presets) {
  const b = document.createElement('button');
  b.className = 'chip'; b.textContent = p.name.replace(/（.*）/, ''); b.dataset.id = p.id; b.title = p.desc;
  b.onclick = () => { usePreset(p.id); showPreset(p); };
  presetsEl.append(b);
}
for (const v of DATA.views) {
  const b = document.createElement('button');
  b.className = 'chip'; b.textContent = v.name; b.dataset.id = v.id;
  b.onclick = () => setView(v.id);
  viewsEl.append(b);
}
DATA.techniques.forEach((t, i) => {
  const row = document.createElement('label');
  row.className = 'row'; row.dataset.id = t.id;
  const cb = document.createElement('input'); cb.type = 'checkbox';
  cb.onchange = () => { state[t.id] = cb.checked; currentPreset = null; apply(); showTech(t); };
  const name = document.createElement('span'); name.textContent = t.name;
  const right = document.createElement('span'); right.className = 'num'; right.textContent = String(i + 1).padStart(2, '0');
  row.append(cb, name, right);
  if (t.id === 'tonemap') {
    const sel = document.createElement('select');
    for (const [k, label] of [['agx', 'AgX'], ['neutral', 'Neutral'], ['aces', 'ACES']]) sel.add(new Option(label, k));
    sel.onclick = (e) => e.preventDefault();
    sel.onchange = () => { state.toneMode = sel.value; currentPreset = null; apply(); };
    right.replaceWith(sel);
  }
  row.onmouseenter = () => showTech(t);
  techsEl.append(row);
});
const actions = document.createElement('div');
actions.className = 'actions';
actions.innerHTML = '<button class="btn" id="genPrompt">依目前勾選產生 prompt</button>';
techsEl.after(actions);
$('genPrompt').onclick = () => {
  const on = DATA.techniques.filter((t) => state[t.id]);
  const text = ['目標：用 three.js r186（WebGPURenderer + TSL）做一個藍調時刻的冰峽灣場景：雪山、冰海、漂流浮冰、亮著暖色窗燈的漁村小屋。單位一律用公尺。',
    '請依下面的技巧分階段實作，每個階段都截圖，和參考圖比對後列出 3 個最大差異並修正：',
    ...on.map((t, i) => `${i + 1}. 【${t.name}】${t.prompt}`)].join('\n');
  navigator.clipboard?.writeText(text).then(() => toast('prompt 已複製到剪貼簿'), () => {});
  infoEl.innerHTML = `<div class="label">依目前勾選產生的 PROMPT</div><p class="prompt">${escapeHtml(text)}</p>`;
};
$('toggle').onclick = () => $('panel').classList.remove('hidden');
addEventListener('keydown', (e) => { if (e.key === 'h' || e.key === 'H') $('panel').classList.toggle('hidden'); });

function escapeHtml(s) { return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('show'); setTimeout(() => t.classList.remove('show'), 1600); }
function showTech(t) {
  focusId = t.id;
  infoEl.innerHTML = `<b>${t.name}</b><div class="src">出處：${escapeHtml(t.source)}</div><p class="what">${escapeHtml(t.what)}</p>`
    + `<div class="label">下次這樣提示我</div><p class="prompt">${escapeHtml(t.prompt)}</p>`;
  for (const r of techsEl.children) r.classList.toggle('focus', r.dataset.id === t.id);
}
function showPreset(p) {
  infoEl.innerHTML = `<b>${p.name}</b><p class="what">${escapeHtml(p.desc)}</p><div class="label">風格 PROMPT</div><p class="prompt">${escapeHtml(p.prompt)}</p>`;
}
function usePreset(id) {
  const p = DATA.presets.find((x) => x.id === id);
  for (const k of Object.keys(state)) delete state[k];
  Object.assign(state, p.state);
  currentPreset = id;
  apply();
}
function syncUI() {
  for (const b of presetsEl.children) b.classList.toggle('on', b.dataset.id === currentPreset);
  for (const b of viewsEl.children) b.classList.toggle('on', b.dataset.id === currentView);
  for (const r of techsEl.children) {
    r.querySelector('input').checked = !!state[r.dataset.id];
    const sel = r.querySelector('select'); if (sel) sel.value = state.toneMode || 'agx';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop
// ─────────────────────────────────────────────────────────────────────────────
const timer = new THREE.Timer();
let simT = 37, frozen = false, fpsAcc = 0, fpsN = 0;
const hud = $('hud');
function frame(now) {
  timer.update(now);
  const dt = Math.min(timer.getDelta(), 0.1);
  if (!frozen && state.motion) simT += dt;
  U.time.value = simT;
  U.frame.value = (U.frame.value + 1) % 1000;
  for (const f of floes) {
    const u = f.userData;
    const z = u.z - ((simT * u.drift) % 400);
    f.position.set(u.x + Math.sin(simT * 0.05 + u.ph) * 2, u.free + Math.sin(simT * 0.9 + u.ph) * 0.035, z < -1200 ? z + 400 : z);
    f.rotation.set(Math.sin(simT * 0.7 + u.ph) * 0.012, u.rot + simT * 0.004 * (u.ph > 3 ? 1 : -1), Math.cos(simT * 0.6 + u.ph) * 0.012);
  }
  controls.update();
  camera.updateMatrixWorld();
  camRight.value.setFromMatrixColumn(camera.matrixWorld, 0);
  camUp.value.setFromMatrixColumn(camera.matrixWorld, 1);
  camPos.value.copy(camera.position);
  // shadow frustum follows what the camera looks at
  const focus = controls.target;
  sun.target.position.set(focus.x, 0, focus.z);
  sun.position.copy(sun.target.position).addScaledVector(SUN_DIR, 3000);
  pipeline.render();
  if (pendingShot) { const s = pendingShot; if (--s.wait <= 0) { pendingShot = null; s.resolve(canvas.toDataURL('image/jpeg', 0.9)); } }
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { hud.textContent = `${Math.round(fpsN / fpsAcc)} FPS　·　拖曳旋轉、滾輪縮放、H 隱藏面板`; fpsAcc = 0; fpsN = 0; }
}

function resize() {
  if (captureSize) return;
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

// ─────────────────────────────────────────────────────────────────────────────
// Capture API (used to build the PDF): fixed size, frozen time, posts JPEGs to tools/serve.py
// ─────────────────────────────────────────────────────────────────────────────
let pendingShot = null, captureSize = null;
function grab(wait = 12) { return new Promise((resolve) => { pendingShot = { wait, resolve }; }); }
window.fjord = {
  state, apply, setView, usePreset, VIEWS, U,
  set(partial) { Object.assign(state, partial); currentPreset = null; apply(); },
  async captureBegin(w = 1600, h = 900) {
    captureSize = [w, h]; frozen = true; simT = 37; document.body.classList.add('capture');
    renderer.setPixelRatio(1); renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  },
  captureEnd() {
    captureSize = null; frozen = false; document.body.classList.remove('capture');
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); resize();
  },
  async shot(name, wait = 12) {
    const url = await grab(wait);
    const r = await fetch(`/save?name=${name}`, { method: 'POST', body: url });
    return r.ok;
  },
  // jobs: [{ name, view, preset?, state? }]
  async batch(jobs) {
    const done = [];
    for (const j of jobs) {
      if (j.preset) usePreset(j.preset);
      if (j.state) { Object.assign(state, j.state); apply(); }
      setView(j.view);
      if (await this.shot(j.name, j.wait ?? 14)) done.push(j.name);
    }
    return done;
  },
};

usePreset('full');
setView('village');
showTech(DATA.techniques[0]);
renderer.setAnimationLoop(frame);
setTimeout(() => loaderEl.classList.add('gone'), 400);
