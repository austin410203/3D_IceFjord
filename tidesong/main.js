// 潮聲島・渲染技巧實驗室
// Tidesong Isle (the Unity cozy-fishing game) rebuilt as a web scene — village, jetty, turquoise lagoon, river and
// waterfall — to show what the Minamo / Pond Line techniques do to it. Every switch is a uniform or a visibility flag.
import * as THREE from 'three/webgpu';
import {
  Fn, uniform, float, vec2, vec3, vec4, mix, sin, cos, max, min, clamp, pow, exp, dot, normalize, length, smoothstep,
  fwidth, reflector, positionWorld, positionLocal, normalWorld, cameraPosition, screenUV, mx_noise_float,
  mx_fractal_noise_float, mx_worley_noise_vec2, pass, luminance, hash, instanceIndex, agxToneMapping,
  neutralToneMapping, acesFilmicToneMapping, sRGBTransferOETF, saturate, fog, uv, dFdx, dFdy, cross, sign, abs,
  positionView, normalView, texture, refract, attribute, perspectiveDepthToViewZ, viewportSize,
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const DATA = await (await fetch('./techniques.json')).json();

// bump from a procedural height (three's bumpMap() only takes textures): Mikkelsen's surface-gradient perturbation
function proceduralBump(H, scale) {
  const sx = dFdx(positionView), sy = dFdy(positionView), n = normalView;
  const r1 = cross(sy, n), r2 = cross(n, sx);
  const det = dot(sx, r1);
  const grad = sign(det).mul(r1.mul(dFdx(H)).add(r2.mul(dFdy(H)))).mul(scale);
  return normalize(abs(det).mul(n).sub(grad));
}

// ─────────────────────────────────────────────────────────────────────────────
// Island layout (metres). Sea level y = 0, sea to the south (+z), the mountain and waterfall cliff to the north.
// ─────────────────────────────────────────────────────────────────────────────
function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function smooth(a, b, x) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
function hash2(x, y) { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); }
function vnoise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x, y, oct = 5) { let s = 0, a = 0.5, f = 1; for (let i = 0; i < oct; i++) { s += a * vnoise(x * f, y * f); a *= 0.5; f *= 2.03; } return s; }
function ridged(x, y) { let s = 0, a = 0.55, f = 1; for (let i = 0; i < 5; i++) { const n = 1 - Math.abs(vnoise(x * f, y * f) * 2 - 1); s += a * n * n; a *= 0.5; f *= 2.1; } return s; }
function segDist(px, pz, ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az, t = Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / (dx * dx + dz * dz)));
  return Math.hypot(px - ax - dx * t, pz - az - dz * t);
}
function polyDist(px, pz, pts) { let d = 1e9; for (let i = 0; i < pts.length - 1; i++) d = Math.min(d, segDist(px, pz, ...pts[i], ...pts[i + 1])); return d; }

const RIVER = [[-15, -50], [-6, -24], [14, 6], [34, 40], [52, 80], [64, 118], [72, 160]];
const PATHS = [[[-20, 95], [-45, 140]], [[-20, 95], [30, 70]], [[-20, 95], [-70, 70]], [[-20, 95], [5, 40], [-8, -30]]];
const PLAZA = [-20, 95];
const BAY = [-72, 162];
const FALL = [-15, -57];            // waterfall lip (x, z); top pool level 25.2

function terrainHeight(x, z) {
  // coastline: a noisy radius around the island centre
  const cx0 = 0, cz0 = -40;
  const ang = Math.atan2(z - cz0, x - cx0), r = Math.hypot(x - cx0, z - cz0);
  const R = 190 + (fbm(Math.cos(ang) * 1.6 + 5, Math.sin(ang) * 1.6 + 5, 4) - 0.5) * 70;
  const inland = R - r;
  // land: gentle rise from the beach; sea: a shallow shelf, then the drop-off
  // one continuous profile through the waterline (a step here reads as a jagged rock rim)
  let h = Math.max(-30, Math.min(inland, 20)) * 0.1 + Math.max(0, inland - 20) * 0.035 - Math.max(0, -inland - 30) * 0.22;
  h = Math.max(h, -28);
  // upper terrace behind the waterfall cliff, and the mountain on it
  const terr = smooth(-48, -64, z + (fbm(x * 0.03, 3) - 0.5) * 16) * smooth(-10, 25, inland);
  h += 22 * terr;
  const mtn = Math.max(0, 1 - Math.hypot(x + 60, z + 135) / 120);
  h += mtn * mtn * (70 + 60 * ridged(x * 0.012, z * 0.012)) * smooth(-10, 20, inland);
  // rolling meadow detail
  h += (fbm(x * 0.02, z * 0.02, 4) - 0.5) * 6 * smooth(0, 30, inland) * (1 - terr);
  // village plaza: flatten
  const dp = Math.hypot(x - PLAZA[0], z - PLAZA[1]);
  h = h + (3.4 - h) * (1 - smooth(26, 46, dp));
  // shallow bay by the jetty
  const db = Math.hypot((x - BAY[0]) * 0.8, z - BAY[1]);
  h = Math.min(h, h + (-1.6 - h) * (1 - smooth(30, 62, db)));
  // pool on the terrace above the fall
  const dpool = Math.hypot(x - FALL[0], (z - FALL[1] + 9) * 0.8);
  h = Math.min(h, 24.4 + dpool * 0.12 + Math.max(0, dpool - 9) * 0.6);
  // river from the plunge pool to the sea, at sea level (all water shares one surface, as in the game)
  const dr = polyDist(x, z, RIVER);
  const bed = -1.7 + Math.min(dr, 6) * 0.15;
  h = Math.min(h, mix1(bed, h, smooth(5, 13, dr)));
  const dplunge = Math.hypot(x - FALL[0], z - FALL[1] - 7);
  h = Math.min(h, mix1(-2.2, h, smooth(6, 14, dplunge)));
  return h;
}
function mix1(a, b, t) { return a + (b - a) * t; }

// ─────────────────────────────────────────────────────────────────────────────
// Renderer, scene, camera
// ─────────────────────────────────────────────────────────────────────────────
const canvas = document.getElementById('view');
const loaderEl = document.getElementById('loader');
if (!navigator.gpu) { loaderEl.innerHTML = '這個場景需要 WebGPU<br><small>請用新版 Chrome / Edge</small>'; throw new Error('WebGPU unavailable'); }
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
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.3, 60000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;

const U = {
  time: uniform(0), sunDir: uniform(new THREE.Vector3(0, 1, 0)), sunColor: uniform(new THREE.Vector3(1, 1, 1)),
  sunI: uniform(3), skyAmb: uniform(new THREE.Vector3(0.5, 0.6, 0.8)), glow: uniform(0.15),
  ground: uniform(1), caustics: uniform(1), foam: uniform(1), lights: uniform(1), fog: uniform(1),
  fogBlue: uniform(new THREE.Vector3()), fogSun: uniform(new THREE.Vector3()), fogDens: uniform(0.0006),
  bloom: uniform(0.3), tone: uniform(1), grade: uniform(1), vig: uniform(0.25), grain: uniform(0.012), ca: uniform(0.0016),
  outline: uniform(0), exposure: uniform(1), frame: uniform(0), near: uniform(0.3), far: uniform(60000),
};

// ─────────────────────────────────────────────────────────────────────────────
// Time of day: sun, sky and the colours that go with it
// ─────────────────────────────────────────────────────────────────────────────
const TOD = {
  // the sun sits west-south-west: side light on the village, straight into the "sunset" view
  day: { elev: 38, az: 2.75, sun: [1.0, 0.95, 0.88], sunI: 3.0, exp: 0.3, tur: 2.2, ray: 1.0, fogB: [0.55, 0.66, 0.82], fogS: [0.9, 0.86, 0.78], dens: 0.00022, glow: 0.12, moon: false },
  golden: { elev: 3.5, az: 2.75, sun: [1.0, 0.6, 0.34], sunI: 3.4, exp: 0.75, tur: 4.0, ray: 3.0, fogB: [0.26, 0.31, 0.44], fogS: [0.8, 0.46, 0.28], dens: 0.00028, glow: 0.55, moon: false },
  blue: { elev: -0.8, az: 2.75, sun: [0.55, 0.65, 1.0], sunI: 0.35, exp: 2.4, tur: 3, ray: 2.6, fogB: [0.06, 0.08, 0.15], fogS: [0.22, 0.14, 0.18], dens: 0.0003, glow: 1.0, moon: true },
};
function sunVec(elevDeg, az) { const e = elevDeg * Math.PI / 180; return new THREE.Vector3(Math.cos(az) * Math.cos(e), Math.sin(e), Math.sin(az) * Math.cos(e)).normalize(); }

const sky = new SkyMesh();
sky.scale.setScalar(40000);
sky.mieCoefficient.value = 0.005;
sky.mieDirectionalG.value = 0.8;
sky.cloudCoverage.value = 0.35;
sky.cloudDensity.value = 0.4;
scene.add(sky);
const FLAT_BG = new THREE.Color(0x9aa6b2);

const sun = new THREE.DirectionalLight(0xffffff, 3);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
Object.assign(sun.shadow.camera, { left: -320, right: 320, top: 320, bottom: -320, near: 10, far: 3000 });
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.35;
scene.add(sun, sun.target);
const ambient = new THREE.AmbientLight(0xffffff, 0);
scene.add(ambient);

const cubeRT = new THREE.CubeRenderTarget(256);
cubeRT.texture.type = THREE.HalfFloatType;
const cubeCam = new THREE.CubeCamera(1, 100000, cubeRT);
function captureSky() {
  world.visible = false; sky.showSunDisc.value = 0;
  cubeCam.update(renderer, scene);
  world.visible = true; sky.showSunDisc.value = 1;
}
scene.environment = cubeRT.texture;
let lightDir = new THREE.Vector3(0, 1, 0);
function setTod(id) {
  const t = TOD[id];
  const sv = sunVec(t.elev, t.az);
  sky.sunPosition.value.copy(sv);
  sky.turbidity.value = t.tur; sky.rayleigh.value = t.ray;
  // blue hour: the sun is below the horizon, the shadow-casting light becomes a cool high moon
  lightDir = t.moon ? sunVec(32, t.az + 2.2) : sv.clone();
  U.sunDir.value.copy(lightDir);
  U.sunColor.value.set(...t.sun);
  sun.color.setRGB(...t.sun);
  U.sunI.value = t.sunI;
  U.fogBlue.value.set(...t.fogB); U.fogSun.value.set(...t.fogS); U.fogDens.value = t.dens;
  U.glow.value = t.glow;
  U.skyAmb.value.set(...t.fogB).multiplyScalar(t.moon ? 0.5 : 1.1);
  todExposure = t.exp;
  captureSky();
}
let todExposure = 1;

scene.fogNode = Fn(() => {
  const toP = positionWorld.sub(cameraPosition);
  const dist = length(toP);
  const dir = toP.div(dist);
  const hAvg = max(positionWorld.y.add(cameraPosition.y).mul(0.5), 0);
  const f = float(1).sub(exp(dist.mul(U.fogDens).mul(exp(hAvg.div(-120))).negate())).mul(U.fog);
  const sunAmt = pow(saturate(dot(dir, U.sunDir)), 6);
  return fog(mix(U.fogBlue, U.fogSun, sunAmt), f);
})();

// ─────────────────────────────────────────────────────────────────────────────
// Height texture: the water shader reads it to find the sea bed under each pixel
// ─────────────────────────────────────────────────────────────────────────────
const EXT = 420, HN = 512;
const heightTex = (() => {
  const data = new Uint16Array(HN * HN);
  for (let j = 0; j < HN; j++) for (let i = 0; i < HN; i++) {
    const x = (i + 0.5) / HN * 2 * EXT - EXT, z = (j + 0.5) / HN * 2 * EXT - EXT;
    data[j * HN + i] = THREE.DataUtils.toHalfFloat(terrainHeight(x, z));
  }
  const t = new THREE.DataTexture(data, HN, HN, THREE.RedFormat, THREE.HalfFloatType);
  t.magFilter = t.minFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.needsUpdate = true;
  return t;
})();
const heightAt = (xz) => texture(heightTex, xz.add(EXT).div(2 * EXT)).r;

// ─────────────────────────────────────────────────────────────────────────────
// Terrain (warped grid, dense around the village) with a per-vertex path mask
// ─────────────────────────────────────────────────────────────────────────────
const CENTER = [-15, 70];
function buildTerrain() {
  const N = 460;
  const pos = new Float32Array((N + 1) * (N + 1) * 3), path = new Float32Array((N + 1) * (N + 1));
  const map = (t, a, b) => (t < 0 ? -a : b) * t * t;
  for (let j = 0; j <= N; j++) {
    const z = CENTER[1] + map(j / N * 2 - 1, 520, 700);
    for (let i = 0; i <= N; i++) {
      const x = CENTER[0] + map(i / N * 2 - 1, 700, 700);
      const k = j * (N + 1) + i;
      pos[k * 3] = x; pos[k * 3 + 1] = terrainHeight(x, z); pos[k * 3 + 2] = z;
      let d = Math.hypot(x - PLAZA[0], z - PLAZA[1]) - 14;
      for (const p of PATHS) d = Math.min(d, polyDist(x, z, p) - 1.4);
      path[k] = 1 - smooth(0, 2.5, d + (vnoise(x * 0.4, z * 0.4) - 0.5) * 1.5);
    }
  }
  const idx = [];
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const a = j * (N + 1) + i, b = a + 1, c = a + N + 1, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('path', new THREE.BufferAttribute(path, 1));
  g.setIndex(idx);
  g.computeVertexNormals();
  if (g.attributes.normal.getY(Math.floor(g.attributes.normal.count / 2)) < 0) {
    g.setIndex(idx.map((_, i) => idx[i - (i % 3) + [0, 2, 1][i % 3]])); g.computeVertexNormals();
  }
  return g;
}
const terrainMat = new THREE.MeshStandardNodeMaterial();
{
  const p = positionWorld, up = normalWorld.y, h = p.y;
  const n1 = mx_fractal_noise_float(p.mul(0.03), 4), n2 = mx_noise_float(p.mul(0.4)), n3 = mx_noise_float(p.mul(2.5));
  const grass = mix(mix(vec3(0.07, 0.17, 0.035), vec3(0.2, 0.3, 0.06), saturate(n1.mul(0.7).add(0.5))), vec3(0.32, 0.3, 0.12), smoothstep(0.35, 0.7, n2.add(n1.mul(0.4))).mul(0.6));
  const sand = vec3(0.78, 0.68, 0.5).mul(n3.mul(0.05).add(1));
  const rock = mix(vec3(0.2, 0.19, 0.18), vec3(0.42, 0.4, 0.36), saturate(n1.mul(0.6).add(0.5)));
  const dirt = vec3(0.4, 0.31, 0.2).mul(n3.mul(0.08).add(1));
  // the wet band follows the swash: it creeps up and down with the waves
  const swash = sin(U.time.mul(0.9).add(p.x.mul(0.05)).add(n2.mul(2))).mul(0.18).add(0.45);
  const wet = smoothstep(swash.add(0.25), swash.sub(0.1), h).mul(step0(h.add(0.5)));
  const beach = smoothstep(2.2, 1.3, h.add(n2.mul(0.4)));
  const steep = smoothstep(0.82, 0.62, up.add(n2.mul(0.06)));
  let col = mix(grass, sand.mul(mix(1, 0.6, wet)), beach);
  col = mix(col, dirt, attribute('path', 'float').mul(float(1).sub(beach)));
  col = mix(col, rock, steep);
  // flat "game" colours: two-tone grass / sand, the way the Unity build reads from afar
  const flat = mix(vec3(0.3, 0.55, 0.15), vec3(0.9, 0.82, 0.62), smoothstep(1.8, 1.5, h));
  terrainMat.colorNode = mix(mix(flat, vec3(0.55, 0.52, 0.48), steep), col, U.ground);
  terrainMat.roughnessNode = mix(float(0.9), mix(mix(float(0.95), float(0.3), wet.mul(beach)), float(0.85), steep), U.ground);
  const fpT = length(fwidth(p));
  const strata = mx_fractal_noise_float(p.mul(vec3(0.08, 0.3, 0.08)), 3).mul(1.2).mul(steep).mul(float(1).sub(smoothstep(0.5, 2, fpT)));
  terrainMat.normalNode = proceduralBump(strata, U.ground);
}
function step0(x) { return smoothstep(-0.01, 0.01, x); }
const terrain = new THREE.Mesh(buildTerrain(), terrainMat);
terrain.castShadow = true; terrain.receiveShadow = true;
world.add(terrain);

// ─────────────────────────────────────────────────────────────────────────────
// Water: one sea-level surface (sea, bay, river). Looks up the bed along the refracted ray, absorbs per RGB with
// distance travelled, adds single scattering, caustics on the bed, shore foam, reflection by Fresnel.
// ─────────────────────────────────────────────────────────────────────────────
const SIGMA_A = vec3(0.42, 0.065, 0.025), SIGMA_S = vec3(0.02, 0.03, 0.045);
const SIGMA_T = SIGMA_A.add(SIGMA_S);
const WAVES = [[0.3, 9, 0.03], [1.1, 5.5, 0.03], [-0.5, 3.4, 0.03], [2.0, 2.0, 0.026], [-1.3, 1.3, 0.024], [0.7, 0.8, 0.02], [2.7, 0.5, 0.016]];
function waveNormal(p) {
  const fp = max(fwidth(p.x), fwidth(p.z));
  let sx = float(0), sz = float(0);
  for (const [a, lambda, steep] of WAVES) {
    const k = 2 * Math.PI / lambda, w = Math.sqrt(9.81 * k), dx = Math.cos(a), dz = Math.sin(a);
    const s = cos(p.x.mul(dx * k).add(p.z.mul(dz * k)).sub(U.time.mul(w))).mul(steep).mul(float(1).sub(smoothstep(lambda * 0.12, lambda * 0.5, fp)));
    sx = sx.add(s.mul(dx)); sz = sz.add(s.mul(dz));
  }
  const gust = smoothstep(-0.4, 0.5, mx_fractal_noise_float(vec3(p.x.mul(0.012), U.time.mul(0.03), p.z.mul(0.018)), 3));
  const amp = mix(float(0.3), float(1.3), gust);
  return normalize(vec3(sx.mul(amp).negate(), 1, sz.mul(amp).negate()));
}
function causticsAt(xz, depth) {
  const t = U.time;
  // domain-warped so the cells bend into curved, irregular filaments instead of straight-edged tiles
  const warp = vec2(mx_noise_float(vec3(xz.mul(0.18), t.mul(0.2))), mx_noise_float(vec3(xz.mul(0.18).add(9.1), t.mul(0.2)))).mul(1.4);
  const q = xz.add(warp);
  const a = mx_worley_noise_vec2(vec3(q.mul(0.7), t.mul(0.45)));
  const b = mx_worley_noise_vec2(vec3(q.mul(0.93).add(vec2(3.7, 1.3)), t.mul(0.55).add(7)));
  const edge = (w) => float(1).sub(smoothstep(0.0, 0.07, w.y.sub(w.x)));
  const c = pow(edge(a).mul(edge(b).add(0.35)), 1.5);
  // fade in from the shoreline, spread / blur out with depth
  return c.mul(smoothstep(0.05, 0.6, depth)).mul(exp(depth.mul(-0.12)));
}
const refl = reflector({ resolutionScale: 0.5 });
refl.target.rotateX(-Math.PI / 2);
world.add(refl.target);
const waterMat = new THREE.MeshBasicNodeMaterial();
{
  const p = positionWorld;
  const n = waveNormal(p);
  const v = normalize(cameraPosition.sub(p));
  refl.uvNode = refl.uvNode.add(clamp(n.xz.mul(0.04), -0.01, 0.01));
  const depth = max(heightAt(p.xz).negate(), 0);
  const r = refract(v.negate(), n, 1 / 1.333);
  const pathLen = depth.div(max(r.y.negate(), 0.25)).min(60);
  const fxz = p.xz.add(r.xz.mul(pathLen));
  const fDepth = max(heightAt(fxz).negate(), 0);
  // the bed: rippled sand, darker sea-grass patches in deeper water
  const rip = sin(fxz.x.mul(3.1).add(mx_noise_float(vec3(fxz.mul(0.5), 0)).mul(5))).mul(0.018);
  const grassPatch = smoothstep(0.25, 0.55, mx_fractal_noise_float(vec3(fxz.mul(0.04), 1), 3)).mul(smoothstep(2.5, 5, fDepth));
  const bed = mix(vec3(0.8, 0.72, 0.55).add(rip), vec3(0.12, 0.2, 0.08), grassPatch);
  const sunAtBed = U.sunColor.mul(U.sunI).mul(max(U.sunDir.y, 0)).mul(exp(SIGMA_T.mul(fDepth).mul(-1.25)));
  const caus = causticsAt(fxz, fDepth).mul(U.caustics).mul(2.2);
  const bedLit = bed.mul(sunAtBed.mul(caus.add(1)).add(U.skyAmb.mul(exp(SIGMA_T.mul(fDepth).mul(-0.8))).mul(0.8)));
  const T = exp(SIGMA_T.mul(pathLen).negate());
  const inscatter = SIGMA_S.div(SIGMA_T).mul(U.sunColor.mul(U.sunI).mul(0.25).add(U.skyAmb)).mul(float(1).sub(T));
  const under = bedLit.mul(T).add(inscatter);
  const fres = float(0.02).add(pow(float(1).sub(saturate(dot(n, v))), 5).mul(0.98));
  const hv = normalize(U.sunDir.add(v));
  const spec = U.sunColor.mul(U.sunI).mul(pow(saturate(dot(n, hv)), 700).mul(25).add(pow(saturate(dot(n, hv)), 60).mul(0.25))).min(40);
  let col = mix(under, refl.rgb, fres).add(spec);
  // shore foam: swash lines pushing up the beach, broken up by noise, and a steady white edge
  const fn = mx_fractal_noise_float(vec3(p.xz.mul(0.35), U.time.mul(0.2)), 3);
  const lines = smoothstep(0.55, 0.95, sin(depth.mul(6).sub(U.time.mul(1.6)).add(fn.mul(3)))).mul(smoothstep(1.5, 0.3, depth));
  const edgeFoam = smoothstep(0.35, 0.0, depth.add(fn.mul(0.15)));
  const foam = saturate(lines.mul(smoothstep(-0.1, 0.4, fn)).add(edgeFoam)).mul(U.foam);
  const foamLit = U.sunColor.mul(U.sunI).mul(max(U.sunDir.y, 0)).mul(0.8).add(U.skyAmb).mul(0.85);
  col = mix(col, foamLit, foam);
  waterMat.colorNode = col;
}
const waterGeo = new THREE.PlaneGeometry(80000, 80000).rotateX(-Math.PI / 2);
const water = new THREE.Mesh(waterGeo, waterMat);
world.add(water);
// the "game" water: a translucent turquoise sheet
const waterBasic = new THREE.Mesh(waterGeo, new THREE.MeshStandardNodeMaterial({ color: 0x3fc9c9, roughness: 0.12, transparent: true, opacity: 0.78 }));
waterBasic.receiveShadow = true;
waterBasic.visible = false;
world.add(waterBasic);
// pool above the waterfall (its own little surface)
const poolMat = new THREE.MeshStandardNodeMaterial({ color: 0x1d5a63, roughness: 0.08 });
const pool = new THREE.Mesh(new THREE.CircleGeometry(11, 32).rotateX(-Math.PI / 2), poolMat);
pool.position.set(FALL[0], 25.2, FALL[1] - 9);
pool.scale.set(1, 1, 1.25);
world.add(pool);

// waterfall sheet + plunge foam
{
  const H = 25.2, Wd = 7;
  const g = new THREE.PlaneGeometry(Wd, H + 1.5, 8, 24);
  const pz = g.attributes.position;
  for (let i = 0; i < pz.count; i++) {
    const y = pz.getY(i) + (H + 1.5) / 2;           // 0 bottom … H top
    const t = 1 - y / (H + 1.5);
    pz.setZ(i, Math.pow(t, 1.6) * 3.2 * (1 - t) + t * 2.5);   // bows out from the lip and lands a little forward
    pz.setY(i, y - 1);
  }
  g.computeVertexNormals();
  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide });
  const st = uv();
  const streak = mx_fractal_noise_float(vec3(st.x.mul(9), st.y.mul(1.4).add(U.time.mul(1.3)), 0), 3);
  const lit = U.sunColor.mul(U.sunI).mul(0.45).add(U.skyAmb).mul(0.9);
  m.colorNode = lit.mul(mix(0.75, 1.15, streak));
  m.opacityNode = smoothstep(-0.25, 0.35, streak).mul(0.75).add(0.18).mul(smoothstep(0, 0.12, st.x)).mul(smoothstep(1, 0.88, st.x));
  const fall = new THREE.Mesh(g, m);
  fall.position.set(FALL[0], 0, FALL[1] - 0.5);
  world.add(fall);
  const fm = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
  const r = length(uv().sub(0.5)).mul(2);
  const fnz = mx_fractal_noise_float(vec3(uv().mul(6), U.time.mul(0.8)), 3);
  fm.colorNode = lit;
  fm.opacityNode = smoothstep(1, 0.2, r.add(fnz.mul(0.35))).mul(0.85);
  const foamDisc = new THREE.Mesh(new THREE.CircleGeometry(7, 40).rotateX(-Math.PI / 2), fm);
  foamDisc.position.set(FALL[0], 0.06, FALL[1] + 3.5);
  world.add(foamDisc);
}

// ─────────────────────────────────────────────────────────────────────────────
// Vegetation: instanced grass blades and trees, all bending to the same travelling gusts
// ─────────────────────────────────────────────────────────────────────────────
const windAt = (p) => {
  // gust fronts travel along +x: neighbours lean together, then each plant adds its own flutter
  const front = sin(p.x.mul(0.06).add(p.z.mul(0.02)).sub(U.time.mul(1.4)));
  return smoothstep(-0.2, 1, front).mul(0.8).add(0.25);
};
const vegetation = new THREE.Group();
world.add(vegetation);
function slopeAt(x, z) { const e = 1.5; return Math.hypot(terrainHeight(x + e, z) - terrainHeight(x - e, z), terrainHeight(x, z + e) - terrainHeight(x, z - e)) / (2 * e); }
function onPath(x, z) { let d = Math.hypot(x - PLAZA[0], z - PLAZA[1]) - 15; for (const p of PATHS) d = Math.min(d, polyDist(x, z, p) - 2); return d < 0; }
{
  // grass blade: 5 verts, tapering to a tip; uv.y = 0 root … 1 tip
  const bg = new THREE.BufferGeometry();
  const w = 0.055, h = 0.55;
  bg.setAttribute('position', new THREE.Float32BufferAttribute([-w, 0, 0, w, 0, 0, -w * 0.7, h * 0.5, 0.02, w * 0.7, h * 0.5, 0.02, 0, h, 0.06], 3));
  bg.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 0.5, 1, 0.5, 0.5, 1], 2));
  bg.setAttribute('normal', new THREE.Float32BufferAttribute([0, 0.4, 1, 0, 0.4, 1, 0, 0.5, 1, 0, 0.5, 1, 0, 0.7, 1], 3));
  bg.setIndex([0, 1, 2, 1, 3, 2, 2, 3, 4]);
  const gm = new THREE.MeshStandardNodeMaterial({ side: THREE.DoubleSide, roughness: 0.8 });
  const tip = uv().y;
  const vary = hash(float(instanceIndex));
  gm.colorNode = mix(vec3(0.06, 0.13, 0.03), mix(vec3(0.24, 0.38, 0.07), vec3(0.42, 0.42, 0.14), vary.mul(vary)), tip);
  const sway = windAt(positionLocal).add(sin(U.time.mul(3.1).add(vary.mul(30))).mul(0.12));
  gm.positionNode = positionLocal.add(vec3(sway.mul(0.32), sway.mul(-0.08), sway.mul(0.1)).mul(tip.mul(tip)));
  const gr = mulberry32(3), mats = [];
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), pv = new THREE.Vector3(), yA = new THREE.Vector3(0, 1, 0);
  for (let tries = 0; tries < 400000 && mats.length < 60000; tries++) {
    // denser near the village, thinning out up the hills
    const x = PLAZA[0] + (gr() * 2 - 1) * 170, z = PLAZA[1] - 120 + gr() * 175;
    const hh = terrainHeight(x, z);
    if (hh < 2.5 || hh > 40 || onPath(x, z) || slopeAt(x, z) > 0.5) continue;      // keep blades off the sand
    if (gr() < Math.hypot(x - PLAZA[0], z - PLAZA[1]) / 230) continue;
    q.setFromAxisAngle(yA, gr() * 6.28);
    const k = 0.7 + gr() * 0.8;
    mats.push(m4.compose(pv.set(x, hh - 0.05, z), q, s.set(k, k * (0.8 + gr() * 0.6), k)).clone());
  }
  const grass = new THREE.InstancedMesh(bg, gm, mats.length);
  mats.forEach((m, i) => grass.setMatrixAt(i, m));
  grass.receiveShadow = true;
  vegetation.add(grass);
}
function treeGeos() {
  const tag = (g, y0, y1) => {
    const p = g.attributes.position, a = new Float32Array(p.count);
    for (let i = 0; i < p.count; i++) a[i] = Math.max(0, (p.getY(i) - y0) / (y1 - y0));
    g.setAttribute('sway', new THREE.BufferAttribute(a, 1));
    return g.index ? g.toNonIndexed() : g;
  };
  const trunk = (h, r) => new THREE.CylinderGeometry(r * 0.7, r, h, 7).translate(0, h / 2, 0);
  const pine = mergeGeometries([trunk(2, 0.25), ...[[2.2, 3.4, 2.6], [1.7, 3, 4.6], [1.2, 2.6, 6.4], [0.7, 2.2, 8]].map(([r, h, y]) => new THREE.ConeGeometry(r, h, 8, 1).translate(0, y, 0))].map((g) => tag(g, 0, 9.5)));
  const blobs = [[0, 5.2, 0, 2.6], [1.4, 4.4, 0.6, 1.9], [-1.3, 4.6, -0.5, 2.0], [0.3, 6.6, -0.3, 1.8]].map(([x, y, z, r]) => new THREE.IcosahedronGeometry(r, 1).translate(x, y, z));
  const round = mergeGeometries([trunk(3.6, 0.3), ...blobs].map((g) => tag(g, 0, 8.4)));
  pine.computeVertexNormals(); round.computeVertexNormals();
  return { pine, round };
}
{
  const { pine, round } = treeGeos();
  const mk = (base, tip) => {
    const m = new THREE.MeshStandardNodeMaterial({ roughness: 0.85, flatShading: true });
    const sw = attribute('sway', 'float');
    const vary = hash(float(instanceIndex).add(5));
    m.colorNode = mix(vec3(0.16, 0.11, 0.07), mix(base, tip, vary), smoothstep(0.12, 0.2, sw));
    const sway = windAt(positionLocal).add(sin(U.time.mul(1.7).add(vary.mul(40))).mul(0.15));
    m.positionNode = positionLocal.add(vec3(sway.mul(0.5), 0, sway.mul(0.15)).mul(sw.mul(sw)));
    return m;
  };
  const tr = mulberry32(17);
  const lists = { pine: [], round: [] };
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), pv = new THREE.Vector3(), yA = new THREE.Vector3(0, 1, 0);
  for (let tries = 0; tries < 60000 && lists.pine.length + lists.round.length < 1500; tries++) {
    const x = -260 + tr() * 480, z = -320 + tr() * 420;
    const hh = terrainHeight(x, z);
    if (hh < 2.5 || onPath(x, z) || slopeAt(x, z) > 0.75) continue;
    if (Math.hypot(x - PLAZA[0], z - PLAZA[1]) < 62 || Math.hypot(x - BAY[0], z - BAY[1]) < 60) continue;
    if (polyDist(x, z, RIVER) < 15 || Math.hypot(x - FALL[0], z - FALL[1]) < 20) continue;
    if (fbm(x * 0.02 + 9, z * 0.02) < 0.42) continue;          // clumps, with clearings
    const kind = hh > 22 || tr() < 0.45 ? 'pine' : 'round';
    const k = 0.75 + tr() * 0.7;
    q.setFromAxisAngle(yA, tr() * 6.28);
    lists[kind].push(m4.compose(pv.set(x, hh - 0.3, z), q, s.set(k, k * (0.85 + tr() * 0.35), k)).clone());
  }
  for (const [kind, geo, mat] of [['pine', pine, mk(vec3(0.05, 0.16, 0.08), vec3(0.09, 0.24, 0.1))], ['round', round, mk(vec3(0.17, 0.33, 0.07), vec3(0.32, 0.42, 0.1))]]) {
    const im = new THREE.InstancedMesh(geo, mat, lists[kind].length);
    lists[kind].forEach((m, i) => im.setMatrixAt(i, m));
    im.castShadow = true; im.receiveShadow = true;
    vegetation.add(im);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Village: cottages, the fish stall, string lights, lamps, jetty, rowboat, bridge
// ─────────────────────────────────────────────────────────────────────────────
const bevelSwaps = [];
const glowWarm = vec3(1.0, 0.58, 0.25);
const winMat = new THREE.MeshStandardNodeMaterial({ roughness: 0.15 });
winMat.colorNode = vec3(0.05, 0.07, 0.09);
winMat.emissiveNode = glowWarm.mul(U.lights.mul(U.glow).mul(9));
const bulbMat = new THREE.MeshBasicNodeMaterial();
bulbMat.colorNode = mix(vec3(0.85, 0.8, 0.7), glowWarm.mul(U.glow.mul(22).add(0.6)), U.lights);
const M = (c, r = 0.75) => new THREE.MeshStandardNodeMaterial({ color: c, roughness: r });
const wallMat = M(0xe9e1cf, 0.85), woodMat = M(0x6a4a30, 0.85), darkWood = M(0x3b2a1d, 0.9), trimMat = M(0xf4f1ea, 0.6);
const roofMats = [0xa8402c, 0x3f6f9a, 0x4f7f4e, 0xb5562f, 0x7a4a7a].map((c) => M(c, 0.7));
const pointLights = [];

function swap(gB, gS, mat) { const m = new THREE.Mesh(gB, mat); m.castShadow = m.receiveShadow = true; bevelSwaps.push({ mesh: m, gB, gS }); return m; }
function roofGeo(w, d, h, bevel) {
  const s = new THREE.Shape([new THREE.Vector2(-d / 2, 0), new THREE.Vector2(d / 2, 0), new THREE.Vector2(0, h)]);
  const g = new THREE.ExtrudeGeometry(s, bevel ? { depth: w, bevelEnabled: true, bevelThickness: 0.08, bevelSize: 0.08, bevelSegments: 2 } : { depth: w, bevelEnabled: false });
  g.rotateY(Math.PI / 2); g.translate(-w / 2, 0, 0);
  return g;
}
function cottage(x, z, yaw, roofMat, w = 7, d = 6, h = 3.6) {
  const g = new THREE.Group();
  g.position.set(x, terrainHeight(x, z) - 0.2, z);
  g.rotation.y = yaw;
  const body = swap(new RoundedBoxGeometry(w, h, d, 3, 0.14), new THREE.BoxGeometry(w, h, d), wallMat);
  body.position.y = h / 2;
  const roof = swap(roofGeo(w + 0.8, d + 1.2, 2.4, true), roofGeo(w + 0.8, d + 1.2, 2.4, false), roofMat);
  roof.position.y = h;
  g.add(body, roof);
  // front (+z): door + two windows; sides: one window each
  const door = new THREE.Mesh(new THREE.BoxGeometry(1.1, 2.1, 0.12), darkWood);
  door.position.set(0, 1.05, d / 2 + 0.02);
  g.add(door);
  for (const [px, py, pz, ry] of [[-2.1, 1.9, d / 2 + 0.03, 0], [2.1, 1.9, d / 2 + 0.03, 0], [w / 2 + 0.03, 1.9, 0, Math.PI / 2], [-w / 2 - 0.03, 1.9, 0, Math.PI / 2]]) {
    const fr = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 0.06), trimMat);
    const pane = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.95, 0.08), winMat);
    fr.position.set(px, py, pz); pane.position.set(px, py, pz + (ry ? 0 : 0.01));
    fr.rotation.y = pane.rotation.y = ry;
    g.add(fr, pane);
  }
  const chim = new THREE.Mesh(new RoundedBoxGeometry(0.7, 1.8, 0.7, 2, 0.06), M(0x8a7a6a));
  chim.position.set(w * 0.25, h + 1.6, -d * 0.18);
  chim.castShadow = true;
  g.add(chim);
  world.add(g);
  return g;
}
// ring of cottages around the plaza, facing it
{
  const spots = [[-58, 72], [-50, 112], [-2, 62], [18, 92], [12, 124], [-40, 50], [-72, 98], [-26, 132]];
  spots.forEach(([x, z], i) => {
    const yaw = Math.atan2(PLAZA[0] - x, PLAZA[1] - z);
    cottage(x, z, yaw, roofMats[i % roofMats.length], 6.5 + (i % 3), 5.5 + (i % 2), 3.4 + (i % 2) * 0.6);
  });
}
// fish stall with a striped awning (the game's market stall)
{
  const g = new THREE.Group();
  g.position.set(PLAZA[0] + 6, 3.4, PLAZA[1] + 4);
  g.rotation.y = 0.4;
  for (const [x, z] of [[-1.6, -1], [1.6, -1], [-1.6, 1], [1.6, 1]]) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.6, 8), woodMat);
    post.position.set(x, 1.3, z); post.castShadow = true; g.add(post);
  }
  const table = swap(new RoundedBoxGeometry(3.4, 0.2, 1.4, 2, 0.05), new THREE.BoxGeometry(3.4, 0.2, 1.4), woodMat);
  table.position.set(0, 0.95, 0.3); g.add(table);
  const aw = new THREE.MeshStandardNodeMaterial({ roughness: 0.8, side: THREE.DoubleSide });
  aw.colorNode = mix(vec3(0.85, 0.83, 0.78), vec3(0.72, 0.16, 0.12), smoothstep(0.45, 0.55, sin(uv().x.mul(Math.PI * 12)).mul(0.5).add(0.5)));
  const awn = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 2.4, 3.8, 24, 1, true, -0.7, 1.4).rotateZ(Math.PI / 2), aw);
  awn.position.set(0, 0.7, 0.2); awn.scale.set(1, 0.5, 1.1); awn.castShadow = true;
  g.add(awn);
  for (let i = 0; i < 5; i++) {
    const fish = new THREE.Mesh(new THREE.CapsuleGeometry(0.07, 0.35, 4, 8).rotateZ(Math.PI / 2), M(0x8fa3ad, 0.35));
    fish.position.set(-1.2 + i * 0.6, 1.1, 0.3 + (i % 2) * 0.2); g.add(fish);
  }
  world.add(g);
}
// lamp posts + string lights across the plaza
{
  const poles = [];
  for (let i = 0; i < 6; i++) {
    const a = i / 6 * Math.PI * 2 + 0.3, x = PLAZA[0] + Math.cos(a) * 16, z = PLAZA[1] + Math.sin(a) * 16;
    const y = terrainHeight(x, z);
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.12, 4.2, 8), darkWood);
    pole.position.set(x, y + 2.1, z); pole.castShadow = true; world.add(pole);
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.22, 12, 8), bulbMat);
    lamp.position.set(x, y + 4.3, z); world.add(lamp);
    poles.push(new THREE.Vector3(x, y + 4.0, z));
    if (i % 2 === 0) { const L = new THREE.PointLight(0xffa860, 0, 30, 2); L.position.set(x, y + 3.6, z); world.add(L); pointLights.push(L); }
  }
  const bulbs = [];
  for (let i = 0; i < poles.length; i++) for (const j of [(i + 2) % poles.length, (i + 3) % poles.length]) {
    if (j < i) continue;
    const a = poles[i], b = poles[j];
    for (let k = 1; k < 16; k++) {
      const t = k / 16, p = a.clone().lerp(b, t);
      p.y -= Math.sin(t * Math.PI) * 1.4;          // catenary-ish sag
      bulbs.push(p);
    }
  }
  const im = new THREE.InstancedMesh(new THREE.SphereGeometry(0.09, 8, 6), bulbMat, bulbs.length);
  bulbs.forEach((p, i) => im.setMatrixAt(i, new THREE.Matrix4().makeTranslation(p.x, p.y, p.z)));
  world.add(im);
}
// jetty into the bay, with a moored rowboat
let rowboat;
{
  const sx = -44, sz0 = 138, len = 34;
  const deckB = new RoundedBoxGeometry(2.6, 0.22, len, 2, 0.05), deckS = new THREE.BoxGeometry(2.6, 0.22, len);
  const deck = swap(deckB, deckS, woodMat);
  deck.position.set(sx, 1.25, sz0 + len / 2);
  world.add(deck);
  for (let i = 0; i <= 8; i++) for (const side of [-1.2, 1.2]) {
    const z = sz0 + i * len / 8, y0 = terrainHeight(sx + side, z);
    const hpost = 1.4 - Math.min(y0, 0) + 0.3;
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.15, hpost, 8), darkWood);
    post.position.set(sx + side, 1.3 - hpost / 2 + 0.3, z); post.castShadow = true;
    world.add(post);
  }
  // rowboat: a bevelled hull outline extruded upward, with a lighter inset floor and a thwart
  const outline = (s) => {
    const sh = new THREE.Shape();
    sh.moveTo(0, -2.1 * s);
    sh.bezierCurveTo(0.95 * s, -1.4 * s, 0.85 * s, 1.3 * s, 0.55 * s, 2.0 * s);
    sh.lineTo(-0.55 * s, 2.0 * s);
    sh.bezierCurveTo(-0.85 * s, 1.3 * s, -0.95 * s, -1.4 * s, 0, -2.1 * s);
    return sh;
  };
  const hullG = new THREE.ExtrudeGeometry(outline(1), { depth: 0.55, bevelEnabled: true, bevelThickness: 0.08, bevelSize: 0.08, bevelSegments: 3, curveSegments: 16 }).rotateX(-Math.PI / 2).translate(0, -0.3, 0);
  const floorG = new THREE.ExtrudeGeometry(outline(0.84), { depth: 0.04, bevelEnabled: false, curveSegments: 16 }).rotateX(-Math.PI / 2).translate(0, 0.32, 0);   // inset deck just above the hull cap
  rowboat = new THREE.Group();
  const hm = new THREE.Mesh(hullG, M(0x8d4f2c, 0.7)); hm.castShadow = true;
  const fl = new THREE.Mesh(floorG, M(0xc69a6a, 0.8)); fl.receiveShadow = true;
  const seat = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.08, 0.35), M(0xd9b88a));
  seat.position.y = 0.46;
  const oar = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 3.2, 6).rotateZ(Math.PI / 2 - 0.12), M(0xd9b88a));
  oar.position.set(1.1, 0.46, 0.3); oar.rotation.y = 0.35;
  rowboat.add(hm, fl, seat, oar);
  rowboat.position.set(sx + 3.2, 0.15, sz0 + len - 3);
  rowboat.rotation.y = 0.15;
  world.add(rowboat);
}
// footbridge over the river near the village
{
  const p0 = new THREE.Vector3(28, 0, 30), dir = new THREE.Vector3(1, 0, -0.55).normalize();
  const deck = swap(new RoundedBoxGeometry(22, 0.3, 2.8, 2, 0.06), new THREE.BoxGeometry(22, 0.3, 2.8), woodMat);
  deck.position.set(p0.x, 3.0, p0.z);
  deck.rotation.y = -Math.atan2(dir.z, dir.x);
  world.add(deck);
  for (const s of [-1.3, 1.3]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(22, 0.1, 0.1), darkWood);
    rail.position.set(p0.x, 4.0, p0.z); rail.rotation.y = deck.rotation.y;
    rail.position.add(new THREE.Vector3(-dir.z, 0, dir.x).multiplyScalar(s));
    world.add(rail);
  }
}
// beach rocks
{
  const rr = mulberry32(41);
  const rockMat = M(0x77736c, 0.9);
  for (let i = 0; i < 40; i++) {
    const a = rr() * Math.PI * 2, rad = 160 + rr() * 60;
    const x = Math.cos(a) * rad, z = -40 + Math.sin(a) * rad, h = terrainHeight(x, z);
    if (h < -0.6 || h > 2.4) continue;
    const g = new THREE.IcosahedronGeometry(0.6 + rr() * 1.6, 1);
    const p = g.attributes.position;
    for (let k = 0; k < p.count; k++) p.setY(k, p.getY(k) * 0.6);
    g.computeVertexNormals();
    const m = new THREE.Mesh(g, rockMat);
    m.position.set(x, h, z); m.rotation.y = rr() * 6;
    m.castShadow = m.receiveShadow = true;
    world.add(m);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Post: CA → warm bloom → exposure/WB → tone map → grade → ink outline → vignette, grain
// ─────────────────────────────────────────────────────────────────────────────
const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
const hdr = scenePass.getTextureNode();
const depthTex = scenePass.getTextureNode('depth');
const glow = bloom(hdr, 0.3, 0.55, 1.0);
glow.strength = U.bloom;
const WB = vec3(1.06, 0.99, 0.92);
pipeline.outputNode = Fn(() => {
  const uvc = screenUV.sub(0.5);
  const ca = uvc.mul(U.ca);
  const col = vec3(hdr.sample(screenUV.sub(ca)).r, hdr.sample(screenUV).g, hdr.sample(screenUV.add(ca)).b);
  const lit = col.add(glow.rgb.mul(vec3(1.0, 0.82, 0.6))).mul(U.exposure).mul(mix(vec3(1), WB, U.grade));
  const one = float(1);
  const w = (k) => max(float(1).sub(U.tone.sub(k).abs()), 0);
  const tm = agxToneMapping(lit, one).mul(w(1)).add(neutralToneMapping(lit, one).mul(w(2)))
    .add(acesFilmicToneMapping(lit, one).mul(w(3))).add(lit.mul(w(0)));
  let o = sRGBTransferOETF(tm);
  const l = luminance(o);
  let g = mix(vec3(l), o, 1.2);
  g = g.add(vec3(0.05, 0.025, -0.03).mul(smoothstep(0.35, 1.0, l)));
  g = g.add(vec3(-0.015, 0.01, 0.028).mul(pow(float(1).sub(l), 2)));
  g = mix(g, smoothstep(0.0, 1.0, g), 0.3);
  o = mix(o, g, U.grade);
  // ink outline (the game's InkOutline): depth Laplacian relative to depth
  const px = vec2(1).div(viewportSize);
  const vz = (o2) => perspectiveDepthToViewZ(depthTex.sample(screenUV.add(o2)).r, U.near, U.far).negate();
  const c0 = vz(vec2(0));
  const lap = vz(vec2(px.x, 0)).add(vz(vec2(px.x.negate(), 0))).add(vz(vec2(0, px.y))).add(vz(vec2(0, px.y.negate()))).sub(c0.mul(4));
  const edge = smoothstep(0.012, 0.03, abs(lap).div(c0)).mul(smoothstep(4000, 1500, c0));
  o = mix(o, vec3(0.13, 0.11, 0.12), edge.mul(U.outline).mul(0.85));
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
const state = { ...DATA.presets.find((p) => p.id === 'minamo').state };
let currentTod = null;
function apply() {
  const s = state;
  if (s.tod !== currentTod) { currentTod = s.tod; setTod(s.tod); }
  sky.visible = s.sky;
  scene.background = s.sky ? null : FLAT_BG;
  scene.environmentIntensity = s.sky ? 1 : 0;
  ambient.intensity = s.sky ? 0 : 1.2;
  sun.intensity = U.sunI.value * (s.sky ? 1 : 0.8);
  sun.shadow.intensity = s.shadow ? 1 : 0;
  U.ground.value = s.ground ? 1 : 0;
  water.visible = s.water; waterBasic.visible = !s.water;
  U.caustics.value = s.caustics ? 1 : 0;
  U.foam.value = s.foam ? 1 : 0;
  for (const b of bevelSwaps) b.mesh.geometry = s.bevel ? b.gB : b.gS;
  vegetation.children[0].visible = s.density;          // grass blades
  U.lights.value = s.lights ? 1 : 0;
  for (const L of pointLights) L.intensity = s.lights ? 260 * U.glow.value : 0;
  U.fog.value = s.fog ? 1 : 0;
  U.bloom.value = s.bloom ? (s.bloomStrength ?? 0.3) : 0;
  U.tone.value = s.tonemap ? TONES[s.toneMode || 'agx'] : 0;
  U.exposure.value = todExposure * (s.tonemap ? 1 : 0.8);
  U.grade.value = s.grade ? 1 : 0;
  U.vig.value = s.lens ? 0.25 : 0;
  U.grain.value = s.lens ? 0.012 : 0;
  U.ca.value = s.lens ? (s.ca ?? 0.0016) : 0;
  U.outline.value = s.outline ? 1 : 0;
  syncUI();
}

const VIEWS = {
  village: { pos: [30, 30, 182], at: [-32, 4, 82] },
  lagoon: { pos: [-20, 2.4, 192], at: [-60, -0.4, 140] },
  topdown: { pos: [-36, 22, 182], at: [-39, 0, 167] },
  sunset: { pos: [-150, 3.5, 225], at: [-520, 12, 360] },
  waterfall: { pos: [10, 9, 6], at: [-15, 13, -54] },
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
const presetsEl = $('presets'), todsEl = $('tods'), viewsEl = $('views'), techsEl = $('techs'), infoEl = $('info');
let currentPreset = 'minamo';
const chip = (parent, label, id, onclick, title = '') => {
  const b = document.createElement('button');
  b.className = 'chip'; b.textContent = label; b.dataset.id = id; b.title = title; b.onclick = onclick;
  parent.append(b);
};
for (const p of DATA.presets) chip(presetsEl, p.name, p.id, () => { usePreset(p.id); showPreset(p); }, p.desc);
for (const t of DATA.tods) chip(todsEl, t.name, t.id, () => { state.tod = t.id; apply(); });
for (const v of DATA.views) chip(viewsEl, v.name, v.id, () => setView(v.id));
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
  const text = ['目標：用 three.js r186（WebGPURenderer + TSL）重做潮聲島：南邊有漁村廣場、碼頭和淺水潟湖，河流從北邊的瀑布流進海裡，島上有草地和樹林。單位一律用公尺，所有水面共用海平面。',
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
  for (const b of todsEl.children) b.classList.toggle('on', b.dataset.id === state.tod);
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
  if (!frozen) simT += dt;
  U.time.value = simT;
  U.frame.value = (U.frame.value + 1) % 1000;
  rowboat.position.y = 0.12 + Math.sin(simT * 1.1) * 0.05;
  rowboat.rotation.z = Math.sin(simT * 0.9) * 0.03;
  rowboat.rotation.x = Math.cos(simT * 0.7) * 0.02;
  controls.update();
  const f = controls.target;
  sun.target.position.set(f.x, 0, f.z);
  sun.position.copy(sun.target.position).addScaledVector(lightDir, 1500);
  pipeline.render();
  if (pendingShot && --pendingShot.wait <= 0) { const s = pendingShot; pendingShot = null; s.resolve(canvas.toDataURL('image/jpeg', 0.9)); }
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

// capture API (same as the fjord page): fixed size, frozen time, JPEGs posted to tools/serve.py
let pendingShot = null, captureSize = null;
function grab(wait) { return new Promise((resolve) => { pendingShot = { wait, resolve }; }); }
window.tide = {
  state, apply, setView, usePreset, U, VIEWS,
  set(partial) { Object.assign(state, partial); currentPreset = null; apply(); },
  async captureBegin(w = 1600, h = 900) {
    captureSize = [w, h]; frozen = true; simT = 37; document.body.classList.add('capture');
    renderer.setPixelRatio(1); renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  },
  captureEnd() { captureSize = null; frozen = false; document.body.classList.remove('capture'); renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); resize(); },
  async shot(name, wait = 14) {
    const url = await grab(wait);
    return (await fetch(`/save?name=${name}`, { method: 'POST', body: url })).ok;
  },
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

usePreset('minamo');
setView('village');
showTech(DATA.techniques[4]);
renderer.setAnimationLoop(frame);
setTimeout(() => loaderEl.classList.add('gone'), 400);
