import { createCamera } from './camera.js';
import { createQuad, createRamFromData } from './mesh.js';
import { loadOBJScene } from './objLoader.js';

// Active scene:
// - 'ram'        : OBJ scene in data/scenes/ram (exported from ram-mesh.json + meshes.csv)
// - 'sponza'     : OBJ scene in data/scenes/sponza
// - 'conference' : OBJ scene in data/scenes/conference
const ACTIVE_SCENE = 'ram';

// Very small CSV helper: returns all fields as strings.
// Callers are responsible for converting numeric fields.
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim());
    if (!values[0]) continue; // skip empty lines
    const row = {};
    headers.forEach((h, j) => {
      row[h] = values[j] ?? '';
    });
    rows.push(row);
  }
  return rows;
}

async function loadMaterials() {
  console.log('[Scene] Loading materials from data/materials.csv');
  const res = await fetch('data/materials.csv');
  const rows = parseCSV(await res.text());
  console.log('[Scene] Materials CSV rows:', rows.length);
  const materials = rows.map(r => ({
    albedo: [Number(r.albedo_r), Number(r.albedo_g), Number(r.albedo_b)],
    roughness: Number(r.roughness),
    metalness: Number(r.metalness),
  }));
  console.log('[Scene] Parsed materials:', materials);
  return materials;
}

async function loadLights(spot, angle) {
  console.log('[Scene] Loading lights from data/lights.csv');
  const res = await fetch('data/lights.csv');
  const rows = parseCSV(await res.text());
  console.log('[Scene] Lights CSV rows:', rows.length);
  const lights = rows.map(r => ({
    position: [Number(r.px), Number(r.py), Number(r.pz)],
    intensity: Number(r.intensity),
    color: [Number(r.cr), Number(r.cg), Number(r.cb)],
    spot: spot,
    angle: angle,
    useRaytracedShadows: r.useRaytracedShadows === '1' || r.useRaytracedShadows === 'true',
  }));
  console.log('[Scene] Parsed lights:', lights);
  return lights;
}

// Note: the RAM scene is now exported as an OBJ file under data/scenes/ram.
// The CSV / JSON helpers above are kept for reference but are no longer used
// by the main scene creation path.

function computeMeshesBounds(meshes) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let hasPositions = false;

  for (const mesh of meshes) {
    if (!mesh.positions || mesh.positions.length < 3) continue;
    hasPositions = true;
    const p = mesh.positions;
    for (let i = 0; i < p.length; i += 3) {
      const x = p[i], y = p[i + 1], z = p[i + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
  }

  if (!hasPositions) return null;
  return { minX, minY, minZ, maxX, maxY, maxZ };
}

function fitCameraToScene(scene) {
  if (!scene.meshes || scene.meshes.length === 0) {
    // Fallback to a reasonable default around the origin.
    const s = 0.5;
    scene.camera.target = [0.0, s, 0.0];
    scene.camera.radius = 4.0 * s;
    return;
  }

  const bounds = computeMeshesBounds(scene.meshes);
  if (!bounds) return;

  const center = [
    0.5 * (bounds.minX + bounds.maxX),
    0.5 * (bounds.minY + bounds.maxY),
    0.5 * (bounds.minZ + bounds.maxZ),
  ];

  let radiusSq = 0;
  for (const mesh of scene.meshes) {
    if (!mesh.positions || mesh.positions.length < 3) continue;
    const p = mesh.positions;
    for (let i = 0; i < p.length; i += 3) {
      const dx = p[i] - center[0];
      const dy = p[i + 1] - center[1];
      const dz = p[i + 2] - center[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > radiusSq) radiusSq = d2;
    }
  }

  const radius = Math.max(Math.sqrt(radiusSq), 0.1);

  scene.camera.target = center;
  scene.camera.radius = Math.min(
    Math.max(radius * 2.5, scene.camera.minRadius * 2),
    Math.max(scene.camera.maxRadius, radius * 4.0),
  );
  scene.camera.maxRadius = Math.max(scene.camera.maxRadius, radius * 4.0);
  scene.camera.near = Math.max(radius / 100.0, 0.01);
  scene.camera.far = Math.max(radius * 10.0, scene.camera.near * 10.0);
}

function addRamCeilingLights(scene) {
  const bounds = computeMeshesBounds(scene.meshes);
  if (!bounds) return;

  const centerX = 0.5 * (bounds.minX + bounds.maxX);
  const centerZ = 0.5 * (bounds.minZ + bounds.maxZ);
  const sizeX = (bounds.maxX - bounds.minX) * 1.4;
  const sizeZ = (bounds.maxZ - bounds.minZ) * 1.4;
  const y = bounds.maxY + (bounds.maxY - bounds.minY) * 0.6;
  const targetY = 0.5 * (bounds.minY + bounds.maxY);

  const gridX = 10;
  const gridZ = 10;
  const spacingX = sizeX / gridX;
  const spacingZ = sizeZ / gridZ;

  const baseIntensity = 0.25;
  const color = [1.0, 0.95, 0.9];
  const angle = 0.5;

  let added = 0;
  for (let ix = 0; ix < gridX; ix++) {
    for (let iz = 0; iz < gridZ; iz++) {
      const cx = centerX - sizeX * 0.5 + (ix + 0.5) * spacingX;
      const cz = centerZ - sizeZ * 0.5 + (iz + 0.5) * spacingZ;
      scene.lightSources.push({
        position: [cx, y, cz],
        intensity: baseIntensity,
        color: color,
        spot: [centerX, targetY, centerZ],
        angle: angle,
        useRaytracedShadows: true,
      });
      added++;
    }
  }
  console.log('[Scene] Added RAM ceiling lights:', added, 'total lights =', scene.lightSources.length);
}

function debugLightsAtPoint(label, scene, point) {
  const lights = scene.lightSources || [];
  if (!lights.length) {
    console.log('[Scene][DebugLights]', label, 'no lights');
    return;
  }
  let inCone = 0;
  let sumContribution = 0;
  let maxContribution = 0;
  for (const l of lights) {
    const dx = point[0] - l.position[0];
    const dy = point[1] - l.position[1];
    const dz = point[2] - l.position[2];
    const distSq = dx*dx + dy*dy + dz*dz;
    const dist = Math.sqrt(distSq);
    if (dist <= 0.0) continue;
    const wi = [-dx / dist, -dy / dist, -dz / dist];
    const lx = l.spot[0] - l.position[0];
    const ly = l.spot[1] - l.position[1];
    const lz = l.spot[2] - l.position[2];
    const lenL = Math.sqrt(lx*lx + ly*ly + lz*lz) || 1.0;
    const lightDir = [lx / lenL, ly / lenL, lz / lenL];
    const dotVal = -(wi[0]*lightDir[0] + wi[1]*lightDir[1] + wi[2]*lightDir[2]);
    const spotConeDecay = dotVal - l.angle;
    if (spotConeDecay <= 0.0) continue;
    inCone++;
    const att = spotConeDecay / distSq;
    const contrib = l.intensity * att;
    sumContribution += contrib;
    if (contrib > maxContribution) maxContribution = contrib;
  }
  console.log('[Scene][DebugLights]', label, {
    totalLights: lights.length,
    inCone,
    sumContribution,
    maxContribution,
  });
}

function debugLights(scene) {
  const bounds = computeMeshesBounds(scene.meshes);
  if (!bounds) return;
  const center = [
    0.5 * (bounds.minX + bounds.maxX),
    0.5 * (bounds.minY + bounds.maxY),
    0.5 * (bounds.minZ + bounds.maxZ),
  ];
  const ground = [
    center[0],
    bounds.minY + 0.01,
    center[2],
  ];
  debugLightsAtPoint('center', scene, center);
  debugLightsAtPoint('ground', scene, ground);
}

export async function createScene(camAspect) {
  console.log('[Scene] createScene start, aspect =', camAspect, 'activeScene =', ACTIVE_SCENE);
  const scene = {};
  scene.camera = createCamera(camAspect);
  const tgt = [0.0, 0.5, 0.0];
  const ang = 0.3;

  const materials = await loadMaterials();
  scene.materials = materials;

  // Load OBJ scene placed under data/scenes/<ACTIVE_SCENE>.
  const materialIndex = Math.max(scene.materials.length - 1, 0);
  const objData = await loadOBJScene(ACTIVE_SCENE, materialIndex);
  const meshes = objData.meshes || [];
  const objLights = objData.lights || [];
  scene.meshes = meshes;

  scene.lightSources = await loadLights(tgt, ang);
  if (ACTIVE_SCENE === 'ram' && objLights.length > 0) {
    const bounds = computeMeshesBounds(scene.meshes);
    if (bounds) {
      const centerX = 0.5 * (bounds.minX + bounds.maxX);
      const centerZ = 0.5 * (bounds.minZ + bounds.maxZ);
      const targetY = 0.5 * (bounds.minY + bounds.maxY);
      const color = [1.0, 0.95, 0.9];
      const angle = 0.5;
      const baseIntensity = 0.15;
      let added = 0;
      for (const p of objLights) {
        scene.lightSources.push({
          position: [p[0], p[1], p[2]],
          intensity: baseIntensity,
          color,
          spot: [centerX, targetY, centerZ],
          angle,
          useRaytracedShadows: true,
        });
        added++;
      }
      console.log('[Scene] Added RAM OBJ lights from RamLight faces:', added, 'total lights =', scene.lightSources.length);
    }
  }
  fitCameraToScene(scene);
  scene.time = 0;
  debugLights(scene);
  console.log('[Scene] Scene created:', {
    camera: scene.camera,
    numMeshes: scene.meshes.length,
    numLights: scene.lightSources.length,
    numMaterials: scene.materials.length,
  });
  return scene;
}
