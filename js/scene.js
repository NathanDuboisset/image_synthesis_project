import { createCamera } from './camera.js';
import { createQuad, createRamFromData } from './mesh.js';

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

async function loadMeshes(ramData) {
  console.log('[Scene] Loading meshes from data/meshes.csv');
  const res = await fetch('data/meshes.csv');
  const rows = parseCSV(await res.text());
  console.log('[Scene] Meshes CSV rows:', rows.length, rows);
  const meshes = [];
  for (const r of rows) {
    if (r.type === 'ram') {
      const mesh = createRamFromData(ramData);
      mesh.materialIndex = Number(r.material_index);
      meshes.push(mesh);
    } else if (r.type === 'quad') {
      const mesh = createQuad(
        [Number(r.ox), Number(r.oy), Number(r.oz)],
        [Number(r.e0x), Number(r.e0y), Number(r.e0z)],
        [Number(r.e1x), Number(r.e1y), Number(r.e1z)]
      );
      mesh.materialIndex = Number(r.material_index);
      meshes.push(mesh);
    }
  }
  console.log('[Scene] Built meshes:', meshes);
  return meshes;
}

export async function createScene(camAspect) {
  console.log('[Scene] createScene start, aspect =', camAspect);
  const scene = {};
  const s = 0.5;
  scene.camera = createCamera(camAspect);
  scene.camera.target = [0.0, s, 0.0];
  scene.camera.radius = 4.0 * s;

  const tgt = [0.0, s, 0.0];
  const ang = 0.3;

  const [materials, ramData] = await Promise.all([
    loadMaterials(),
    fetch('data/ram-mesh.json').then(r => {
      console.log('[Scene] Loading RAM mesh from data/ram-mesh.json');
      return r.json();
    }),
  ]);

  console.log('[Scene] RAM mesh data:', ramData ? { positions: ramData.positions.length, indices: ramData.indices.length } : null);

  scene.lightSources = await loadLights(tgt, ang);
  scene.materials = materials;
  scene.meshes = await loadMeshes(ramData);
  scene.time = 0;
  console.log('[Scene] Scene created:', {
    camera: scene.camera,
    numMeshes: scene.meshes.length,
    numLights: scene.lightSources.length,
    numMaterials: scene.materials.length,
  });
  return scene;
}
