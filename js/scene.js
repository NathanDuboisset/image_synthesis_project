import { createCamera } from './camera.js';
import { loadOBJScene } from './objLoader.js';
import { createSphere } from './mesh.js';

// Scene names: 'ram' | 'sponza' | 'conference' (must match data/scenes/<name>)

async function loadMaterialsFromMTL(sceneName) {
  const url = `data/scenes/${sceneName}/${sceneName}.mtl`;
  console.log('[Scene] Loading materials from', url);
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    const lines = text.split(/\r?\n/);

    const materialsWithNames = [];
    let current = null;

    for (let raw of lines) {
      const line = raw.trim();
      if (!line || line.startsWith('#')) continue;
      const parts = line.split(/\s+/);
      const kw = parts[0];
      if (kw === 'newmtl') {
        const name = parts[1] || '';
        // Treat materials whose name ends with "Light" as emitter-only; skip them here.
        current = {
          name,
          albedo: [0.8, 0.8, 0.8],
          roughness: 0.5,
          metalness: 0.0,
        };
        if (!/light$/i.test(name)) {
          materialsWithNames.push(current);
        }
      } else if (kw === 'Kd' && current) {
        const r = Number(parts[1]);
        const g = Number(parts[2]);
        const b = Number(parts[3]);
        if (!Number.isNaN(r) && !Number.isNaN(g) && !Number.isNaN(b)) {
          current.albedo = [r, g, b];
        }
      }
    }

    if (!materialsWithNames.length) {
      console.warn('[Scene] No non-light materials found in MTL, using default.');
      materialsWithNames.push({
        name: 'Default',
        albedo: [0.8, 0.8, 0.8],
        roughness: 0.5,
        metalness: 0.0,
      });
    }

    const materials = materialsWithNames.map(m => ({
      albedo: m.albedo,
      roughness: m.roughness,
      metalness: m.metalness,
    }));
    console.log('[Scene] Parsed MTL materials:', materialsWithNames);
    return materials;
  } catch (err) {
    console.error('[Scene] Failed to load MTL materials, using fallback.', err);
    return [{
      albedo: [0.8, 0.8, 0.8],
      roughness: 0.5,
      metalness: 0.0,
    }];
  }
}

// Note: at runtime we now rely only on OBJ + MTL files.

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
    const distSq = dx * dx + dy * dy + dz * dz;
    const dist = Math.sqrt(distSq);
    if (dist <= 0.0) continue;
    const wi = [-dx / dist, -dy / dist, -dz / dist];
    const lx = l.spot[0] - l.position[0];
    const ly = l.spot[1] - l.position[1];
    const lz = l.spot[2] - l.position[2];
    const lenL = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1.0;
    const lightDir = [lx / lenL, ly / lenL, lz / lenL];
    const dotVal = -(wi[0] * lightDir[0] + wi[1] * lightDir[1] + wi[2] * lightDir[2]);
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
  const checkbox = typeof document !== 'undefined' ? document.getElementById('debug_lights_checkbox') : null;
  if (!checkbox || !checkbox.checked) return;
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

function addDebugLightMeshes(scene) {
  const lights = scene.lightSources || [];
  if (!lights.length || typeof document === 'undefined') return;

  // Remember how many "real" meshes we have so we can keep ray tracing
  // operating only on the main geometry.
  if (typeof scene.baseMeshCount !== 'number') {
    scene.baseMeshCount = scene.meshes.length;
  }

  // Add a bright material for light markers.
  const debugMaterialIndex = scene.materials.length;
  scene.materials.push({
    albedo: [1.0, 0.9, 0.2],
    roughness: 0.2,
    metalness: 0.0,
  });

  // Small sphere used as a template, then translated to each light position.
  const radius = 0.02;
  const sphere = createSphere(radius, 8, 8);
  const basePositions = sphere.positions;
  const baseNormals = sphere.normals;
  const baseIndices = sphere.indices;

  scene.debugLightMeshStart = scene.meshes.length;

  for (const l of lights) {
    const [cx, cy, cz] = l.position;
    const vertCount = basePositions.length / 3;

    const positions = new Float32Array(basePositions.length);
    for (let i = 0; i < vertCount; i++) {
      const px = basePositions[3 * i];
      const py = basePositions[3 * i + 1];
      const pz = basePositions[3 * i + 2];
      positions[3 * i] = px + cx;
      positions[3 * i + 1] = py + cy;
      positions[3 * i + 2] = pz + cz;
    }

    const normals = new Float32Array(baseNormals.length);
    normals.set(baseNormals);

    const indices = new Uint32Array(baseIndices.length);
    indices.set(baseIndices);

    scene.meshes.push({
      positions,
      normals,
      indices,
      materialIndex: debugMaterialIndex,
    });
  }

  console.log('[Scene] Added debug light meshes:', scene.meshes.length - scene.baseMeshCount);
}

/**
 * Set camera to a top-down view (from above) with optional yaw (azimuth) in radians.
 * Call after createScene / fitCameraToScene so radius and target are set.
 */
export function setCameraTopDown(scene, yawRad = 0) {
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
      const dx = p[i] - center[0], dy = p[i + 1] - center[1], dz = p[i + 2] - center[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > radiusSq) radiusSq = d2;
    }
  }
  const radius = Math.max(Math.sqrt(radiusSq), 0.1);
  scene.camera.target = center;
  scene.camera.radius = Math.min(Math.max(radius * 2.5, scene.camera.minRadius * 2), scene.camera.maxRadius);
  scene.camera.pitch = Math.PI / 2;
  scene.camera.yaw = yawRad;
}

export async function createScene(camAspect, sceneName = 'ram') {
  console.log('[Scene] createScene start, aspect =', camAspect, 'sceneName =', sceneName);
  const scene = {};
  scene.camera = createCamera(camAspect);

  // Load materials directly from the scene's MTL file.
  scene.materials = await loadMaterialsFromMTL(sceneName);

  // Load OBJ scene placed under data/scenes/<sceneName>.
  const materialIndex = Math.max(scene.materials.length - 1, 0);
  const objData = await loadOBJScene(sceneName, materialIndex);
  const meshes = objData.meshes || [];
  const objLights = objData.lights || [];
  scene.meshes = meshes;
  scene.baseMeshCount = meshes.length;

  scene.lightSources = [];

  const bounds = computeMeshesBounds(scene.meshes);
  if (sceneName === 'ram' && objLights.length > 0 && bounds) {
    const centerX = 0.5 * (bounds.minX + bounds.maxX);
    const centerZ = 0.5 * (bounds.minZ + bounds.maxZ);
    const targetY = 0.5 * (bounds.minY + bounds.maxY);
    const color = [1.0, 0.95, 0.9];
    const angle = 0.5;
    const baseIntensity = 0.05;
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
  } else if (bounds) {
    // Fallback single light above the scene center.
    const center = [
      0.5 * (bounds.minX + bounds.maxX),
      0.5 * (bounds.minY + bounds.maxY),
      0.5 * (bounds.minZ + bounds.maxZ),
    ];
    const y = bounds.maxY + (bounds.maxY - bounds.minY) * 0.6;
    scene.lightSources.push({
      position: [center[0], y, center[2]],
      intensity: 1.0,
      color: [1.0, 1.0, 1.0],
      spot: center,
      angle: 0.5,
      useRaytracedShadows: true,
    });
    console.log('[Scene] Added fallback light above scene center, total lights =', scene.lightSources.length);
  }
  fitCameraToScene(scene);

  // Build small debug meshes at each light position so they can be
  // toggled on in raster mode.
  addDebugLightMeshes(scene);
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
