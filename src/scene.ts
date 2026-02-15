import type { Vec3, Scene, Material, Mesh, LightSource, CameraConfig, SceneBounds, NamedMaterial } from './types.ts';
import { createCamera } from './camera.ts';
import { loadOBJScene, loadOBJLights } from './objLoader.ts';

// Scene names: 'ram' | 'sponza' | 'conference' (must match data/scenes/<name>)

async function loadMaterialsFromMTL(sceneName: string): Promise<Material[]> {
  const url = `data/scenes/${sceneName}/${sceneName}.mtl`;
  console.log('[Scene] Loading materials from', url);
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    const lines = text.split(/\r?\n/);

    const materialsWithNames: NamedMaterial[] = [];
    let current: NamedMaterial | null = null;

    for (const raw of lines) {
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

    const materials: Material[] = materialsWithNames.map(m => ({
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

function computeMeshesBounds(meshes: Mesh[]): SceneBounds | null {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let hasPositions = false;

  for (const mesh of meshes) {
    if (!mesh.positions || mesh.positions.length < 3) continue;
    hasPositions = true;
    const p = mesh.positions;
    for (let i = 0; i < p.length; i += 3) {
      const x = p[i]!, y = p[i + 1]!, z = p[i + 2]!;
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

function fitCameraToScene(scene: Scene): void {
  if (!scene.meshes || scene.meshes.length === 0) {
    const s = 0.5;
    scene.camera.target = [0.0, s, 0.0];
    scene.camera.radius = 4.0 * s;
    return;
  }

  const bounds = computeMeshesBounds(scene.meshes);
  if (!bounds) return;

  const center: Vec3 = [
    0.5 * (bounds.minX + bounds.maxX),
    0.5 * (bounds.minY + bounds.maxY),
    0.5 * (bounds.minZ + bounds.maxZ),
  ];

  let radiusSq = 0;
  for (const mesh of scene.meshes) {
    if (!mesh.positions || mesh.positions.length < 3) continue;
    const p = mesh.positions;
    for (let i = 0; i < p.length; i += 3) {
      const dx = p[i]! - center[0];
      const dy = p[i + 1]! - center[1];
      const dz = p[i + 2]! - center[2];
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

async function loadCameraConfig(sceneName: string): Promise<CameraConfig | null> {
  const url = `data/scenes/${sceneName}/camera.txt`;
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const text = await res.text();
    const config: CameraConfig = {};
    for (const line of text.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const eq = trimmed.indexOf('=');
      if (eq <= 0) continue;
      const key = trimmed.slice(0, eq).trim();
      const value = trimmed.slice(eq + 1).trim();
      const num = Number(value);
      config[key] = Number.isNaN(num) ? value : num;
    }
    if (Object.keys(config).length === 0) return null;
    return config;
  } catch {
    return null;
  }
}

function applyCameraConfig(scene: Scene, config: CameraConfig): void {
  if (config.radiusScale != null) {
    scene.camera.radius *= config.radiusScale;
    scene.camera.radius = Math.max(scene.camera.minRadius, Math.min(scene.camera.maxRadius, scene.camera.radius));
  }
  if (config.radius != null) {
    scene.camera.radius = Math.max(scene.camera.minRadius, Math.min(scene.camera.maxRadius, config.radius));
  }
  if (config.yaw != null) scene.camera.yaw = config.yaw;
  if (config.pitch != null) scene.camera.pitch = config.pitch;
  if (config.targetX != null) scene.camera.target[0] = config.targetX;
  if (config.targetY != null) scene.camera.target[1] = config.targetY;
  if (config.targetZ != null) scene.camera.target[2] = config.targetZ;
}

function applyCameraConfigRadius(scene: Scene): void {
  const config = scene.cameraConfig;
  if (!config) return;
  if (config.radiusScale != null) {
    scene.camera.radius *= config.radiusScale;
    scene.camera.radius = Math.max(scene.camera.minRadius, Math.min(scene.camera.maxRadius, scene.camera.radius));
  }
  if (config.radius != null) {
    scene.camera.radius = Math.max(scene.camera.minRadius, Math.min(scene.camera.maxRadius, config.radius));
  }
}

function debugLightsAtPoint(label: string, scene: Scene, point: Vec3): void {
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
    const wi: Vec3 = [-dx / dist, -dy / dist, -dz / dist];
    const lx = l.spot[0] - l.position[0];
    const ly = l.spot[1] - l.position[1];
    const lz = l.spot[2] - l.position[2];
    const lenL = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1.0;
    const lightDir: Vec3 = [lx / lenL, ly / lenL, lz / lenL];
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

function debugLights(scene: Scene): void {
  const bounds = computeMeshesBounds(scene.meshes);
  if (!bounds) return;
  const center: Vec3 = [
    0.5 * (bounds.minX + bounds.maxX),
    0.5 * (bounds.minY + bounds.maxY),
    0.5 * (bounds.minZ + bounds.maxZ),
  ];
  const ground: Vec3 = [
    center[0],
    bounds.minY + 0.01,
    center[2],
  ];
  debugLightsAtPoint('center', scene, center);
  debugLightsAtPoint('ground', scene, ground);
}

function addDebugLightMeshes(scene: Scene): void {
  const lights = scene.lightSources || [];
  if (!lights.length || typeof document === 'undefined') return;

  if (typeof scene.baseMeshCount !== 'number') {
    scene.baseMeshCount = scene.meshes.length;
  }

  const debugMaterialIndex = scene.materials.length;
  scene.materials.push({
    albedo: [1.0, 0.9, 0.2],
    roughness: 0.2,
    metalness: 0.0,
  });

  const radius = 0.05;
  const basePositions = new Float32Array([
    -radius, 0.0, -radius,
    radius, 0.0, -radius,
    0.0, 0.0, radius,
  ]);
  const baseNormals = new Float32Array([
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
  ]);
  const baseIndices = new Uint32Array([0, 1, 2, 0, 2, 1]);

  scene.debugLightMeshStart = scene.meshes.length;

  for (const l of lights) {
    const [cx, cy, cz] = l.position;
    const vertCount = basePositions.length / 3;

    const positions = new Float32Array(basePositions.length);
    for (let i = 0; i < vertCount; i++) {
      const px = basePositions[3 * i]!;
      const py = basePositions[3 * i + 1]!;
      const pz = basePositions[3 * i + 2]!;
      positions[3 * i] = px + cx;
      positions[3 * i + 1] = py + cy - 0.03;
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

  console.log('[Scene] Added debug light meshes:', scene.meshes.length - (scene.baseMeshCount ?? 0));
}

function getSceneCenterAndRadius(meshes: Mesh[]): { center: Vec3; radius: number } | null {
  const bounds = computeMeshesBounds(meshes);
  if (!bounds) return null;
  const center: Vec3 = [
    0.5 * (bounds.minX + bounds.maxX),
    0.5 * (bounds.minY + bounds.maxY),
    0.5 * (bounds.minZ + bounds.maxZ),
  ];
  let radiusSq = 0;
  for (const mesh of meshes) {
    if (!mesh.positions || mesh.positions.length < 3) continue;
    const p = mesh.positions;
    for (let i = 0; i < p.length; i += 3) {
      const dx = p[i]! - center[0], dy = p[i + 1]! - center[1], dz = p[i + 2]! - center[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > radiusSq) radiusSq = d2;
    }
  }
  const radius = Math.max(Math.sqrt(radiusSq), 0.1);
  return { center, radius };
}

export function setCameraTopDown(scene: Scene, yawRad: number = 0): void {
  const data = getSceneCenterAndRadius(scene.meshes);
  if (!data) return;
  const { center, radius } = data;
  scene.camera.target = center;
  scene.camera.radius = Math.min(Math.max(radius * 2.5, scene.camera.minRadius * 2), scene.camera.maxRadius);
  scene.camera.pitch = Math.PI / 2;
  scene.camera.yaw = yawRad;
  applyCameraConfigRadius(scene);
}

export function setCameraRandomNorthHemisphere(scene: Scene): void {
  const data = getSceneCenterAndRadius(scene.meshes);
  if (!data) return;
  const { center, radius } = data;
  scene.camera.target = center;
  scene.camera.radius = Math.min(Math.max(radius * 2.5, scene.camera.minRadius * 2), scene.camera.maxRadius);
  scene.camera.pitch = Math.random() * (Math.PI / 2);
  scene.camera.yaw = Math.random() * 2 * Math.PI;
  applyCameraConfigRadius(scene);
}

export async function createScene(camAspect: number, sceneName: string = 'ram'): Promise<Scene> {
  console.log('[Scene] createScene start, aspect =', camAspect, 'sceneName =', sceneName);
  const scene: Scene = {
    camera: createCamera(camAspect),
    meshes: [],
    materials: [],
    lightSources: [],
  };

  // Load materials directly from the scene's MTL file.
  scene.materials = await loadMaterialsFromMTL(sceneName);

  // Load OBJ scene placed under data/scenes/<sceneName>.
  const materialIndex = Math.max(scene.materials.length - 1, 0);
  const objData = await loadOBJScene(sceneName, materialIndex);
  const meshes = objData.meshes || [];
  let objLights = objData.lights || [];
  scene.meshes = meshes;
  scene.baseMeshCount = meshes.length;

  scene.lightSources = [];

  const bounds = computeMeshesBounds(scene.meshes);

  // Prefer lights from a separate OBJ (data/scenes/<sceneName>/lights.obj) if present.
  try {
    const separateLights = await loadOBJLights(sceneName, 'lights');
    if (separateLights && separateLights.length > 0) {
      console.log('[Scene] Using lights from separate OBJ file for scene', sceneName);
      objLights = separateLights;
    }
  } catch (err) {
    console.warn('[Scene] Failed to load separate lights OBJ for scene', sceneName, err);
  }

  if (sceneName === 'ram' && objLights.length > 0 && bounds) {
    const centerX = 0.5 * (bounds.minX + bounds.maxX);
    const centerZ = 0.5 * (bounds.minZ + bounds.maxZ);
    const targetY = 0.5 * (bounds.minY + bounds.maxY);
    const color: Vec3 = [1.0, 0.95, 0.9];
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
    const center: Vec3 = [
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
  const cameraConfig = await loadCameraConfig(sceneName);
  if (cameraConfig) {
    scene.cameraConfig = cameraConfig;
    applyCameraConfig(scene, cameraConfig);
    console.log('[Scene] Applied camera config from camera.txt');
  } else {
    scene.cameraConfig = null;
  }

  // Normalize light intensities for energy conservation
  const BASE_TOTAL_LUMINANCE = 2.0;
  const numLights = Math.max(1, scene.lightSources.length);
  const perLightIntensity = BASE_TOTAL_LUMINANCE / numLights;
  for (const l of scene.lightSources) {
    l.intensity = perLightIntensity;
  }

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
