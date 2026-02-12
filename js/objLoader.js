import { computeNormals } from './mesh.js';

/**
 * Minimal OBJ parser.
 * Supports:
 *   - v x y z
 *   - f i j k ...  (triangulated with a simple fan if n > 3)
 *   - usemtl <name> (used to detect light faces)
 * Ignores texture coordinates and normals.
 */
function parseOBJ(text) {
  const positions = [];
  const indices = [];
  const lightPositions = [];

  // Store positions 1-based to make OBJ indexing (1-based, with possible negatives) easier.
  const tempPositions = [null];

  let currentMaterial = '';

  const lines = text.split(/\r?\n/);
  for (let line of lines) {
    line = line.trim();
    if (!line || line.startsWith('#')) continue;

    const parts = line.split(/\s+/);
    const keyword = parts[0];

    if (keyword === 'v') {
      if (parts.length < 4) continue;
      const x = Number(parts[1]);
      const y = Number(parts[2]);
      const z = Number(parts[3]);
      tempPositions.push([x, y, z]);
    } else if (keyword === 'usemtl') {
      currentMaterial = parts[1] || '';
    } else if (keyword === 'f') {
      if (parts.length < 4) continue;
      const faceIndices = [];
      for (let i = 1; i < parts.length; i++) {
        const token = parts[i];
        if (!token) continue;
        const vStr = token.split('/')[0]; // handle v, v/vt, v//vn, v/vt/vn
        let idx = parseInt(vStr, 10);
        if (Number.isNaN(idx)) continue;
        if (idx < 0) idx = tempPositions.length + idx; // negative indices are relative to the end
        faceIndices.push(idx);
      }
      if (faceIndices.length < 3) continue;
      // Triangulate polygon into a fan
      for (let i = 1; i < faceIndices.length - 1; i++) {
        const i0 = faceIndices[0];
        const i1 = faceIndices[i];
        const i2 = faceIndices[i + 1];
        indices.push(
          i0 - 1,
          i1 - 1,
          i2 - 1,
        );
        if (currentMaterial === 'RamLight') {
          const p0 = tempPositions[i0];
          const p1 = tempPositions[i1];
          const p2 = tempPositions[i2];
          if (p0 && p1 && p2) {
            lightPositions.push([
              (p0[0] + p1[0] + p2[0]) / 3,
              (p0[1] + p1[1] + p2[1]) / 3,
              (p0[2] + p1[2] + p2[2]) / 3,
            ]);
          }
        }
      }
    }
  }

  // Flatten positions
  for (let i = 1; i < tempPositions.length; i++) {
    const p = tempPositions[i];
    positions.push(p[0], p[1], p[2]);
  }

  return { positions, indices, lightPositions };
}

/**
 * For certain scenes (currently 'ram') we want to compress
 * light triangles into a single representative point per quad.
 */
function compressSceneLights(sceneName, lightPositions) {
  if (sceneName === 'ram' && lightPositions.length > 0) {
    const compressed = [];
    for (let i = 0; i < lightPositions.length; i += 2) {
      const p0 = lightPositions[i];
      const p1 = lightPositions[i + 1] || p0;
      compressed.push([
        0.5 * (p0[0] + p1[0]),
        0.5 * (p0[1] + p1[1]),
        0.5 * (p0[2] + p1[2]),
      ]);
    }
    console.log('[OBJ] Compressed RAM light triangles:', lightPositions.length, '->', compressed.length);
    return compressed;
  }
  return lightPositions;
}

/**
 * Load a scene from an OBJ file located at:
 *   data/scenes/<sceneName>/<sceneName>.obj
 *
 * Returns:
 *   {
 *     meshes: [ ... ],
 *     lights: [ [x,y,z], ... ]   // one entry per light triangle center
 *   }
 */
export async function loadOBJScene(sceneName, materialIndex = 0) {
  const url = `data/scenes/${sceneName}/${sceneName}.obj`;
  console.log('[OBJ] Loading scene from', url);

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`[OBJ] Failed to load OBJ for scene "${sceneName}": HTTP ${res.status}`);
  }

  const text = await res.text();
  let { positions, indices, lightPositions } = parseOBJ(text);

  // Compress lights for scenes that need it (e.g. RAM quads -> 1 light).
  lightPositions = compressSceneLights(sceneName, lightPositions);

  if (!positions.length || !indices.length) {
    console.warn('[OBJ] Parsed empty geometry for scene', sceneName);
  } else {
    console.log(
      '[OBJ] Parsed scene',
      sceneName,
      'vertices =', positions.length / 3,
      'triangles =', indices.length / 3,
      'lightTriangles =', lightPositions.length,
    );
  }

  const mesh = {
    positions: new Float32Array(positions),
    normals: new Float32Array(positions.length),
    indices: new Uint32Array(indices),
    materialIndex,
  };

  computeNormals(mesh);
  return { meshes: [mesh], lights: lightPositions };
}

/**
 * Load lights from a separate OBJ file, typically:
 *   data/scenes/<sceneName>/lights.obj
 *
 * The OBJ should contain geometry using `usemtl RamLight` for light quads.
 * We parse only the light triangle centers and ignore geometry.
 */
export async function loadOBJLights(sceneName, lightObjName = 'lights') {
  const url = `data/scenes/${sceneName}/${lightObjName}.obj`;
  console.log('[OBJ] Loading separate lights from', url);

  let res;
  try {
    res = await fetch(url);
  } catch (err) {
    console.warn('[OBJ] Failed to fetch separate lights OBJ for scene', sceneName, err);
    return [];
  }

  if (!res.ok) {
    console.log('[OBJ] No separate lights OBJ found for scene', sceneName, '- HTTP', res.status);
    return [];
  }

  const text = await res.text();
  let { lightPositions } = parseOBJ(text);
  lightPositions = compressSceneLights(sceneName, lightPositions);
  console.log('[OBJ] Parsed separate lights for scene', sceneName, 'count =', lightPositions.length);
  return lightPositions;
}

