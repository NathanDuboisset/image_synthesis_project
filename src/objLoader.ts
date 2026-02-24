import type { Vec3, Mesh, ParsedOBJ, OBJSceneResult, NamedMaterial } from './types.ts';
import { computeNormals } from './mesh.ts';

// Minimal OBJ parser.
// Supports: v, f, usemtl
export function parseOBJ(text: string): { positions: number[]; indicesByMaterial: Map<string, number[]>; lightPositions: Vec3[] } {
  const positions: number[] = [];
  const indicesByMaterial = new Map<string, number[]>();
  const lightPositions: Vec3[] = [];

  // Store positions 1-based to make OBJ indexing (1-based, with possible negatives) easier.
  const tempPositions: (Vec3 | null)[] = [null];

  let currentMaterial = 'default';

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
      currentMaterial = parts[1] || 'default';
    } else if (keyword === 'f') {
      if (parts.length < 4) continue;
      const faceIndices: number[] = [];
      for (let i = 1; i < parts.length; i++) {
        const token = parts[i];
        if (!token) continue;
        const vStr = token.split('/')[0]!; // handle v, v/vt, v//vn, v/vt/vn
        let idx = parseInt(vStr, 10);
        if (Number.isNaN(idx)) continue;
        if (idx < 0) idx = tempPositions.length + idx; // negative indices are relative to the end
        faceIndices.push(idx);
      }
      if (faceIndices.length < 3) continue;

      let targetIndices = indicesByMaterial.get(currentMaterial);
      if (!targetIndices) {
        targetIndices = [];
        indicesByMaterial.set(currentMaterial, targetIndices);
      }

      // Triangulate polygon into a fan
      for (let i = 1; i < faceIndices.length - 1; i++) {
        const i0 = faceIndices[0]!;
        const i1 = faceIndices[i]!;
        const i2 = faceIndices[i + 1]!;
        targetIndices.push(
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
    const p = tempPositions[i]!;
    positions.push(p[0], p[1], p[2]);
  }

  return { positions, indicesByMaterial, lightPositions };
}

// Compress light triangles into a single point for "ram" scene
function compressSceneLights(sceneName: string, lightPositions: Vec3[]): Vec3[] {
  if (sceneName === 'ram' && lightPositions.length > 0) {
    const compressed: Vec3[] = [];
    for (let i = 0; i < lightPositions.length; i += 2) {
      const p0 = lightPositions[i]!;
      const p1 = lightPositions[i + 1] ?? p0;
      compressed.push([
        0.5 * (p0[0] + p1[0]),
        0.5 * (p0[1] + p1[1]),
        0.5 * (p0[2] + p1[2]),
      ]);
    }
    console.log('compressed RAM lights:', lightPositions.length, '->', compressed.length);
    return compressed;
  }
  return lightPositions;
}

// Load OBJ file
export async function loadOBJScene(sceneName: string, materials: NamedMaterial[] = []): Promise<OBJSceneResult> {
  const url = `data/scenes/${sceneName}/${sceneName}.obj`;
  console.log('loading OBJ from', url);

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`failed to load OBJ for "${sceneName}": HTTP ${res.status}`);
  }

  const text = await res.text();
  let { positions, indicesByMaterial, lightPositions } = parseOBJ(text);

  // Compress lights for scenes that need it (e.g. RAM quads -> 1 light).
  lightPositions = compressSceneLights(sceneName, lightPositions);

  const meshes: Mesh[] = [];

  if (indicesByMaterial.size === 0) {
    console.warn('parsed empty geometry for scene', sceneName);
  } else {
    // We need to create separate meshes for each material subset
    // Since positions are shared, we can either duplicate positions for each mesh (simple)
    // or try to keep them shared (but Mesh struct implies separate buffers).
    // Let's duplicate positions for now to be safe and simple, optimization later.

    for (const [matName, matIndices] of indicesByMaterial.entries()) {
      if (matIndices.length === 0) continue;

      // Find material index
      let materialIndex = 0;
      const foundMatIndex = materials.findIndex(m => m.name === matName);
      if (foundMatIndex >= 0) {
        materialIndex = foundMatIndex;
      } else {
        // Fallback or default
        // If "default", maybe index 0?
        // If unknown, maybe index 0?
        // Try to match 'Default' if not found
        const defIdx = materials.findIndex(m => m.name === 'Default');
        if (defIdx >= 0) materialIndex = defIdx;
      }

      // Re-map indices to be 0-based relative to the subset of vertices?
      // Actually, standard engines re-index vertices to be compact.
      // But here we can just dump all positions and use the original indices if we want,
      // BUT `Mesh` expects `positions` and `indices`. If we use ALL positions for EVERY mesh,
      // it's wasteful but correct.
      // Let's try attempting to implement a compaction if it's not too complex.
      // Compaction is cleaner.

      const subPositions: number[] = [];
      const subIndices: number[] = [];
      const indexMap = new Map<number, number>(); // old index -> new index

      for (const oldIdx of matIndices) {
        let newIdx = indexMap.get(oldIdx);
        if (newIdx === undefined) {
          newIdx = subPositions.length / 3;
          indexMap.set(oldIdx, newIdx);
          subPositions.push(
            positions[3 * oldIdx]!,
            positions[3 * oldIdx + 1]!,
            positions[3 * oldIdx + 2]!
          );
        }
        subIndices.push(newIdx);
      }

      const mesh: Mesh = {
        positions: new Float32Array(subPositions),
        normals: new Float32Array(subPositions.length),
        indices: new Uint32Array(subIndices),
        materialIndex,
      };

      computeNormals(mesh);
      meshes.push(mesh);
    }

    console.log(
      'parsed', sceneName,
      '- meshes:', meshes.length,
      'triangles:', Array.from(indicesByMaterial.values()).reduce((a, b) => a + b.length, 0) / 3,
      'light triangles:', lightPositions.length,
    );
  }

  return { meshes, lights: lightPositions };
}

// Load separate lights OBJ if valid
export async function loadOBJLights(sceneName: string, lightObjName: string = 'lights'): Promise<Vec3[]> {
  const url = `data/scenes/${sceneName}/${lightObjName}.obj`;
  console.log('loading lights OBJ from', url);

  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    console.warn('failed to fetch lights OBJ for scene', sceneName, err);
    return [];
  }

  if (!res.ok) {
    console.log('no lights OBJ for', sceneName, '- HTTP', res.status);
    return [];
  }

  const text = await res.text();
  let { lightPositions } = parseOBJ(text);
  lightPositions = compressSceneLights(sceneName, lightPositions);
  console.log('lights for', sceneName, ':', lightPositions.length);
  return lightPositions;
}
