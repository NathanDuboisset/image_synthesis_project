// Node.js helper script to export the RAM test scene
// (quads + RAM mesh) into OBJ/MTL files, similar to the
// conference/sponza assets.
//
// Usage (from project root):
//   node tools/exportRamSceneToObj.js
//
// Output:
//   data/scenes/ram/ram.obj
//   data/scenes/ram/ram.mtl

const fs = require('fs');
const path = require('path');

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const values = line.split(',').map((v) => v.trim());
    const row = {};
    headers.forEach((h, j) => {
      row[h] = values[j] ?? '';
    });
    rows.push(row);
  }
  return rows;
}

function addQuad(origin, e0, e1, positions, indices, vertexOffset) {
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const o = origin;
  const a = add(origin, e0);
  const b = add(a, e1);
  const c = add(origin, e1);

  const verts = [o, a, b, c];
  for (const v of verts) {
    positions.push(v[0], v[1], v[2]);
  }

  indices.push(
    vertexOffset,
    vertexOffset + 1,
    vertexOffset + 2,
    vertexOffset,
    vertexOffset + 2,
    vertexOffset + 3,
  );

  return vertexOffset + 4;
}

function addRamMesh(ramData, positions, indices, vertexOffset) {
  const pos = ramData.positions;
  const idx = ramData.indices;

  // Copy positions
  for (let i = 0; i < pos.length; i++) {
    positions.push(pos[i]);
  }

  // Copy indices with offset
  for (let i = 0; i < idx.length; i++) {
    indices.push(idx[i] + vertexOffset);
  }

  return vertexOffset + pos.length / 3;
}

function computeRamBounds(ramData) {
  const p = ramData.positions;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < p.length; i += 3) {
    const x = p[i];
    const y = p[i + 1];
    const z = p[i + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  return { minX, minY, minZ, maxX, maxY, maxZ };
}

function addLightGridAboveRam(ramData, positions, indices, vertexOffset) {
  const bounds = computeRamBounds(ramData);
  const centerX = 0.5 * (bounds.minX + bounds.maxX);
  const centerZ = 0.5 * (bounds.minZ + bounds.maxZ);
  const sizeX = (bounds.maxX - bounds.minX) * 1.4;
  const sizeZ = (bounds.maxZ - bounds.minZ) * 1.4;
  const y = bounds.maxY + (bounds.maxY - bounds.minY) * 0.6;

  const gridX = 10;
  const gridZ = 10;
  const spacingX = sizeX / gridX;
  const spacingZ = sizeZ / gridZ;
  const tileSizeX = spacingX * 0.4;
  const tileSizeZ = spacingZ * 0.4;

  console.log('[exportRam] Adding light grid above RAM: grid', gridX, 'x', gridZ);

  for (let ix = 0; ix < gridX; ix++) {
    for (let iz = 0; iz < gridZ; iz++) {
      const cx = centerX - sizeX * 0.5 + (ix + 0.5) * spacingX;
      const cz = centerZ - sizeZ * 0.5 + (iz + 0.5) * spacingZ;

      const origin = [cx - tileSizeX * 0.5, y, cz - tileSizeZ * 0.5];
      const e0 = [tileSizeX, 0, 0];
      const e1 = [0, 0, tileSizeZ];

      vertexOffset = addQuad(origin, e0, e1, positions, indices, vertexOffset);
    }
  }

  return vertexOffset;
}

function buildRamSceneGeometry(baseDir) {
  const ramJsonPath = path.join(baseDir, 'data', 'ram-mesh.json');
  const meshesCsvPath = path.join(baseDir, 'data', 'meshes.csv');

  console.log('[exportRam] Reading RAM mesh from', ramJsonPath);
  const ramData = JSON.parse(fs.readFileSync(ramJsonPath, 'utf8'));

  console.log('[exportRam] Reading meshes CSV from', meshesCsvPath);
  const meshesCsv = fs.readFileSync(meshesCsvPath, 'utf8');
  const rows = parseCSV(meshesCsv);

  const positions = [];
  const indices = [];
  let vertexOffset = 0;

  for (const r of rows) {
    if (r.type === 'quad') {
      // Skip the ceiling quad (oy === 1) to keep the box open.
      if (Number(r.oy) === 1) {
        continue;
      }
      const origin = [Number(r.ox), Number(r.oy), Number(r.oz)];
      const e0 = [Number(r.e0x), Number(r.e0y), Number(r.e0z)];
      const e1 = [Number(r.e1x), Number(r.e1y), Number(r.e1z)];
      vertexOffset = addQuad(origin, e0, e1, positions, indices, vertexOffset);
    } else if (r.type === 'ram') {
      vertexOffset = addRamMesh(ramData, positions, indices, vertexOffset);
    }
  }

  // Record how many indices belong to the base geometry (walls + RAM).
  const baseIndexCount = indices.length;

  // Add a grid of small quads above the RAM mesh to mark light sources.
  vertexOffset = addLightGridAboveRam(ramData, positions, indices, vertexOffset);

  console.log(
    '[exportRam] Combined geometry:',
    'vertices =', positions.length / 3,
    'triangles =', indices.length / 3,
  );

  return { positions, indices, baseIndexCount };
}

function writeOBJMTL(baseDir, positions, indices, baseIndexCount) {
  const sceneDir = path.join(baseDir, 'data', 'scenes', 'ram');
  const objPath = path.join(sceneDir, 'ram.obj');
  const mtlPath = path.join(sceneDir, 'ram.mtl');

  fs.mkdirSync(sceneDir, { recursive: true });

  console.log('[exportRam] Writing', objPath);
  const objLines = [];
  objLines.push('# RAM scene exported from ram-mesh.json + meshes.csv');
  objLines.push('mtllib ram.mtl');
  objLines.push('o ram_scene');

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    objLines.push(`v ${x} ${y} ${z}`);
  }

  let currentMat = null;
  for (let i = 0; i < indices.length; i += 3) {
    const isLightFace = i >= baseIndexCount;
    const desiredMat = isLightFace ? 'RamLight' : 'RamMaterial';
    if (desiredMat !== currentMat) {
      objLines.push(`usemtl ${desiredMat}`);
      currentMat = desiredMat;
    }
    const a = indices[i] + 1;
    const b = indices[i + 1] + 1;
    const c = indices[i + 2] + 1;
    objLines.push(`f ${a} ${b} ${c}`);
  }

  fs.writeFileSync(objPath, objLines.join('\n'), 'utf8');

  console.log('[exportRam] Writing', mtlPath);
  const mtlLines = [];
  mtlLines.push('# Simple material for RAM scene');
  mtlLines.push('newmtl RamMaterial');
  // Diffuse color roughly matches one of the default materials.
  mtlLines.push('Kd 0.8 0.8 0.8');
  mtlLines.push('Ks 0.0 0.0 0.0');
  mtlLines.push('Ns 0.0');
  mtlLines.push('d 1.0');
  mtlLines.push('');
  mtlLines.push('newmtl RamLight');
  mtlLines.push('Kd 5.0 5.0 5.0');
  mtlLines.push('Ks 0.0 0.0 0.0');
  mtlLines.push('Ns 0.0');
  mtlLines.push('d 1.0');

  fs.writeFileSync(mtlPath, mtlLines.join('\n'), 'utf8');
}

function main() {
  const baseDir = path.resolve(__dirname, '..');
  const { positions, indices, baseIndexCount } = buildRamSceneGeometry(baseDir);
  writeOBJMTL(baseDir, positions, indices, baseIndexCount);
  console.log('[exportRam] Done.');
}

if (require.main === module) {
  main();
}

