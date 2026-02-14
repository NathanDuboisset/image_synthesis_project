// Utility to generate a grid "panel" of emissive faces over a rectangular area
// and write them as an OBJ snippet under the scene folder
// Usage (Node.js only, not in browser):
//   node js/addLightsPanel.js <sceneName> <textureName> <lightsPerSide> <height> <minX> <maxX> <minY> <maxY>
// Example (ram, 20 per side, height 1.4, X/Z -1.5 to 1.5):
//   node js/addLightsPanel.js ram RamLight 20 1.4 -1.5 1.5 -1.5 1.5

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function createLightPanel(
  sceneName,
  textureName,
  lightsPerSide,
  height,
  minX,
  maxX,
  minY,
  maxY,
) {
  if (typeof sceneName !== 'string' || !sceneName.length) {
    throw new Error('sceneName must be a non-empty string');
  }
  if (typeof textureName !== 'string' || !textureName.length) {
    throw new Error('textureName must be a non-empty string');
  }
  const n = Math.max(1, (lightsPerSide | 0));
  const nblights = n * n;

  const thisFile = fileURLToPath(import.meta.url);
  const projectRoot = path.resolve(path.dirname(thisFile), '..');
  const sceneDir = path.join(projectRoot, 'data', 'scenes', sceneName);
  const outPath = path.join(sceneDir, `lights${nblights}.txt`);
  const objPath = path.join(sceneDir, `${sceneName}.obj`);

  // Ensure scene directory and OBJ exist.
  if (!fs.existsSync(sceneDir)) {
    throw new Error(`Scene directory does not exist: ${sceneDir}`);
  }
  if (!fs.existsSync(objPath)) {
    throw new Error(`Scene OBJ file does not exist: ${objPath}`);
  }
  const baseVertexIndex = 0;

  const vertsPerSide = n + 1;

  const lines = [];
  lines.push(`usemtl ${textureName}`);

  // v x y z
  for (let j = 0; j < vertsPerSide; j++) {
    const tj = vertsPerSide > 1 ? j / (vertsPerSide - 1) : 0.5;
    const z = lerp(minY, maxY, tj);
    for (let i = 0; i < vertsPerSide; i++) {
      const ti = vertsPerSide > 1 ? i / (vertsPerSide - 1) : 0.5;
      const x = lerp(minX, maxX, ti);
      const y = height;
      lines.push(`v ${x} ${y} ${z}`);
    }
  }
  const vIndex = (i, j) => 1 + j * vertsPerSide + i;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const v00 = baseVertexIndex + vIndex(i, j);
      const v10 = baseVertexIndex + vIndex(i + 1, j);
      const v11 = baseVertexIndex + vIndex(i + 1, j + 1);
      const v01 = baseVertexIndex + vIndex(i, j + 1);
      lines.push(`f ${v00} ${v10} ${v11}`);
      lines.push(`f ${v00} ${v11} ${v01}`);
    }
  }

  const content = lines.join('\n') + '\n';
  fs.writeFileSync(outPath, content, 'utf8');
  console.log(`[addLightsPanel] Wrote OBJ light panel to ${outPath}`);
}

const args = process.argv.slice(2);
if (args.length >= 8) {
  const [sceneName, textureName, lightsPerSide, height, minX, maxX, minY, maxY] = args;
  createLightPanel(
    sceneName,
    textureName,
    Number(lightsPerSide),
    Number(height),
    Number(minX),
    Number(maxX),
    Number(minY),
    Number(maxY),
  );
} else if (args.length > 0) {
  console.error('Usage: node js/addLightsPanel.js <sceneName> <textureName> <lightsPerSide> <height> <minX> <maxX> <minY> <maxY>');
  process.exit(1);
}
