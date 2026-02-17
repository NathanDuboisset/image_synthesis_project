// Utility to create a light panel OBJ
// Usage: node js/addLightsPanel.js <sceneName> <textureName> <lightsPerSide> <height> <minX> <maxX> <minY> <maxY>

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
  totalIntensity = 1000.0
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

  // Output to vlights.txt (or lights<N>.txt if we want to keep specific N naming, 
  // but simpler to use vlights.txt if that's the standard now. 
  // User asked to put it in "vlights.txt" for virtual lights. 
  // This script seems to generate specific grid. 
  // Let's output to vlights.txt to match the request "put it in a file vlights.txt").
  // But wait, "lights50.txt" suggests user might want multiple configs?
  // User said "update also @[src/addLightsPanel.js] so it similarly creates these virtual ligths".
  // Let's use `vlights.txt` as standard.
  const outPath = path.join(sceneDir, `vlights.txt`);

  // Create directory if needed
  if (!fs.existsSync(sceneDir)) {
    fs.mkdirSync(sceneDir, { recursive: true });
  }

  const lines = [];
  const perLightIntensity = totalIntensity / nblights;

  // Color: User said "using the reflectiveness etc". 
  // For a light panel, it's usually emissive white or based on texture name.
  // We'll assume white [1,1,1] for now as it's an emitter.
  // Or maybe parse textureName? e.g. "RamLight" -> white.
  const r = 1.0, g = 1.0, b = 1.0;

  for (let j = 0; j < n; j++) {
    // 0..1 parameter
    const tj = n > 1 ? j / (n - 1) : 0.5;
    const z = lerp(minY, maxY, tj);

    for (let i = 0; i < n; i++) {
      const ti = n > 1 ? i / (n - 1) : 0.5;
      const x = lerp(minX, maxX, ti);
      const y = height;

      // Format: x,y,z,intensity,r,g,b
      lines.push(`${x.toFixed(4)},${y.toFixed(4)},${z.toFixed(4)},${perLightIntensity.toFixed(4)},${r},${g},${b}`);
    }
  }

  const content = lines.join('\n') + '\n';
  fs.writeFileSync(outPath, content, 'utf8');
  console.log(`[addLightsPanel] Wrote vlights.txt to ${outPath}`);
}

const args = process.argv.slice(2);
if (args.length >= 8) {
  const [sceneName, textureName, lightsPerSide, height, minX, maxX, minY, maxY, totalIntensityStr] = args;
  const totalIntensity = totalIntensityStr ? Number(totalIntensityStr) : 1000.0;

  createLightPanel(
    sceneName,
    textureName,
    Number(lightsPerSide),
    Number(height),
    Number(minX),
    Number(maxX),
    Number(minY),
    Number(maxY),
    totalIntensity
  );
} else if (args.length > 0) {
  console.error('Usage: node js/addLightsPanel.js <sceneName> <textureName> <lightsPerSide> <height> <minX> <maxX> <minY> <maxY> [totalIntensity]');
  process.exit(1);
}
