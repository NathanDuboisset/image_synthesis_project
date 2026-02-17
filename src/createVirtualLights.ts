// @ts-nocheck
import * as fs from 'fs';
import * as path from 'path';
import { generateVirtualLights } from './virtual_lights.ts';
import { parseOBJ } from './objLoader.ts';
import { computeNormals } from './mesh.ts';
import type { Scene, Mesh, LightSource, Vec3 } from './types.ts';

console.log('[DEBUG] Script starting...');
async function main() {
    const args = process.argv.slice(2);
    if (args.length < 1) {
        console.error('Usage: tsx src/createVirtualLights.ts <sceneName>');
        process.exit(1);
    }

    const sceneName = args[0];
    const projectRoot = process.cwd();
    const sceneDir = path.join(projectRoot, 'data', 'scenes', sceneName);
    const objPath = path.join(sceneDir, `${sceneName}.obj`);
    const paramsPath = path.join(sceneDir, 'params.txt');

    if (!fs.existsSync(sceneDir) || !fs.existsSync(objPath)) {
        console.error(`Scene not found: ${sceneDir}`);
        process.exit(1);
    }

    // Load params
    let params: any = {};
    if (fs.existsSync(paramsPath)) {
        const text = fs.readFileSync(paramsPath, 'utf-8');
        for (const line of text.split(/\r?\n/)) {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) continue;
            const eq = trimmed.indexOf('=');
            if (eq > 0) {
                const key = trimmed.slice(0, eq).trim();
                const value = trimmed.slice(eq + 1).trim();
                if (key === 'defaultLightPos' || key === 'defaultLightColor') {
                    params[key] = value;
                } else if (key === 'do_virtual') {
                    params[key] = (value.toLowerCase() === 'true' || value === '1');
                } else {
                    const num = Number(value);
                    params[key] = Number.isNaN(num) ? value : num;
                }
            }
        }
    }

    if (!params.do_virtual) {
        console.log(`[VirtualLights] 'do_virtual' is false in params.txt. Skipping generation.`);
        process.exit(0);
    }

    const div = params.virtual_dir_div || 100;
    console.log(`[VirtualLights] Generating lights for scene '${sceneName}' with div=${div}`);

    // Load Materials
    const mtlPath = path.join(sceneDir, `${sceneName}.mtl`);
    const materials: { name: string; albedo: Vec3 }[] = [];

    // Default material
    materials.push({ name: 'Default', albedo: [0.8, 0.8, 0.8] });

    if (fs.existsSync(mtlPath)) {
        console.log('[VirtualLights] Loading materials from', mtlPath);
        const mtlText = fs.readFileSync(mtlPath, 'utf-8');
        let currentMat: { name: string; albedo: Vec3 } | null = null;

        for (const line of mtlText.split(/\r?\n/)) {
            const parts = line.trim().split(/\s+/);
            if (parts.length < 2) continue;
            const kw = parts[0];

            if (kw === 'newmtl') {
                const name = parts[1] || 'Unknown';
                currentMat = { name, albedo: [0.8, 0.8, 0.8] };
                materials.push(currentMat);
            } else if (kw === 'Kd' && currentMat) {
                const r = Number(parts[1]);
                const g = Number(parts[2]);
                const b = Number(parts[3]);
                if (!Number.isNaN(r)) currentMat.albedo = [r, g, b];
            }
        }
        console.log(`[VirtualLights] Loaded ${materials.length} materials.`);
    } else {
        console.warn(`[VirtualLights] No MTL file found at ${mtlPath}`);
    }

    // Load Mesh
    const objText = fs.readFileSync(objPath, 'utf-8');
    const { positions, indicesByMaterial } = parseOBJ(objText);

    const meshes: Mesh[] = [];
    for (const [matName, matIndices] of indicesByMaterial.entries()) {
        if (matIndices.length === 0) continue;

        // Find material index
        let matIndex = materials.findIndex(m => m.name === matName);
        if (matIndex === -1) matIndex = 0; // Default

        const subPositions: number[] = [];
        const subIndices: number[] = [];
        const indexMap = new Map<number, number>();

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
            materialIndex: matIndex
        };
        computeNormals(mesh);
        meshes.push(mesh);
    }

    // Create Scene object
    const scene: Scene = {
        camera: {} as any, // Mock
        meshes,
        materials: materials as any, // Only need albedo really
        lightSources: []
    };

    // Create Source Light
    let sourceLight: LightSource | null = null;
    if (params.defaultLightPos) {
        const posParts = params.defaultLightPos.split(',').map(Number);
        const colorParts = params.defaultLightColor ? params.defaultLightColor.split(',').map(Number) : [1, 1, 1];
        const intensity = params.defaultLightIntensity ?? 500000.0;

        if (posParts.length === 3) {
            sourceLight = {
                position: [posParts[0]!, posParts[1]!, posParts[2]!],
                intensity,
                color: [colorParts[0]!, colorParts[1]!, colorParts[2]!],
                spot: [0, 0, 0],
                angle: -2.0,
                useRaytracedShadows: true,
                fixedIntensity: true
            };
        }
    }

    if (!sourceLight) {
        console.error('[VirtualLights] No default light found in params.txt');
        process.exit(1);
    }

    // Generate VPLs
    const vpls = generateVirtualLights(scene, sourceLight, div);
    if (vpls.length === 0) {
        console.warn('[VirtualLights] No virtual lights generated.');
        process.exit(0);
    }

    // Write output to vlights.txt
    // Format: x,y,z,intensity,r,g,b
    const outPath = path.join(sceneDir, 'vlights.txt');
    const lines: string[] = [];

    for (const l of vpls) {
        const [x, y, z] = l.position;
        const [r, g, b] = l.color;
        lines.push(`${x.toFixed(4)},${y.toFixed(4)},${z.toFixed(4)},${l.intensity.toFixed(4)},${r.toFixed(4)},${g.toFixed(4)},${b.toFixed(4)}`);
    }

    fs.writeFileSync(outPath, lines.join('\n'), 'utf-8');
    console.log(`[VirtualLights] Wrote ${vpls.length} virtual lights to ${outPath}`);
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
