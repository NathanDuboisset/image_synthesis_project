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

    // Load Mesh
    const objText = fs.readFileSync(objPath, 'utf-8');
    const { positions, indicesByMaterial } = parseOBJ(objText);

    const meshes: Mesh[] = [];
    for (const [matName, matIndices] of indicesByMaterial.entries()) {
        if (matIndices.length === 0) continue;
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
            materialIndex: 0 // Mock
        };
        computeNormals(mesh);
        meshes.push(mesh);
    }

    // Create Scene object
    const scene: Scene = {
        camera: {} as any, // Mock
        meshes,
        materials: [],
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

    // Write OBJ output
    const outPath = path.join(sceneDir, 'lights.obj'); // Using standard name consistently
    const lines: string[] = [];

    // We'll write them as small triangles with material 'RamLight'
    // To mimic existing loader logic: centroid of triangle is light position.
    // So for each VPL at P, we write a tiny triangle centered at P.
    // e.g. (P + dx, P + dy, P + dz) ...
    // Let's use a small offset like 0.01

    lines.push(`usemtl RamLight`);
    const offset = 0.01;
    let vCount = 1;

    for (const l of vpls) {
        const [x, y, z] = l.position;
        // Triangle: (x, y, z-offset), (x-offset, y, z+offset), (x+offset, y, z+offset)
        // Centroid: (x, y, z + offset/3) -> close enough

        // Let's make it simpler: (x-d, y, z), (x+d, y, z), (x, y+d, z) -> centroid (x, y+d/3, z)
        // Or equilateral triangle in XZ plane:
        // P1 = (x, y, z - d)
        // P2 = (x - d*0.866, y, z + d*0.5)
        // P3 = (x + d*0.866, y, z + d*0.5)
        // Centroid = (x, y, z) exactly.

        const d = 0.05;
        const h = d * 0.866; // sqrt(3)/2
        const r = d * 0.5;

        // v1: top
        lines.push(`v ${x} ${y} ${z - d}`);
        // v2: left
        lines.push(`v ${x - h} ${y} ${z + r}`);
        // v3: right
        lines.push(`v ${x + h} ${y} ${z + r}`);

        lines.push(`f ${vCount} ${vCount + 1} ${vCount + 2}`);
        vCount += 3;
    }

    fs.writeFileSync(outPath, lines.join('\n'), 'utf-8');
    console.log(`[VirtualLights] Wrote ${vpls.length} virtual lights to ${outPath}`);
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
