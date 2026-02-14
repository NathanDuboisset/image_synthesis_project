import type { AABB, Mesh, Material, LightcutNode } from './types.ts';

const DEFAULT_MARGIN = 0.02;

function createSolidBox(aabb: AABB, materialIndex: number, margin: number = DEFAULT_MARGIN): Mesh {
    const x0 = aabb.min[0] - margin;
    const y0 = aabb.min[1] - margin;
    const z0 = aabb.min[2] - margin;
    const x1 = aabb.max[0] + margin;
    const y1 = aabb.max[1] + margin;
    const z1 = aabb.max[2] + margin;

    const positions = new Float32Array([
        // Front face (z = z0)
        x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0,
        // Back face (z = z1)
        x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
        // Top face (y = y1)
        x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1,
        // Bottom face (y = y0)
        x0, y0, z0, x1, y0, z0, x1, y0, z1, x0, y0, z1,
        // Right face (x = x1)
        x1, y0, z0, x1, y1, z0, x1, y1, z1, x1, y0, z1,
        // Left face (x = x0)
        x0, y0, z0, x0, y1, z0, x0, y1, z1, x0, y0, z1,
    ]);

    const normals = new Float32Array([
        // Front (facing -Z)
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
        // Back (facing +Z)
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
        // Top (facing +Y)
        0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
        // Bottom (facing -Y)
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
        // Right (facing +X)
        1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        // Left (facing -X)
        -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
    ]);

    const indices = new Uint32Array([
        // Front face
        0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2,
        // Back face
        4, 6, 5, 4, 7, 6, 4, 5, 6, 4, 6, 7,
        // Top face
        8, 9, 10, 8, 10, 11, 8, 10, 9, 8, 11, 10,
        // Bottom face
        12, 14, 13, 12, 15, 14, 12, 13, 14, 12, 14, 15,
        // Right face
        16, 17, 18, 16, 18, 19, 16, 18, 17, 16, 19, 18,
        // Left face
        20, 22, 21, 20, 23, 22, 20, 21, 22, 20, 22, 23,
    ]);

    return {
        positions,
        normals,
        indices,
        materialIndex,
    };
}

export function createBBoxMeshes(nodes: LightcutNode[], baseMaterialIndex: number, margin: number = DEFAULT_MARGIN): Mesh[] {
    const meshes: Mesh[] = [];
    for (let i = 0; i < nodes.length; i++) {
        meshes.push(createSolidBox(nodes[i]!.aabb, baseMaterialIndex + i, margin));
    }
    return meshes;
}

export function createIntensityMaterials(nodes: LightcutNode[]): Material[] {
    if (nodes.length === 0) return [];

    let minI = Infinity, maxI = -Infinity;
    for (const n of nodes) {
        const intensity = n.totalIntensity || 0;
        if (intensity < minI) minI = intensity;
        if (intensity > maxI) maxI = intensity;
    }
    const range = maxI - minI;

    const materials: Material[] = [];
    for (const n of nodes) {
        const intensity = n.totalIntensity || 0;
        const t = range > 1e-8 ? (intensity - minI) / range : 0.5;

        let r: number, g: number, b: number;
        if (t < 0.5) {
            const s = t * 2;
            r = 0.9;
            g = 0.2 + 0.7 * s;
            b = 0.1;
        } else {
            const s = (t - 0.5) * 2;
            r = 0.9 - 0.7 * s;
            g = 0.9;
            b = 0.1;
        }

        materials.push({
            albedo: [r, g, b],
            roughness: 0.3,
            metalness: 0.0,
        });
    }

    return materials;
}
