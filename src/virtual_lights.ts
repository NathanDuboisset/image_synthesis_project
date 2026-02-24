import type { LightSource, Scene, Mesh, Vec3 } from './types.ts';
import { vec3Sub, vec3Cross, vec3Normalize, vec3Add } from './math.ts';

function vec3Dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vec3Scale(a: Vec3, s: number): Vec3 {
    return [a[0] * s, a[1] * s, a[2] * s];
}

interface Hit {
    t: number;
    position: Vec3;
    normal: Vec3;
    materialIndex?: number;
}

interface AABB {
    min: Vec3;
    max: Vec3;
}

function computeMeshAABB(mesh: Mesh): AABB | null {
    if (!mesh.positions || mesh.positions.length < 3) return null;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    // mesh.positions is Float32Array
    for (let i = 0; i < mesh.positions.length; i += 3) {
        const x = mesh.positions[i]!;
        const y = mesh.positions[i + 1]!;
        const z = mesh.positions[i + 2]!;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    }
    return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

function intersectAABB(orig: Vec3, dir: Vec3, box: AABB): boolean {
    let tmin = (box.min[0] - orig[0]) / dir[0];
    let tmax = (box.max[0] - orig[0]) / dir[0];

    if (tmin > tmax) { const temp = tmin; tmin = tmax; tmax = temp; }

    let tymin = (box.min[1] - orig[1]) / dir[1];
    let tymax = (box.max[1] - orig[1]) / dir[1];

    if (tymin > tymax) { const temp = tymin; tymin = tymax; tymax = temp; }

    if ((tmin > tymax) || (tymin > tmax)) return false;

    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    let tzmin = (box.min[2] - orig[2]) / dir[2];
    let tzmax = (box.max[2] - orig[2]) / dir[2];

    if (tzmin > tzmax) { const temp = tzmin; tzmin = tzmax; tzmax = temp; }

    if ((tmin > tzmax) || (tzmin > tmax)) return false;

    return true;
}

function intersectTriangle(orig: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3): { t: number, u: number, v: number } | null {
    const EPSILON = 1e-6;
    const edge1 = vec3Sub(v1, v0);
    const edge2 = vec3Sub(v2, v0);
    const h = vec3Cross(dir, edge2);
    const a = vec3Dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) return null;

    const f = 1.0 / a;
    const s = vec3Sub(orig, v0);
    const u = f * vec3Dot(s, h);

    if (u < 0.0 || u > 1.0) return null;

    const q = vec3Cross(s, edge1);
    const v = f * vec3Dot(dir, q);

    if (v < 0.0 || u + v > 1.0) return null;

    const t = f * vec3Dot(edge2, q);

    if (t > EPSILON) return { t, u, v };

    return null;
}

export function generateVirtualLights(scene: Scene, sourceLight: LightSource, div: number): LightSource[] {
    console.log('generating virtual lights (cube), div =', div);
    const virtualLights: LightSource[] = [];
    const start = performance.now();

    // Precompute AABBs
    const meshAABBs = scene.meshes.map(m => computeMeshAABB(m));

    // Limit div if excessive
    const safeDiv = Math.min(div, 100);
    const totalRays = safeDiv * safeDiv * 6;
    const intensityPerLight = sourceLight.intensity / totalRays;
    const orig = sourceLight.position;

    for (let face = 0; face < 6; face++) {
        for (let i = 0; i < safeDiv; i++) {
            for (let j = 0; j < safeDiv; j++) {
                const u = ((i + 0.5) / safeDiv) * 2.0 - 1.0;
                const v = ((j + 0.5) / safeDiv) * 2.0 - 1.0;

                let x = 0, y = 0, z = 0;
                if (face === 0) { x = 1; y = u; z = v; }       // +X
                else if (face === 1) { x = -1; y = u; z = v; } // -X
                else if (face === 2) { x = u; y = 1; z = v; }  // +Y
                else if (face === 3) { x = u; y = -1; z = v; } // -Y
                else if (face === 4) { x = u; y = v; z = 1; }  // +Z
                else if (face === 5) { x = u; y = v; z = -1; } // -Z

                const dir = vec3Normalize([x, y, z]);

                let minT = Infinity;
                let bestHit: Hit | null = null;

                for (let m = 0; m < scene.meshes.length; m++) {
                    const mesh = scene.meshes[m]!;
                    const aabb = meshAABBs[m];
                    if (!aabb || !mesh.positions || !mesh.indices) continue;

                    // Simple box culling
                    if (!intersectAABB(orig, dir, aabb)) continue;

                    const numTris = mesh.indices.length / 3;
                    for (let k = 0; k < numTris; k++) {
                        const i0 = mesh.indices[3 * k]!;
                        const i1 = mesh.indices[3 * k + 1]!;
                        const i2 = mesh.indices[3 * k + 2]!;

                        const v0: Vec3 = [mesh.positions[3 * i0]!, mesh.positions[3 * i0 + 1]!, mesh.positions[3 * i0 + 2]!];
                        const v1: Vec3 = [mesh.positions[3 * i1]!, mesh.positions[3 * i1 + 1]!, mesh.positions[3 * i1 + 2]!];
                        const v2: Vec3 = [mesh.positions[3 * i2]!, mesh.positions[3 * i2 + 1]!, mesh.positions[3 * i2 + 2]!];

                        const triHit = intersectTriangle(orig, dir, v0, v1, v2);
                        if (triHit && triHit.t < minT) {
                            minT = triHit.t;

                            // Interpolate normal
                            const n0: Vec3 = [mesh.normals[3 * i0]!, mesh.normals[3 * i0 + 1]!, mesh.normals[3 * i0 + 2]!];
                            const n1: Vec3 = [mesh.normals[3 * i1]!, mesh.normals[3 * i1 + 1]!, mesh.normals[3 * i1 + 2]!];
                            const n2: Vec3 = [mesh.normals[3 * i2]!, mesh.normals[3 * i2 + 1]!, mesh.normals[3 * i2 + 2]!];

                            const w = 1.0 - triHit.u - triHit.v;
                            const nx = w * n0[0] + triHit.u * n1[0] + triHit.v * n2[0];
                            const ny = w * n0[1] + triHit.u * n1[1] + triHit.v * n2[1];
                            const nz = w * n0[2] + triHit.u * n1[2] + triHit.v * n2[2];
                            const normal = vec3Normalize([nx, ny, nz]);

                            bestHit = {
                                t: minT,
                                position: vec3Add(orig, vec3Scale(dir, minT)),
                                normal: normal,
                                materialIndex: mesh.materialIndex
                            };
                        }
                    }
                }

                if (bestHit) {
                    const pos = vec3Add(bestHit.position, vec3Scale(bestHit.normal, 0.1));

                    // Calculate new color
                    let color = sourceLight.color;
                    if (bestHit.materialIndex !== undefined && scene.materials && scene.materials[bestHit.materialIndex]) {
                        const mat = scene.materials[bestHit.materialIndex];
                        if (mat && mat.albedo) {
                            color = [
                                color[0] * mat.albedo[0],
                                color[1] * mat.albedo[1],
                                color[2] * mat.albedo[2]
                            ];
                        }
                    }

                    // Attenuate intensity by distance squared
                    const dist = bestHit.t;
                    // Avoid division by zero or extremely high values for very close hits
                    const distSq = Math.max(0.1, dist * dist);
                    const attenuatedIntensity = intensityPerLight / distSq;

                    virtualLights.push({
                        position: pos,
                        intensity: attenuatedIntensity,
                        color: color,
                        spot: [0, 0, 0],
                        angle: -2.0,
                        useRaytracedShadows: sourceLight.useRaytracedShadows,
                        fixedIntensity: true
                    });
                }
            }
        }
        console.log(`face ${face + 1}/6 done, lights so far: ${virtualLights.length}`);
    }

    const end = performance.now();
    console.log(`generated ${virtualLights.length} virtual lights in ${(end - start).toFixed(1)}ms`);
    return virtualLights;
}
