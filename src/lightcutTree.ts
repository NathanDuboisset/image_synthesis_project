import type { Vec3, AABB, LightSource, LightcutNode, LightcutRepresentative } from './types.ts';
// @ts-ignore - no official types for this package
import PriorityQueue from 'js-priority-queue';

function vec3Min(a: Vec3, b: Vec3): Vec3 {
    return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2])];
}

function vec3Max(a: Vec3, b: Vec3): Vec3 {
    return [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2])];
}

function vec3Dist2(a: Vec3, b: Vec3): number {
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

function aabbUnion(a: AABB, b: AABB): AABB {
    return {
        min: vec3Min(a.min, b.min),
        max: vec3Max(a.max, b.max),
    };
}

function aabbFromPoint(p: Vec3): AABB {
    return { min: [p[0], p[1], p[2]], max: [p[0], p[1], p[2]] };
}

function mergeCost(nodeA: LightcutNode, nodeB: LightcutNode): number {
    const posDistSq = vec3Dist2(nodeA.representative.position, nodeB.representative.position);
    // Add volume term to keep clusters compact
    const merged = aabbUnion(nodeA.aabb, nodeB.aabb);
    const dx = merged.max[0] - merged.min[0];
    const dy = merged.max[1] - merged.min[1];
    const dz = merged.max[2] - merged.min[2];
    const volEstimate = dx * dy * dz;
    return posDistSq + volEstimate;
}

function createLeafNode(light: LightSource, index: number): LightcutNode {
    // If not visible in RT, zero out intensity so it has no flux in the tree
    const intensity = (light.visibleInRT === false) ? 0.0 : light.intensity;
    return {
        aabb: aabbFromPoint(light.position),
        representative: {
            position: [...light.position],
            intensity: intensity,
            color: [...light.color],
        },
        totalIntensity: intensity,
        left: null,
        right: null,
        depth: 0,
        lightCount: 1,
        lightIndex: index,
    };
}

function createInternalNode(left: LightcutNode, right: LightcutNode): LightcutNode {
    const aabb = aabbUnion(left.aabb, right.aabb);
    const totalInt = left.totalIntensity + right.totalIntensity;
    // Weighted average for position and color
    const wL = left.totalIntensity / (totalInt || 1);
    const wR = right.totalIntensity / (totalInt || 1);
    const representative: LightcutRepresentative = {
        position: [
            left.representative.position[0] * wL + right.representative.position[0] * wR,
            left.representative.position[1] * wL + right.representative.position[1] * wR,
            left.representative.position[2] * wL + right.representative.position[2] * wR,
        ],
        intensity: totalInt,
        color: [
            left.representative.color[0] * wL + right.representative.color[0] * wR,
            left.representative.color[1] * wL + right.representative.color[1] * wR,
            left.representative.color[2] * wL + right.representative.color[2] * wR,
        ],
    };
    return {
        aabb,
        representative,
        totalIntensity: totalInt,
        left,
        right,
        depth: 0,
        lightCount: left.lightCount + right.lightCount,
        lightIndex: -1,
    };
}

function assignDepths(node: LightcutNode | null, depth: number): void {
    if (!node) return;
    node.depth = depth;
    assignDepths(node.left, depth + 1);
    assignDepths(node.right, depth + 1);
}

export function getTreeMaxDepth(node: LightcutNode | null): number {
    if (!node) return -1;
    if (!node.left && !node.right) return node.depth;
    return Math.max(getTreeMaxDepth(node.left), getTreeMaxDepth(node.right));
}

export function buildLightcutTreeBruteForce(lightSources: LightSource[]): LightcutNode | null {
    if (!lightSources || lightSources.length === 0) return null;
    if (lightSources.length === 1) {
        const root = createLeafNode(lightSources[0]!, 0);
        root.depth = 0;
        return root;
    }

    const nodes: LightcutNode[] = lightSources.map((l, i) => createLeafNode(l, i));

    while (nodes.length > 1) {
        let bestI = 0, bestJ = 1;
        let bestCost = mergeCost(nodes[0]!, nodes[1]!);
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const cost = mergeCost(nodes[i]!, nodes[j]!);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestI = i;
                    bestJ = j;
                }
            }
        }
        const merged = createInternalNode(nodes[bestI]!, nodes[bestJ]!);
        // Remove the two nodes (remove higher index first to keep lower valid)
        nodes.splice(bestJ, 1);
        nodes.splice(bestI, 1);
        nodes.push(merged);
    }

    const root = nodes[0]!;
    assignDepths(root, 0);
    return root;
}

interface LightItem {
    light: LightSource;
    index: number;
}

export function buildLightcutTreeKDTree(lightSources: LightSource[], method: 'spatial' | 'median' = 'spatial'): LightcutNode | null {
    if (!lightSources || lightSources.length === 0) return null;

    const items: LightItem[] = lightSources
        .map((l, i) => ({ light: l, index: i }))
        .filter(item => item.light.visibleInRT !== false);

    if (items.length === 0) return null;
    const useSpatial = method === 'spatial';

    function buildRecursive(subset: LightItem[]): LightcutNode | null {
        if (subset.length === 0) return null;
        if (subset.length === 1) {
            return createLeafNode(subset[0]!.light, subset[0]!.index);
        }

        let minP: Vec3 = [Infinity, Infinity, Infinity];
        let maxP: Vec3 = [-Infinity, -Infinity, -Infinity];
        for (const item of subset) {
            const p = item.light.position;
            minP = vec3Min(minP, p);
            maxP = vec3Max(maxP, p);
        }

        const extents: Vec3 = [maxP[0] - minP[0], maxP[1] - minP[1], maxP[2] - minP[2]];
        let axis: 0 | 1 | 2 = 0;
        if (extents[1] > extents[axis]) axis = 1;
        if (extents[2] > extents[axis]) axis = 2;

        let left: LightcutNode | null;
        let right: LightcutNode | null;
        if (useSpatial) {
            const midpoint = (minP[axis] + maxP[axis]) / 2;
            const leftSlice: LightItem[] = [];
            const rightSlice: LightItem[] = [];
            for (const item of subset) {
                if (item.light.position[axis] < midpoint) {
                    leftSlice.push(item);
                } else {
                    rightSlice.push(item);
                }
            }

            if (leftSlice.length === 0 || rightSlice.length === 0) {
                // Fallback: split in half
                const half = Math.floor(subset.length / 2);
                leftSlice.length = 0;
                rightSlice.length = 0;
                for (let i = 0; i < subset.length; i++) {
                    if (i < half) leftSlice.push(subset[i]!);
                    else rightSlice.push(subset[i]!);
                }
            }

            left = buildRecursive(leftSlice);
            right = buildRecursive(rightSlice);
        }
        else {
            subset.sort((a, b) => a.light.position[axis] - b.light.position[axis]);
            const mid = Math.floor(subset.length / 2);
            left = buildRecursive(subset.slice(0, mid));
            right = buildRecursive(subset.slice(mid));
        }
        if (!left) return right;
        if (!right) return left;

        return createInternalNode(left, right);
    }

    const root = buildRecursive(items);
    if (root) assignDepths(root, 0);
    return root;
}

export function getNodesAtDepth(root: LightcutNode | null, targetDepth: number): LightcutNode[] {
    const result: LightcutNode[] = [];
    if (!root) return result;

    function walk(node: LightcutNode | null): void {
        if (!node) return;
        if (node.depth === targetDepth) {
            result.push(node);
            return;
        }
        if (!node.left && !node.right) {
            result.push(node);
            return;
        }
        walk(node.left);
        walk(node.right);
    }

    walk(root);
    return result;
}

export function flattenTree(root: LightcutNode | null): LightcutNode[] {
    if (!root) return [];
    const result: LightcutNode[] = [];
    const queue: LightcutNode[] = [root];
    while (queue.length > 0) {
        const node = queue.shift()!;
        result.push(node);
        if (node.left) queue.push(node.left);
        if (node.right) queue.push(node.right);
    }
    return result;
}

type HeapItem = { idx: number; err: number };

function makeMaxHeap(): InstanceType<typeof PriorityQueue> {
    return new PriorityQueue({ comparator: (a: HeapItem, b: HeapItem) => b.err - a.err });
}

// Error bound matching lightcutErrorBound in shader: intensity * diag / dist³
function cpuErrorBound(data: Float32Array, nodeIdx: number, px: number, py: number, pz: number): number {
    const o = nodeIdx * 16;
    const intensity = data[o + 3]!;
    const dx = data[o + 0]! - px, dy = data[o + 1]! - py, dz = data[o + 2]! - pz;
    const distSq = dx * dx + dy * dy + dz * dz + 0.001;
    const dist = Math.sqrt(distSq);
    const ddx = data[o + 12]! - data[o + 8]!;
    const ddy = data[o + 13]! - data[o + 9]!;
    const ddz = data[o + 14]! - data[o + 10]!;
    const diag = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
    return intensity * diag / (distSq * dist);
}

function cpuIsLeaf(data: Float32Array, nodeIdx: number): boolean {
    const o = nodeIdx * 16;
    return Math.round(data[o + 11]!) < 0 && Math.round(data[o + 15]!) < 0;
}

// Build a cut for a single world position using a max-heap (O(k log n) vs GPU O(k²)).
function buildCutForPoint(data: Float32Array, nodeCount: number, px: number, py: number, pz: number, maxCutSize: number): number[] {
    if (nodeCount === 0) return [];
    const heap = makeMaxHeap();
    const leaves: number[] = [];

    function addNode(idx: number): void {
        if (cpuIsLeaf(data, idx)) leaves.push(idx);
        else heap.queue({ idx, err: cpuErrorBound(data, idx, px, py, pz) } as HeapItem);
    }

    addNode(0);
    while (heap.length > 0 && heap.length + leaves.length < maxCutSize) {
        const top = heap.dequeue() as HeapItem;
        const o = top.idx * 16;
        const leftI = Math.round(data[o + 11]!);
        const rightI = Math.round(data[o + 15]!);
        if (leftI >= 0 && leftI < nodeCount) addNode(leftI);
        if (rightI >= 0 && rightI < nodeCount && heap.length + leaves.length < maxCutSize) addNode(rightI);
    }

    const result = [...leaves];
    while (heap.length > 0) result.push((heap.dequeue() as HeapItem).idx);
    return result.slice(0, maxCutSize);
}

// Must match LC_MAX_CUT in shaders.wgsl and TILE_CUT_STRIDE = LC_MAX_CUT + 1.
const LC_MAX_CUT_CPU = 32;
const TILE_CUT_STRIDE_CPU = LC_MAX_CUT_CPU + 1; // 33

// Build one cut per tile using camera ray directions as representative world positions.
// invViewMat: camera-to-world matrix (column-major Float32Array, 16 elements).
export function buildCutsForTiles(
    gpuData: Float32Array, nodeCount: number,
    canvasW: number, canvasH: number, tileSize: number,
    invViewMat: Float32Array, fov: number, aspect: number,
): Uint32Array {
    if (nodeCount === 0 || tileSize <= 0) return new Uint32Array(LC_MAX_CUT_CPU + 1);

    const numTilesX = Math.ceil(canvasW / tileSize);
    const numTilesY = Math.ceil(canvasH / tileSize);
    const out = new Uint32Array(numTilesX * numTilesY * TILE_CUT_STRIDE_CPU);

    // Camera position (column 3 of column-major invViewMat)
    const camX = invViewMat[12]!, camY = invViewMat[13]!, camZ = invViewMat[14]!;

    // Scene center from root AABB (node 0)
    const sceneCx = (gpuData[8]! + gpuData[12]!) / 2;
    const sceneCy = (gpuData[9]! + gpuData[13]!) / 2;
    const sceneCz = (gpuData[10]! + gpuData[14]!) / 2;
    const dx = sceneCx - camX, dy = sceneCy - camY, dz = sceneCz - camZ;
    const sceneDepth = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;

    const tanHalfFov = Math.tan(fov / 2);

    for (let ty = 0; ty < numTilesY; ty++) {
        // Screen-space Y of tile center, NDC Y (flipped: screen Y increases downward)
        const pixY = (ty + 0.5) * tileSize;
        const ndcY = 1.0 - (pixY / canvasH) * 2.0;
        const vrY = ndcY * tanHalfFov;

        for (let tx = 0; tx < numTilesX; tx++) {
            const pixX = (tx + 0.5) * tileSize;
            const ndcX = (pixX / canvasW) * 2.0 - 1.0;
            const vrX = ndcX * aspect * tanHalfFov;
            const vrZ = -1.0;

            // Transform view-space ray to world space using upper-left 3x3 of invViewMat (column-major)
            const wrX = invViewMat[0]! * vrX + invViewMat[4]! * vrY + invViewMat[8]!  * vrZ;
            const wrY = invViewMat[1]! * vrX + invViewMat[5]! * vrY + invViewMat[9]!  * vrZ;
            const wrZ = invViewMat[2]! * vrX + invViewMat[6]! * vrY + invViewMat[10]! * vrZ;
            const len = Math.sqrt(wrX * wrX + wrY * wrY + wrZ * wrZ) + 1e-10;

            const wx = camX + (wrX / len) * sceneDepth;
            const wy = camY + (wrY / len) * sceneDepth;
            const wz = camZ + (wrZ / len) * sceneDepth;

            const cut = buildCutForPoint(gpuData, nodeCount, wx, wy, wz, LC_MAX_CUT_CPU);
            const base = (ty * numTilesX + tx) * TILE_CUT_STRIDE_CPU;
            out[base] = cut.length;
            for (let i = 0; i < cut.length; i++) out[base + 1 + i] = cut[i]!;
        }
    }
    return out;
}

function spreadBits(v: number): number {
    let x = v & 0x3FF;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x <<  8)) & 0x0300F00F;
    x = (x | (x <<  4)) & 0x030C30C3;
    x = (x | (x <<  2)) & 0x09249249;
    return x;
}

function mortonCode3D(pos: Vec3, minX: number, minY: number, minZ: number,
                      rangeX: number, rangeY: number, rangeZ: number): number {
    const xi = Math.min(1023, Math.floor(((pos[0] - minX) / rangeX) * 1024)) & 0x3FF;
    const yi = Math.min(1023, Math.floor(((pos[1] - minY) / rangeY) * 1024)) & 0x3FF;
    const zi = Math.min(1023, Math.floor(((pos[2] - minZ) / rangeZ) * 1024)) & 0x3FF;
    return spreadBits(xi) | (spreadBits(yi) << 1) | (spreadBits(zi) << 2);
}

function nextPowerOf2(n: number): number {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
}

export function buildPerfectBinaryTreeForGPU(lightSources: LightSource[]): { data: Float32Array; nodeCount: number } {
    const lights = lightSources.filter(l => l.visibleInRT !== false && l.intensity > 0);
    if (lights.length === 0) return { data: new Float32Array(16), nodeCount: 0 };

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const l of lights) {
        const [x, y, z] = l.position;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const rangeX = maxX - minX + 1e-6;
    const rangeY = maxY - minY + 1e-6;
    const rangeZ = maxZ - minZ + 1e-6;

    const sorted = lights
        .map(l => ({ l, code: mortonCode3D(l.position, minX, minY, minZ, rangeX, rangeY, rangeZ) }))
        .sort((a, b) => a.code - b.code)
        .map(x => x.l);

    const N = sorted.length;
    const Np = nextPowerOf2(N);
    const nodeCount = 2 * Np - 1;
    const F = 16;
    const data = new Float32Array(nodeCount * F);

    // Fill leaf level: nodes [Np-1 .. 2*Np-2]
    for (let i = 0; i < Np; i++) {
        const o = (Np - 1 + i) * F;
        if (i < N) {
            const l = sorted[i]!;
            const [px, py, pz] = l.position;
            const [cx, cy, cz] = l.color;
            data[o+0]=px; data[o+1]=py; data[o+2]=pz; data[o+3]=l.intensity;
            data[o+4]=cx; data[o+5]=cy; data[o+6]=cz; data[o+7]=1;
            data[o+8]=px; data[o+9]=py; data[o+10]=pz; data[o+11]=-1;
            data[o+12]=px; data[o+13]=py; data[o+14]=pz; data[o+15]=-1;
        } else {
            // bogus: zero intensity, point AABB, no children
            data[o+11]=-1; data[o+15]=-1;
        }
    }

    // Build internal nodes bottom-up: i from Np-2 down to 0
    for (let i = Np - 2; i >= 0; i--) {
        const li = 2 * i + 1;
        const ri = 2 * i + 2;
        const lo = li * F, ro = ri * F, o = i * F;
        const lI = data[lo+3]!, rI = data[ro+3]!;
        const total = lI + rI;
        const wL = total > 0 ? lI / total : 0.5;
        const wR = 1 - wL;
        data[o+0] = data[lo+0]! * wL + data[ro+0]! * wR;
        data[o+1] = data[lo+1]! * wL + data[ro+1]! * wR;
        data[o+2] = data[lo+2]! * wL + data[ro+2]! * wR;
        data[o+3] = total;
        data[o+4] = data[lo+4]! * wL + data[ro+4]! * wR;
        data[o+5] = data[lo+5]! * wL + data[ro+5]! * wR;
        data[o+6] = data[lo+6]! * wL + data[ro+6]! * wR;
        data[o+7] = data[lo+7]! + data[ro+7]!;
        data[o+8]  = Math.min(data[lo+8]!,  data[ro+8]!);
        data[o+9]  = Math.min(data[lo+9]!,  data[ro+9]!);
        data[o+10] = Math.min(data[lo+10]!, data[ro+10]!);
        data[o+11] = li;
        data[o+12] = Math.max(data[lo+12]!, data[ro+12]!);
        data[o+13] = Math.max(data[lo+13]!, data[ro+13]!);
        data[o+14] = Math.max(data[lo+14]!, data[ro+14]!);
        data[o+15] = ri;
    }

    return { data, nodeCount };
}

export function flattenTreeForGPU(root: LightcutNode | null): { data: Float32Array; nodeCount: number } {
    if (!root) return { data: new Float32Array(16), nodeCount: 0 };

    // BFS to assign contiguous indices
    const ordered: LightcutNode[] = [];
    const indexMap = new Map<LightcutNode, number>();
    const queue: LightcutNode[] = [root];
    while (queue.length > 0) {
        const node = queue.shift()!;
        indexMap.set(node, ordered.length);
        ordered.push(node);
        if (node.left) queue.push(node.left);
        if (node.right) queue.push(node.right);
    }

    const nodeCount = ordered.length;
    const data = new Float32Array(nodeCount * 16);

    for (let i = 0; i < nodeCount; i++) {
        const n = ordered[i]!;
        const o = i * 16;

        // representative position + totalIntensity
        data[o + 0] = n.representative.position[0];
        data[o + 1] = n.representative.position[1];
        data[o + 2] = n.representative.position[2];
        data[o + 3] = n.totalIntensity;

        // representative color + lightCount
        data[o + 4] = n.representative.color[0];
        data[o + 5] = n.representative.color[1];
        data[o + 6] = n.representative.color[2];
        data[o + 7] = n.lightCount;

        // aabb.min + leftChild index
        data[o + 8] = n.aabb.min[0];
        data[o + 9] = n.aabb.min[1];
        data[o + 10] = n.aabb.min[2];
        data[o + 11] = n.left ? indexMap.get(n.left)! : -1;

        // aabb.max + rightChild index
        data[o + 12] = n.aabb.max[0];
        data[o + 13] = n.aabb.max[1];
        data[o + 14] = n.aabb.max[2];
        data[o + 15] = n.right ? indexMap.get(n.right)! : -1;
    }

    return { data, nodeCount };
}
