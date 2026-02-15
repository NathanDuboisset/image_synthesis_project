import type { Vec3, AABB, LightSource, LightcutNode, LightcutRepresentative } from './types.ts';

// ─── Helpers ────────────────────────────────────────────────────────────────

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

/** Metric for pairing: distance between representative positions + bounding-box diagonal growth. */
function mergeCost(nodeA: LightcutNode, nodeB: LightcutNode): number {
    const posDistSq = vec3Dist2(nodeA.representative.position, nodeB.representative.position);
    // Also penalise resulting bounding-box volume to discourage pairing far-apart lights
    const merged = aabbUnion(nodeA.aabb, nodeB.aabb);
    const dx = merged.max[0] - merged.min[0];
    const dy = merged.max[1] - merged.min[1];
    const dz = merged.max[2] - merged.min[2];
    const volEstimate = dx * dy * dz;
    return posDistSq + volEstimate;
}

// ─── Node creation ──────────────────────────────────────────────────────────

function createLeafNode(light: LightSource, index: number): LightcutNode {
    return {
        aabb: aabbFromPoint(light.position),
        representative: {
            position: [...light.position],
            intensity: light.intensity,
            color: [...light.color],
        },
        totalIntensity: light.intensity,
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
    // Intensity-weighted average position and color
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

// ─── Depth assignment ───────────────────────────────────────────────────────

function assignDepths(node: LightcutNode | null, depth: number): void {
    if (!node) return;
    node.depth = depth;
    assignDepths(node.left, depth + 1);
    assignDepths(node.right, depth + 1);
}

/** Returns the maximum depth in the tree. */
function treeMaxDepth(node: LightcutNode | null): number {
    if (!node) return -1;
    if (!node.left && !node.right) return node.depth;
    return Math.max(treeMaxDepth(node.left), treeMaxDepth(node.right));
}

// ─── 1) Brute-force pairing ────────────────────────────────────────────────

export function buildLightcutTreeBruteForce(lightSources: LightSource[]): LightcutNode | null {
    if (!lightSources || lightSources.length === 0) return null;
    if (lightSources.length === 1) {
        const root = createLeafNode(lightSources[0]!, 0);
        root.depth = 0;
        return root;
    }

    // Create leaf nodes
    const nodes: LightcutNode[] = lightSources.map((l, i) => createLeafNode(l, i));

    while (nodes.length > 1) {
        // Find the pair with the smallest merge cost
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
        // Merge the best pair
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

// ─── 2) KD-tree ─────────────────────────────────────────────────────────────

interface LightItem {
    light: LightSource;
    index: number;
}

export function buildLightcutTreeKDTree(lightSources: LightSource[], method: 'spatial' | 'median' = 'spatial'): LightcutNode | null {
    if (!lightSources || lightSources.length === 0) return null;

    const items: LightItem[] = lightSources.map((l, i) => ({ light: l, index: i }));
    const useSpatial = method === 'spatial';

    function buildRecursive(subset: LightItem[]): LightcutNode | null {
        if (subset.length === 0) return null;
        if (subset.length === 1) {
            return createLeafNode(subset[0]!.light, subset[0]!.index);
        }

        // Compute bounding box
        let minP: Vec3 = [Infinity, Infinity, Infinity];
        let maxP: Vec3 = [-Infinity, -Infinity, -Infinity];
        for (const item of subset) {
            const p = item.light.position;
            minP = vec3Min(minP, p);
            maxP = vec3Max(maxP, p);
        }

        // Find longest axis
        const extents: Vec3 = [maxP[0] - minP[0], maxP[1] - minP[1], maxP[2] - minP[2]];
        let axis: 0 | 1 | 2 = 0;
        if (extents[1] > extents[axis]) axis = 1;
        if (extents[2] > extents[axis]) axis = 2;

        let left: LightcutNode | null;
        let right: LightcutNode | null;
        if (useSpatial) {
            // spatial partition : split midpoint
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
            left = buildRecursive(leftSlice);
            right = buildRecursive(rightSlice);
        }
        else {
            // Sort along the chosen axis and split at the median
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

/** Get tree max depth (0-indexed, so a single-node tree has depth 0). */
export function getTreeMaxDepth(root: LightcutNode | null): number {
    return treeMaxDepth(root);
}

/** Flatten all nodes in the tree into an array (level-order). */
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

// ─── GPU serialization ──────────────────────────────────────────────────────

/** Number of f32 values per node in the GPU flat buffer. */
const FLOATS_PER_GPU_NODE = 16;

/**
 * Flatten the lightcut tree into a GPU-ready Float32Array (level-order BFS).
 *
 * Each node occupies 16 floats (64 bytes), matching the WGSL struct layout:
 *
 *   0–2  representative.position (vec3<f32>)
 *   3    totalIntensity          (f32)
 *   4–6  representative.color    (vec3<f32>)
 *   7    lightCount              (f32)
 *   8–10 aabb.min                (vec3<f32>)
 *   11   leftChildIndex          (f32, −1 = leaf)
 *   12–14 aabb.max               (vec3<f32>)
 *   15   rightChildIndex         (f32, −1 = leaf)
 */
export function flattenTreeForGPU(root: LightcutNode | null): { data: Float32Array; nodeCount: number } {
    if (!root) return { data: new Float32Array(FLOATS_PER_GPU_NODE), nodeCount: 0 };

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
    const data = new Float32Array(nodeCount * FLOATS_PER_GPU_NODE);

    for (let i = 0; i < nodeCount; i++) {
        const n = ordered[i]!;
        const o = i * FLOATS_PER_GPU_NODE;

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
