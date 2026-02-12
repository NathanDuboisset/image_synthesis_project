// lightcutTree.js — Lightcut tree construction (brute force & KD-tree approaches)
//
// A lightcut tree is a binary tree of lights. Each leaf is a single light;
// internal nodes represent clusters of lights. Every node stores:
//   - aabb: { min: [x,y,z], max: [x,y,z] } — the axis-aligned bounding box
//   - representative: the "average" or representative light for this cluster
//   - totalIntensity: summed intensity for the cluster
//   - left, right: child nodes (null for leaves)
//   - depth: depth in the tree (root = 0)
//   - lightCount: number of leaves in this subtree
//
// Two build strategies:
//   1) Brute-force: O(n²) greedy bottom-up — at each step merge the two closest nodes.
//   2) KD-tree: top-down recursive split along the longest bounding-box axis.

// ─── Helpers ────────────────────────────────────────────────────────────────

function vec3Min(a, b) {
    return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2])];
}

function vec3Max(a, b) {
    return [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2])];
}

function vec3Dist2(a, b) {
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

function aabbUnion(a, b) {
    return {
        min: vec3Min(a.min, b.min),
        max: vec3Max(a.max, b.max),
    };
}

function aabbFromPoint(p) {
    return { min: [p[0], p[1], p[2]], max: [p[0], p[1], p[2]] };
}

/** Metric for pairing: distance between representative positions + bounding-box diagonal growth. */
function mergeCost(nodeA, nodeB) {
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

function createLeafNode(light, index) {
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
        depth: 0,       // will be set when tree structure is finalized
        lightCount: 1,
        lightIndex: index,     // original light array index (only for leaves)
    };
}

function createInternalNode(left, right) {
    const aabb = aabbUnion(left.aabb, right.aabb);
    const totalInt = left.totalIntensity + right.totalIntensity;
    // Intensity-weighted average position and color
    const wL = left.totalIntensity / (totalInt || 1);
    const wR = right.totalIntensity / (totalInt || 1);
    const representative = {
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

function assignDepths(node, depth) {
    if (!node) return;
    node.depth = depth;
    assignDepths(node.left, depth + 1);
    assignDepths(node.right, depth + 1);
}

/** Returns the maximum depth in the tree. */
function treeMaxDepth(node) {
    if (!node) return -1;
    if (!node.left && !node.right) return node.depth;
    return Math.max(treeMaxDepth(node.left), treeMaxDepth(node.right));
}

// ─── 1) Brute-force pairing ────────────────────────────────────────────────
//
// Classic bottom-up agglomerative clustering. Start with N leaf nodes;
// at each step, find the pair with the smallest mergeCost and merge them.
// Repeat until a single root remains. Complexity: O(n²).

export function buildLightcutTreeBruteForce(lightSources) {
    if (!lightSources || lightSources.length === 0) return null;
    if (lightSources.length === 1) {
        const root = createLeafNode(lightSources[0], 0);
        root.depth = 0;
        return root;
    }

    // Create leaf nodes
    let nodes = lightSources.map((l, i) => createLeafNode(l, i));

    while (nodes.length > 1) {
        // Find the pair with the smallest merge cost
        let bestI = 0, bestJ = 1;
        let bestCost = mergeCost(nodes[0], nodes[1]);
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const cost = mergeCost(nodes[i], nodes[j]);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestI = i;
                    bestJ = j;
                }
            }
        }
        // Merge the best pair
        const merged = createInternalNode(nodes[bestI], nodes[bestJ]);
        // Remove the two nodes (remove higher index first to keep lower valid)
        nodes.splice(bestJ, 1);
        nodes.splice(bestI, 1);
        nodes.push(merged);
    }

    const root = nodes[0];
    assignDepths(root, 0);
    return root;
}

// ─── 2) KD-tree based pairing ──────────────────────────────────────────────
//
// Top-down approach: compute the bounding box of all lights in the set,
// split along the longest axis at the median, recurse on each half.
// Produces a balanced binary tree in O(n log n).

export function buildLightcutTreeKDTree(lightSources) {
    if (!lightSources || lightSources.length === 0) return null;

    // Build an array of { light, index } so we can track original indices
    const items = lightSources.map((l, i) => ({ light: l, index: i }));

    function buildRecursive(subset) {
        if (subset.length === 0) return null;
        if (subset.length === 1) {
            return createLeafNode(subset[0].light, subset[0].index);
        }

        // Compute bounding box
        let minP = [Infinity, Infinity, Infinity];
        let maxP = [-Infinity, -Infinity, -Infinity];
        for (const item of subset) {
            const p = item.light.position;
            minP = vec3Min(minP, p);
            maxP = vec3Max(maxP, p);
        }

        // Find longest axis
        const extents = [maxP[0] - minP[0], maxP[1] - minP[1], maxP[2] - minP[2]];
        let axis = 0;
        if (extents[1] > extents[axis]) axis = 1;
        if (extents[2] > extents[axis]) axis = 2;

        // Sort along the chosen axis and split at the median
        subset.sort((a, b) => a.light.position[axis] - b.light.position[axis]);
        const mid = Math.floor(subset.length / 2);

        const left = buildRecursive(subset.slice(0, mid));
        const right = buildRecursive(subset.slice(mid));

        if (!left) return right;
        if (!right) return left;

        return createInternalNode(left, right);
    }

    const root = buildRecursive(items);
    assignDepths(root, 0);
    return root;
}

// ─── Query utilities ────────────────────────────────────────────────────────

/** Collect all nodes at a specific depth ("rank"). */
export function getNodesAtDepth(root, targetDepth) {
    const result = [];
    if (!root) return result;

    function walk(node) {
        if (!node) return;
        if (node.depth === targetDepth) {
            result.push(node);
            return; // Don't go deeper — we show this node's bounding box
        }
        // If we haven't reached the target depth yet but this is a leaf,
        // include it anyway (the tree may be unbalanced)
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
export function getTreeMaxDepth(root) {
    return treeMaxDepth(root);
}

/** Flatten all nodes in the tree into an array (level-order). */
export function flattenTree(root) {
    if (!root) return [];
    const result = [];
    const queue = [root];
    while (queue.length > 0) {
        const node = queue.shift();
        result.push(node);
        if (node.left) queue.push(node.left);
        if (node.right) queue.push(node.right);
    }
    return result;
}
