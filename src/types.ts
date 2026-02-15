/// <reference types="@webgpu/types" />

// ─── Primitives ─────────────────────────────────────────────────────────────

/** 3-component vector stored as a tuple. */
export type Vec3 = [number, number, number];

/** 4×4 column-major matrix stored in a Float32Array(16). */
export type Mat4 = Float32Array;

// ─── Materials & Lights ─────────────────────────────────────────────────────

export interface Material {
    albedo: Vec3;
    roughness: number;
    metalness: number;
}

export interface LightSource {
    position: Vec3;
    intensity: number;
    color: Vec3;
    spot: Vec3;
    angle: number;
    useRaytracedShadows: boolean;
}

// ─── Geometry ───────────────────────────────────────────────────────────────

export interface Mesh {
    positions: Float32Array;
    normals: Float32Array;
    indices: Uint32Array | Uint16Array;
    materialIndex?: number;
}

// ─── Camera ─────────────────────────────────────────────────────────────────

export interface Camera {
    viewMat: Mat4;
    invViewMat: Mat4;
    transInvViewMat: Mat4;
    projMat: Mat4;
    modelMat: Mat4;
    fov: number;
    aspect: number;
    near: number;
    far: number;
    // Orbit parameters
    yaw: number;
    pitch: number;
    radius: number;
    target: Vec3;
    rotateSpeed: number;
    panSpeed: number;
    zoomSpeed: number;
    minRadius: number;
    maxRadius: number;
    // Mouse interaction state (set at runtime)
    lastX?: number;
    lastY?: number;
    dragging?: boolean;
    panning?: boolean;
    /** Lightcut-canvas dragging flag (set dynamically in main.ts). */
    _lcDragging?: boolean;
    /** Lightcut-canvas panning flag (set dynamically in main.ts). */
    _lcPanning?: boolean;
}

// ─── Camera Config (from camera.txt) ────────────────────────────────────────

export interface CameraConfig {
    radiusScale?: number;
    radius?: number;
    yaw?: number;
    pitch?: number;
    targetX?: number;
    targetY?: number;
    targetZ?: number;
    tileSize?: number;
    [key: string]: number | string | undefined;
}

// ─── Scene ──────────────────────────────────────────────────────────────────

export interface SceneBounds {
    minX: number; minY: number; minZ: number;
    maxX: number; maxY: number; maxZ: number;
}

export interface Scene {
    camera: Camera;
    meshes: Mesh[];
    materials: Material[];
    lightSources: LightSource[];
    baseMeshCount?: number;
    cameraConfig?: CameraConfig | null;
    debugLightMeshStart?: number;
    time?: number;
}

// ─── GPU ────────────────────────────────────────────────────────────────────

export interface MeshBuffers {
    positionBuffer: GPUBuffer;
    normalBuffer: GPUBuffer;
    indexBuffer: GPUBuffer;
    meshBuffer: GPUBuffer;
    indexFormat: GPUIndexFormat;
}

export interface GPUApp {
    canvas: HTMLCanvasElement;
    adapter: GPUAdapter;
    device: GPUDevice;
    context: GPUCanvasContext;
    canvasFormat: GPUTextureFormat;

    // Shader & layouts
    shaderModule: GPUShaderModule;
    bindGroupLayout: GPUBindGroupLayout;
    blitBindGroupLayout: GPUBindGroupLayout;

    // Pipelines
    rasterizationPipeline: GPURenderPipeline;
    rayTracingPipeline: GPURenderPipeline;
    blitPipeline: GPURenderPipeline;
    accumBlitPipeline: GPURenderPipeline;
    accumFinalPipeline: GPURenderPipeline;

    // Textures
    depthTexture: GPUTexture;
    offscreenColorTexture: GPUTexture;
    accumTexture: GPUTexture;

    // Buffers
    meshBuffers: MeshBuffers;
    uniformBuffer: GPUBuffer;
    debugUniformBuffer: GPUBuffer;
    materialBuffer: GPUBuffer;
    lightSourceBuffer: GPUBuffer;
    accumFinalUniformBuffer: GPUBuffer;

    // Staging data (CPU-side)
    uniformData: Float32Array;
    debugUniformData: Uint32Array;
    materialStagingBuffer: Float32Array;
    lightSourceStagingBuffer: Float32Array;
    accumFinalUniformData: Float32Array;

    // Bind groups
    bindGroup: GPUBindGroup;
    blitBindGroup: GPUBindGroup;
    blitSampler: GPUSampler;
    accumBlitBindGroup: GPUBindGroup;
    accumFinalBindGroup: GPUBindGroup;

    // Lightcut tree (GPU storage)
    lightcutTreeBuffer: GPUBuffer;
    lightcutTreeNodeCount: number;
}

// ─── Lightcut Tree ──────────────────────────────────────────────────────────

export interface AABB {
    min: Vec3;
    max: Vec3;
}

export interface LightcutRepresentative {
    position: Vec3;
    intensity: number;
    color: Vec3;
}

export interface LightcutNode {
    aabb: AABB;
    representative: LightcutRepresentative;
    totalIntensity: number;
    left: LightcutNode | null;
    right: LightcutNode | null;
    depth: number;
    lightCount: number;
    lightIndex: number;
}

// ─── Render method unions ───────────────────────────────────────────────────

export type RenderMethod = 'tiles' | 'oneshot' | 'accumulation';
export type RenderingType = 'raster' | 'raytrace' | 'lightcuts' | 'stochastic_lightcuts';

// ─── OBJ Loader ─────────────────────────────────────────────────────────────

export interface ParsedOBJ {
    positions: number[];
    indices: number[];
    lightPositions: Vec3[];
}

export interface OBJSceneResult {
    meshes: Mesh[];
    lights: Vec3[];
}

// ─── Material with name (internal to scene loader) ──────────────────────────

export interface NamedMaterial extends Material {
    name: string;
}
