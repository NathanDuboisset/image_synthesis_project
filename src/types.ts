/// <reference types="@webgpu/types" />

// Primitives

export type Vec3 = [number, number, number];
export type Mat4 = Float32Array;



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
    fixedIntensity?: boolean;
    visibleInRT?: boolean;
}



export interface Mesh {
    positions: Float32Array;
    normals: Float32Array;
    indices: Uint32Array | Uint16Array;
    materialIndex?: number;
}



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
    // Interaction
    lastX?: number;
    lastY?: number;
    dragging?: boolean;
    panning?: boolean;
    _lcDragging?: boolean;
    _lcPanning?: boolean;
}

export interface SceneParams {
    radiusScale?: number;
    radius?: number;
    yaw?: number;
    pitch?: number;
    targetX?: number;
    targetY?: number;
    targetZ?: number;
    tileSize?: number;
    // New params
    defaultLightPos?: string; // "x,y,z"
    defaultLightColor?: string; // "r,g,b"
    defaultLightIntensity?: number;
    do_virtual?: boolean;
    virtual_dir_div?: number;
    lightcutVizRadius?: number;
    [key: string]: number | string | boolean | undefined;
}



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
    params?: SceneParams | null;
    debugLightMeshStart?: number;
    time?: number;
}



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



export type RenderMethod = 'tiles' | 'oneshot' | 'accumulation';
export type RenderingType = 'raster' | 'raytrace' | 'lightcuts' | 'stochastic_lightcuts';



export interface ParsedOBJ {
    positions: number[];
    indices: number[];
    lightPositions: Vec3[];
}

export interface OBJSceneResult {
    meshes: Mesh[];
    lights: Vec3[];
}



export interface NamedMaterial extends Material {
    name: string;
}
