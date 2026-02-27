import type { GPUApp, Scene, Material, LightSource, Mesh, MeshBuffers, RenderingType } from './types.ts';
import { mat4Invert, mat4Transpose } from './math.ts';
import { updateCamera } from './camera.ts';

export function createGPUBuffer(device: GPUDevice, data: ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
  const buffer = device.createBuffer({
    size: (data.byteLength + 3) & ~3,
    usage,
  });
  device.queue.writeBuffer(buffer, 0, data as unknown as BufferSource);
  return buffer;
}

export function createMeshBuffers(app: GPUApp, meshes: Mesh[]): MeshBuffers {
  let positionsTotalLength = 0, normalsTotalLength = 0, indicesTotalLength = 0;
  for (const m of meshes) {
    positionsTotalLength += m.positions.length;
    normalsTotalLength += m.normals.length;
    indicesTotalLength += m.indices.length;
  }
  const P = new Float32Array(positionsTotalLength);
  const N = new Float32Array(normalsTotalLength);
  const I = new Uint32Array(indicesTotalLength);
  const M = new Uint32Array(4 * meshes.length);
  let positionsOffset = 0, normalsOffset = 0, indicesOffset = 0;
  for (let i = 0; i < meshes.length; i++) {
    const m = meshes[i]!;
    const meshOffset = 4 * i;
    P.set(m.positions, positionsOffset);
    M[meshOffset] = positionsOffset / 3;
    positionsOffset += m.positions.length;
    N.set(m.normals, normalsOffset);
    normalsOffset += m.normals.length;
    I.set(m.indices, indicesOffset);
    M[meshOffset + 1] = indicesOffset / 3;
    M[meshOffset + 2] = m.indices.length / 3;
    indicesOffset += m.indices.length;
    M[meshOffset + 3] = m.materialIndex ?? 0;
  }
  return {
    positionBuffer: createGPUBuffer(app.device, P, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    normalBuffer: createGPUBuffer(app.device, N, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    indexBuffer: createGPUBuffer(app.device, I, GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    meshBuffer: createGPUBuffer(app.device, M, GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    indexFormat: 'uint32',
  };
}

export function fillMaterialStagingBuffer(app: GPUApp, materials: Material[]): void {
  const sizeOfMaterial = 8;
  for (let i = 0; i < materials.length; i++) {
    const m = materials[i]!;
    const offset = i * sizeOfMaterial;
    app.materialStagingBuffer.set(m.albedo, offset);
    app.materialStagingBuffer[offset + 3] = m.roughness;
    app.materialStagingBuffer[offset + 4] = m.metalness;
  }
}

export function createMaterialBuffer(app: GPUApp, materials: Material[]): GPUBuffer {
  const sizeOfMaterial = 8;
  app.materialStagingBuffer = new Float32Array(materials.length * sizeOfMaterial);
  fillMaterialStagingBuffer(app, materials);
  return createGPUBuffer(app.device, app.materialStagingBuffer, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
}

export function updateMaterialBuffer(app: GPUApp, materials: Material[]): void {
  fillMaterialStagingBuffer(app, materials);
  app.device.queue.writeBuffer(app.materialBuffer, 0, app.materialStagingBuffer as unknown as BufferSource);
}

export function fillLightSourceStagingBuffer(app: GPUApp, lightSources: LightSource[]): void {
  const sizeOfLightSource = 12;
  const renderingSelect = document.getElementById('rendering_type_select') as HTMLSelectElement | null;
  const renderingType = (renderingSelect ? renderingSelect.value : 'raytrace') as RenderingType;
  const useRT = renderingType === 'raytrace' || renderingType === 'lightcuts'
             || renderingType === 'stochastic_lightcuts' || renderingType === 'realtime_stochastic_lightcuts';

  for (let i = 0; i < lightSources.length; i++) {
    const l = lightSources[i]!;
    const offset = i * sizeOfLightSource;

    let intensity = l.intensity;
    if (useRT && l.visibleInRT === false) {
      intensity = 0.0;
    }

    app.lightSourceStagingBuffer.set(l.position, offset);
    app.lightSourceStagingBuffer[offset + 3] = intensity;
    app.lightSourceStagingBuffer.set(l.color, offset + 4);
    app.lightSourceStagingBuffer[offset + 7] = l.angle;
    app.lightSourceStagingBuffer.set(l.spot, offset + 8);
    app.lightSourceStagingBuffer[offset + 11] = l.useRaytracedShadows ? 1 : 0;
  }
}

export function createLightSourceBuffer(app: GPUApp, lightSources: LightSource[]): GPUBuffer {
  const sizeOfLightSource = 12;
  app.lightSourceStagingBuffer = new Float32Array(lightSources.length * sizeOfLightSource);
  fillLightSourceStagingBuffer(app, lightSources);
  return createGPUBuffer(app.device, app.lightSourceStagingBuffer, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
}

export function updateLightSourceBuffer(app: GPUApp, lightSources: LightSource[]): void {
  fillLightSourceStagingBuffer(app, lightSources);
  app.device.queue.writeBuffer(app.lightSourceBuffer, 0, app.lightSourceStagingBuffer as unknown as BufferSource);
}

export function initGPUBuffers(app: GPUApp, scene: Scene): void {
  app.meshBuffers = createMeshBuffers(app, scene.meshes);
  app.materialBuffer = createMaterialBuffer(app, scene.materials);
  app.lightSourceBuffer = createLightSourceBuffer(app, scene.lightSources);
  app.uniformBuffer = createGPUBuffer(app.device, app.uniformData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  app.debugUniformBuffer = createGPUBuffer(app.device, app.debugUniformData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

  // Placeholder buffers
  app.lightcutTreeBuffer = createGPUBuffer(app.device, new Float32Array(16), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  app.lightcutTreeNodeCount = 0;
  app.tileCutBuffer = createGPUBuffer(app.device, new Uint32Array(33), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

  app.bindGroup = rebuildBindGroup(app);
}

function rebuildBindGroup(app: GPUApp): GPUBindGroup {
  return app.device.createBindGroup({
    layout: app.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: app.uniformBuffer } },
      { binding: 1, resource: { buffer: app.meshBuffers.positionBuffer } },
      { binding: 2, resource: { buffer: app.meshBuffers.normalBuffer } },
      { binding: 3, resource: { buffer: app.meshBuffers.indexBuffer } },
      { binding: 4, resource: { buffer: app.meshBuffers.meshBuffer } },
      { binding: 5, resource: { buffer: app.materialBuffer } },
      { binding: 6, resource: { buffer: app.lightSourceBuffer } },
      { binding: 7, resource: { buffer: app.debugUniformBuffer } },
      { binding: 8, resource: { buffer: app.lightcutTreeBuffer } },
      { binding: 9, resource: { buffer: app.tileCutBuffer } },
    ],
  });
}

export function initRenderPipeline(app: GPUApp, shaderCode: string): void {
  console.log('shader module code length =', shaderCode.length);
  app.shaderModule = app.device.createShaderModule({ label: 'Shaders', code: shaderCode });
  app.bindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 9, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });
  app.rasterizationPipeline = app.device.createRenderPipeline({
    label: 'rasterizationPipeline',
    layout: app.device.createPipelineLayout({ bindGroupLayouts: [app.bindGroupLayout] }),
    vertex: { module: app.shaderModule, entryPoint: 'rasterVertexMain' },
    fragment: { module: app.shaderModule, entryPoint: 'rasterFragmentMain', targets: [{ format: app.canvasFormat }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });
  app.depthTexture = app.device.createTexture({
    size: [app.canvas.width, app.canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  app.offscreenColorTexture = app.device.createTexture({
    label: 'RT offscreen color',
    size: [app.canvas.width, app.canvas.height, 1],
    format: app.canvasFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  app.blitBindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  app.blitPipeline = app.device.createRenderPipeline({
    label: 'blitPipeline',
    layout: app.device.createPipelineLayout({ bindGroupLayouts: [app.blitBindGroupLayout] }),
    vertex: { module: app.shaderModule, entryPoint: 'blitVertexMain' },
    fragment: { module: app.shaderModule, entryPoint: 'blitFragmentMain', targets: [{ format: app.canvasFormat }] },
    primitive: { topology: 'triangle-list' },
  });
  app.blitSampler = app.device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  app.blitBindGroup = app.device.createBindGroup({
    layout: app.blitBindGroupLayout,
    entries: [
      { binding: 0, resource: app.offscreenColorTexture.createView() },
      { binding: 1, resource: app.blitSampler },
    ],
  });
  app.rayTracingPipeline = app.device.createRenderPipeline({
    label: 'rayTracingPipeline',
    layout: app.device.createPipelineLayout({ bindGroupLayouts: [app.bindGroupLayout] }),
    vertex: { module: app.shaderModule, entryPoint: 'rayVertexMain' },
    fragment: { module: app.shaderModule, entryPoint: 'rayFragmentMain', targets: [{ format: app.canvasFormat }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });
  console.log('pipelines created');
}

// Create accumulation textures and pipelines
export function initAccumulationResources(app: GPUApp): void {
  const w = app.canvas.width, h = app.canvas.height;

  // Accumulation texture (float for additive precision)
  app.accumTexture = app.device.createTexture({
    label: 'Accumulation color',
    size: [w, h, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  // Additive blending for accumulation pass
  const accumBlitBGL = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  app.accumBlitPipeline = app.device.createRenderPipeline({
    label: 'accumBlitPipeline',
    layout: app.device.createPipelineLayout({ bindGroupLayouts: [accumBlitBGL] }),
    vertex: { module: app.shaderModule, entryPoint: 'accumBlitVertexMain' },
    fragment: {
      module: app.shaderModule,
      entryPoint: 'accumBlitFragmentMain',
      targets: [{
        format: 'rgba16float',
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
  });
  const accumSampler = app.device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  app.accumBlitBindGroup = app.device.createBindGroup({
    layout: accumBlitBGL,
    entries: [
      { binding: 0, resource: app.offscreenColorTexture.createView() },
      { binding: 1, resource: accumSampler },
    ],
  });

  // Final pass: averaging accumulator
  const accumFinalBGL = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });
  app.accumFinalPipeline = app.device.createRenderPipeline({
    label: 'accumFinalPipeline',
    layout: app.device.createPipelineLayout({ bindGroupLayouts: [accumFinalBGL] }),
    vertex: { module: app.shaderModule, entryPoint: 'accumFinalVertexMain' },
    fragment: {
      module: app.shaderModule,
      entryPoint: 'accumFinalFragmentMain',
      targets: [{ format: app.canvasFormat }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // Uniform for 1/passCount
  app.accumFinalUniformData = new Float32Array(8);
  app.accumFinalUniformBuffer = app.device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  app.accumFinalBindGroup = app.device.createBindGroup({
    layout: accumFinalBGL,
    entries: [
      { binding: 0, resource: app.accumTexture.createView() },
      { binding: 1, resource: accumSampler },
      { binding: 2, resource: { buffer: app.accumFinalUniformBuffer } },
    ],
  });

  console.log('accum resources created');
}

export function updateAccumFinalPassCount(app: GPUApp, passCount: number): void {
  app.accumFinalUniformData[0] = 1.0 / Math.max(1, passCount);
  app.device.queue.writeBuffer(app.accumFinalUniformBuffer, 0, app.accumFinalUniformData as unknown as BufferSource);
}

export async function createGPUApp(): Promise<GPUApp> {
  if (!navigator.gpu) {
    console.error('WebGPU not available');
    throw new Error('WebGPU not supported on this browser.');
  }
  const canvas = document.querySelector('canvas') as HTMLCanvasElement;
  console.log('canvas size:', canvas.width, 'x', canvas.height);
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error('requestAdapter() returned null');
    throw new Error('No appropriate GPUAdapter found.');
  }
  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu')!;
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });
  console.log('canvas format:', canvasFormat);
  return {
    canvas,
    adapter,
    device,
    context,
    canvasFormat,
    uniformData: new Float32Array(96),
    debugUniformData: new Uint32Array(8),
  } as GPUApp;
}

export function updateUniforms(app: GPUApp, scene: Scene): void {
  updateCamera(scene.camera);
  app.uniformData.set(scene.camera.modelMat, 0);
  app.uniformData.set(scene.camera.viewMat, 16);
  const invViewMat = mat4Invert(scene.camera.viewMat);
  app.uniformData.set(invViewMat, 32);
  app.uniformData.set(mat4Transpose(invViewMat), 48);
  app.uniformData.set(scene.camera.projMat, 64);
  app.uniformData[80] = scene.camera.fov;
  app.uniformData[81] = scene.camera.aspect;
  const meshCount = typeof scene.baseMeshCount === 'number' ? scene.baseMeshCount : scene.meshes.length;
  app.uniformData[84] = meshCount;
  app.uniformData[85] = scene.lightSources.length;
  // Default: shade all lights
  app.uniformData[86] = 0;
  app.uniformData[87] = scene.lightSources.length;
  app.uniformData[88] = app.canvas.width;             // screenWidth
  app.uniformData[89] = app.canvas.height;            // screenHeight
  app.uniformData[90] = scene.time ?? 0;              // frameCount
  app.device.queue.writeBuffer(app.uniformBuffer, 0, app.uniformData.buffer, app.uniformData.byteOffset, app.uniformData.byteLength);
}

export function updateLightRange(app: GPUApp, startIndex: number, endIndex: number): void {
  app.uniformData[86] = startIndex;
  app.uniformData[87] = endIndex;
  // Update only the relevant part of the buffer
  app.device.queue.writeBuffer(app.uniformBuffer, 86 * 4, app.uniformData.buffer, app.uniformData.byteOffset + 86 * 4, 8);
}

// Update debug parameters
export function updateDebugUniform(app: GPUApp, lightcutNodeCount: number = 0, maxCutSize: number = 0, algorithm: number = 0, fullbright: boolean = false, tileSize: number = 1, numTilesX: number = 1): void {
  app.debugUniformData[0] = 0;
  app.debugUniformData[1] = lightcutNodeCount;
  app.debugUniformData[2] = maxCutSize;
  app.debugUniformData[3] = algorithm;
  app.debugUniformData[4] = fullbright ? 1 : 0;
  app.debugUniformData[5] = tileSize;
  app.debugUniformData[6] = numTilesX;
  app.debugUniformData[7] = 0; // padding
  app.device.queue.writeBuffer(app.debugUniformBuffer, 0, app.debugUniformData as unknown as BufferSource);
}

// Upload tree buffer
export function uploadLightcutTree(app: GPUApp, treeData: Float32Array, nodeCount: number): void {
  if (app.lightcutTreeBuffer) app.lightcutTreeBuffer.destroy();
  app.lightcutTreeBuffer = createGPUBuffer(app.device, treeData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  app.lightcutTreeNodeCount = nodeCount;
  app.bindGroup = rebuildBindGroup(app);
  console.log(`lightcut tree uploaded: ${nodeCount} nodes`);
}

export function uploadTileCuts(app: GPUApp, data: Uint32Array): void {
  if (app.tileCutBuffer) app.tileCutBuffer.destroy();
  app.tileCutBuffer = createGPUBuffer(app.device, data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  app.bindGroup = rebuildBindGroup(app);
  console.log(`tile cuts uploaded: ${data.byteLength} bytes`);
}
