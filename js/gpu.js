import { mat4Invert, mat4Transpose } from './math.js';
import { updateCamera } from './camera.js';

export function createGPUBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: (data.byteLength + 3) & ~3,
    usage,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export function createMeshBuffers(GPUApp, meshes) {
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
    const m = meshes[i];
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
    M[meshOffset + 3] = m.materialIndex;
  }
  return {
    positionBuffer: createGPUBuffer(GPUApp.device, P, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    normalBuffer: createGPUBuffer(GPUApp.device, N, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    indexBuffer: createGPUBuffer(GPUApp.device, I, GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    meshBuffer: createGPUBuffer(GPUApp.device, M, GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    indexFormat: 'uint32',
  };
}

export function fillMaterialStagingBuffer(GPUApp, materials) {
  const sizeOfMaterial = 8;
  for (let i = 0; i < materials.length; i++) {
    const m = materials[i];
    const offset = i * sizeOfMaterial;
    GPUApp.materialStagingBuffer.set(m.albedo, offset);
    GPUApp.materialStagingBuffer[offset + 3] = m.roughness;
    GPUApp.materialStagingBuffer[offset + 4] = m.metalness;
  }
}

export function createMaterialBuffer(GPUApp, materials) {
  const sizeOfMaterial = 8;
  GPUApp.materialStagingBuffer = new Float32Array(materials.length * sizeOfMaterial);
  fillMaterialStagingBuffer(GPUApp, materials);
  return createGPUBuffer(GPUApp.device, GPUApp.materialStagingBuffer, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
}

export function updateMaterialBuffer(GPUApp, materials) {
  fillMaterialStagingBuffer(GPUApp, materials);
  GPUApp.device.queue.writeBuffer(GPUApp.materialBuffer, 0, GPUApp.materialStagingBuffer);
}

export function fillLightSourceStagingBuffer(GPUApp, lightSources) {
  const sizeOfLightSource = 12;
  const useRT = document.getElementById('raytracingCheckbox').checked;
  // Total luminance budget for RT mode, divided equally among lights.
  const BASE_TOTAL_LUMINANCE = 2;
  const numLights = Math.max(1, lightSources.length);
  const perLightIntensity = useRT ? (BASE_TOTAL_LUMINANCE / numLights) : 1.0;
  for (let i = 0; i < lightSources.length; i++) {
    const l = lightSources[i];
    const offset = i * sizeOfLightSource;
    GPUApp.lightSourceStagingBuffer.set(l.position, offset);
    GPUApp.lightSourceStagingBuffer[offset + 3] = useRT ? perLightIntensity : l.intensity;
    GPUApp.lightSourceStagingBuffer.set(l.color, offset + 4);
    GPUApp.lightSourceStagingBuffer[offset + 7] = l.angle;
    GPUApp.lightSourceStagingBuffer.set(l.spot, offset + 8);
    GPUApp.lightSourceStagingBuffer[offset + 11] = useRT ? 1 : 0;
  }
}

export function createLightSourceBuffer(GPUApp, lightSources) {
  const sizeOfLightSource = 12;
  GPUApp.lightSourceStagingBuffer = new Float32Array(lightSources.length * sizeOfLightSource);
  fillLightSourceStagingBuffer(GPUApp, lightSources);
  return createGPUBuffer(GPUApp.device, GPUApp.lightSourceStagingBuffer, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
}

export function updateLightSourceBuffer(GPUApp, lightSources) {
  fillLightSourceStagingBuffer(GPUApp, lightSources);
  GPUApp.device.queue.writeBuffer(GPUApp.lightSourceBuffer, 0, GPUApp.lightSourceStagingBuffer);
}

export function initGPUBuffers(GPUApp, scene) {
  GPUApp.meshBuffers = createMeshBuffers(GPUApp, scene.meshes);
  GPUApp.materialBuffer = createMaterialBuffer(GPUApp, scene.materials);
  GPUApp.lightSourceBuffer = createLightSourceBuffer(GPUApp, scene.lightSources);
  GPUApp.uniformBuffer = createGPUBuffer(GPUApp.device, GPUApp.uniformData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  GPUApp.debugUniformBuffer = createGPUBuffer(GPUApp.device, GPUApp.debugUniformData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  GPUApp.bindGroup = GPUApp.device.createBindGroup({
    layout: GPUApp.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: GPUApp.uniformBuffer } },
      { binding: 1, resource: { buffer: GPUApp.meshBuffers.positionBuffer } },
      { binding: 2, resource: { buffer: GPUApp.meshBuffers.normalBuffer } },
      { binding: 3, resource: { buffer: GPUApp.meshBuffers.indexBuffer } },
      { binding: 4, resource: { buffer: GPUApp.meshBuffers.meshBuffer } },
      { binding: 5, resource: { buffer: GPUApp.materialBuffer } },
      { binding: 6, resource: { buffer: GPUApp.lightSourceBuffer } },
      { binding: 7, resource: { buffer: GPUApp.debugUniformBuffer } },
    ],
  });
}

export function initRenderPipeline(GPUApp, shaderCode) {
  console.log('[GPU] Creating shader module, code length =', shaderCode.length);
  GPUApp.shaderModule = GPUApp.device.createShaderModule({ label: 'Shaders', code: shaderCode });
  GPUApp.bindGroupLayout = GPUApp.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });
  GPUApp.rasterizationPipeline = GPUApp.device.createRenderPipeline({
    label: 'rasterizationPipeline',
    layout: GPUApp.device.createPipelineLayout({ bindGroupLayouts: [GPUApp.bindGroupLayout] }),
    vertex: { module: GPUApp.shaderModule, entryPoint: 'rasterVertexMain' },
    fragment: { module: GPUApp.shaderModule, entryPoint: 'rasterFragmentMain', targets: [{ format: GPUApp.canvasFormat }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });
  GPUApp.depthTexture = GPUApp.device.createTexture({
    size: [GPUApp.canvas.width, GPUApp.canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  GPUApp.offscreenColorTexture = GPUApp.device.createTexture({
    label: 'RT offscreen color',
    size: [GPUApp.canvas.width, GPUApp.canvas.height, 1],
    format: GPUApp.canvasFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  GPUApp.blitBindGroupLayout = GPUApp.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  GPUApp.blitPipeline = GPUApp.device.createRenderPipeline({
    label: 'blitPipeline',
    layout: GPUApp.device.createPipelineLayout({ bindGroupLayouts: [GPUApp.blitBindGroupLayout] }),
    vertex: { module: GPUApp.shaderModule, entryPoint: 'blitVertexMain' },
    fragment: { module: GPUApp.shaderModule, entryPoint: 'blitFragmentMain', targets: [{ format: GPUApp.canvasFormat }] },
    primitive: { topology: 'triangle-list' },
  });
  GPUApp.blitSampler = GPUApp.device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  GPUApp.blitBindGroup = GPUApp.device.createBindGroup({
    layout: GPUApp.blitBindGroupLayout,
    entries: [
      { binding: 0, resource: GPUApp.offscreenColorTexture.createView() },
      { binding: 1, resource: GPUApp.blitSampler },
    ],
  });
  GPUApp.rayTracingPipeline = GPUApp.device.createRenderPipeline({
    label: 'rayTracingPipeline',
    layout: GPUApp.device.createPipelineLayout({ bindGroupLayouts: [GPUApp.bindGroupLayout] }),
    vertex: { module: GPUApp.shaderModule, entryPoint: 'rayVertexMain' },
    fragment: { module: GPUApp.shaderModule, entryPoint: 'rayFragmentMain', targets: [{ format: GPUApp.canvasFormat }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });
  console.log('[GPU] Pipelines and depth texture created');
}

/**
 * Create accumulation-specific GPU resources (call once after initRenderPipeline).
 * - accumTexture: rgba16float accumulation target
 * - accumBlitPipeline: additive blit (blend: ONE + ONE) from offscreenColorTexture → accumTexture
 * - accumFinalPipeline: final blit from accumTexture → swap chain, dividing by passCount
 */
export function initAccumulationResources(GPUApp) {
  const w = GPUApp.canvas.width, h = GPUApp.canvas.height;

  // Accumulation texture (float for additive precision)
  GPUApp.accumTexture = GPUApp.device.createTexture({
    label: 'Accumulation color',
    size: [w, h, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });

  // --- Accumulation additive blit: offscreen → accum (blend ONE+ONE) ---
  const accumBlitBGL = GPUApp.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  GPUApp.accumBlitPipeline = GPUApp.device.createRenderPipeline({
    label: 'accumBlitPipeline',
    layout: GPUApp.device.createPipelineLayout({ bindGroupLayouts: [accumBlitBGL] }),
    vertex: { module: GPUApp.shaderModule, entryPoint: 'accumBlitVertexMain' },
    fragment: {
      module: GPUApp.shaderModule,
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
  const accumSampler = GPUApp.device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  GPUApp.accumBlitBindGroup = GPUApp.device.createBindGroup({
    layout: accumBlitBGL,
    entries: [
      { binding: 0, resource: GPUApp.offscreenColorTexture.createView() },
      { binding: 1, resource: accumSampler },
    ],
  });

  // --- Final blit: accum → swap chain (divide by passCount) ---
  const accumFinalBGL = GPUApp.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });
  GPUApp.accumFinalPipeline = GPUApp.device.createRenderPipeline({
    label: 'accumFinalPipeline',
    layout: GPUApp.device.createPipelineLayout({ bindGroupLayouts: [accumFinalBGL] }),
    vertex: { module: GPUApp.shaderModule, entryPoint: 'accumFinalVertexMain' },
    fragment: {
      module: GPUApp.shaderModule,
      entryPoint: 'accumFinalFragmentMain',
      targets: [{ format: GPUApp.canvasFormat }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // Uniform for invPassCount (16 bytes for alignment: 1 float + 3 padding)
  GPUApp.accumFinalUniformData = new Float32Array(8);
  GPUApp.accumFinalUniformBuffer = GPUApp.device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  GPUApp.accumFinalBindGroup = GPUApp.device.createBindGroup({
    layout: accumFinalBGL,
    entries: [
      { binding: 0, resource: GPUApp.accumTexture.createView() },
      { binding: 1, resource: accumSampler },
      { binding: 2, resource: { buffer: GPUApp.accumFinalUniformBuffer } },
    ],
  });

  console.log('[GPU] Accumulation resources created');
}

/**
 * Write invPassCount to the accumulation final uniform buffer.
 */
export function updateAccumFinalPassCount(GPUApp, passCount) {
  GPUApp.accumFinalUniformData[0] = 1.0 / Math.max(1, passCount);
  GPUApp.device.queue.writeBuffer(GPUApp.accumFinalUniformBuffer, 0, GPUApp.accumFinalUniformData);
}

export async function createGPUApp() {
  if (!navigator.gpu) {
    console.error('[GPU] navigator.gpu not available');
    throw new Error('WebGPU not supported on this browser.');
  }
  const canvas = document.querySelector('canvas');
  console.log('[GPU] Canvas size:', canvas.width, 'x', canvas.height);
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance'
  });
  if (!adapter) {
    console.error('[GPU] requestAdapter() returned null');
    throw new Error('No appropriate GPUAdapter found.');
  }
  console.log('[GPU] Adapter acquired:', adapter);
  const device = await adapter.requestDevice();
  console.log('[GPU] Device acquired');
  const context = canvas.getContext('webgpu');
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });
  console.log('[GPU] Context configured with format', canvasFormat);
  return {
    canvas,
    adapter,
    device,
    context,
    canvasFormat,
    meshBuffers: {},
    uniformBuffer: {},
    bindGroup: {},
    shaderModule: {},
    bindGroupLayout: {},
    rasterizationPipeline: {},
    rayTracingPipeline: {},
    depthTexture: {},
    offscreenColorTexture: null,
    uniformData: new Float32Array(96),
    debugUniformData: new Uint32Array(8),
  };
}

export function updateUniforms(GPUApp, scene) {
  updateCamera(scene.camera);
  GPUApp.uniformData.set(scene.camera.modelMat, 0);
  GPUApp.uniformData.set(scene.camera.viewMat, 16);
  const invViewMat = mat4Invert(scene.camera.viewMat);
  GPUApp.uniformData.set(invViewMat, 32);
  GPUApp.uniformData.set(mat4Transpose(invViewMat), 48);
  GPUApp.uniformData.set(scene.camera.projMat, 64);
  GPUApp.uniformData[80] = scene.camera.fov;
  GPUApp.uniformData[81] = scene.camera.aspect;
  const meshCount = typeof scene.baseMeshCount === 'number' ? scene.baseMeshCount : scene.meshes.length;
  GPUApp.uniformData[84] = meshCount;
  GPUApp.uniformData[85] = scene.lightSources.length;
  // Default: shade ALL lights (0..numLights). Accumulation overrides these per pass.
  GPUApp.uniformData[86] = 0;                          // lightStartIndex
  GPUApp.uniformData[87] = scene.lightSources.length;   // lightEndIndex
  GPUApp.uniformData[88] = GPUApp.canvas.width;          // screenWidth
  GPUApp.uniformData[89] = GPUApp.canvas.height;         // screenHeight
  GPUApp.device.queue.writeBuffer(GPUApp.uniformBuffer, 0, GPUApp.uniformData.buffer, GPUApp.uniformData.byteOffset, GPUApp.uniformData.byteLength);
}

/**
 * Override the light range in the uniform buffer (for accumulation passes).
 * Call after updateUniforms. Only writes the two changed floats.
 */
export function updateLightRange(GPUApp, startIndex, endIndex) {
  GPUApp.uniformData[86] = startIndex;
  GPUApp.uniformData[87] = endIndex;
  // Write just the relevant portion (offset 86*4 = 344 bytes, 2 floats = 8 bytes)
  GPUApp.device.queue.writeBuffer(GPUApp.uniformBuffer, 86 * 4, GPUApp.uniformData.buffer, GPUApp.uniformData.byteOffset + 86 * 4, 8);
}

export function updateDebugUniform(GPUApp) {
  const select = document.getElementById('debug_mode_select');
  let mode = 0;
  if (select) {
    const parsed = Number(select.value);
    if (!Number.isNaN(parsed)) {
      mode = parsed | 0;
    }
  }
  GPUApp.debugUniformData[0] = mode;
  GPUApp.device.queue.writeBuffer(GPUApp.debugUniformBuffer, 0, GPUApp.debugUniformData);
}
