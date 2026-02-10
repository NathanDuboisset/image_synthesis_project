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
  const useRaytracedShadows = document.getElementById('raytracedShadowsCheckbox').checked;
  for (let i = 0; i < lightSources.length; i++) {
    const l = lightSources[i];
    const offset = i * sizeOfLightSource;
    GPUApp.lightSourceStagingBuffer.set(l.position, offset);
    GPUApp.lightSourceStagingBuffer[offset + 3] = l.intensity;
    GPUApp.lightSourceStagingBuffer.set(l.color, offset + 4);
    GPUApp.lightSourceStagingBuffer[offset + 7] = l.angle;
    GPUApp.lightSourceStagingBuffer.set(l.spot, offset + 8);
    GPUApp.lightSourceStagingBuffer[offset + 11] = useRaytracedShadows ? 1 : 0;
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

export async function createGPUApp() {
  if (!navigator.gpu) {
    console.error('[GPU] navigator.gpu not available');
    throw new Error('WebGPU not supported on this browser.');
  }
  const canvas = document.querySelector('canvas');
  console.log('[GPU] Canvas size:', canvas.width, 'x', canvas.height);
  const adapter = await navigator.gpu.requestAdapter();
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
    uniformData: new Float32Array(88),
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
  GPUApp.uniformData[84] = scene.meshes.length;
  GPUApp.uniformData[85] = scene.lightSources.length;
  GPUApp.device.queue.writeBuffer(GPUApp.uniformBuffer, 0, GPUApp.uniformData.buffer, GPUApp.uniformData.byteOffset, GPUApp.uniformData.byteLength);
}
