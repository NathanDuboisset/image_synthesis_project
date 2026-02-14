import type { GPUApp, Scene, RenderMethod, RenderingType, LightcutNode, MeshBuffers, LightSource } from './types.ts';
import { createScene, setCameraRandomNorthHemisphere } from './scene.ts';
import {
  createGPUApp, initRenderPipeline, initGPUBuffers, updateUniforms,
  updateMaterialBuffer, updateLightSourceBuffer, updateDebugUniform,
  initAccumulationResources, updateLightRange, updateAccumFinalPassCount,
  createMeshBuffers, createMaterialBuffer, createLightSourceBuffer, createGPUBuffer,
} from './gpu.ts';
import { pan, updateCamera } from './camera.ts';
import { mat4Invert, mat4Transpose } from './math.ts';
import { buildLightcutTreeBruteForce, buildLightcutTreeKDTree, getNodesAtDepth, getTreeMaxDepth } from './lightcutTree.ts';
import { createBBoxMeshes, createIntensityMaterials } from './lightcutViz.ts';

// Tile ratio controlling ray-tracing tile size vs. light count:
// tileSize ≈ TILERATIO / numLights, clamped to [32, 256].
const TILERATIO = 128000;

// Number of lights rendered per pass in accumulation mode.
const LIGHTS_PER_PASS = 10;

/** Read the current render method from the UI selector. */
function getSelectedRenderMethod(): RenderMethod {
  const sel = document.getElementById('render_method_select') as HTMLSelectElement | null;
  return (sel ? sel.value : 'tiles') as RenderMethod;
}

/** Get the current rendering type from the dropdown. */
function getRenderingType(): RenderingType {
  const sel = document.getElementById('rendering_type_select') as HTMLSelectElement | null;
  return (sel ? sel.value : 'raytrace') as RenderingType;
}

/** Whether ray tracing is enabled (raytrace, lightcuts, or stochastic lightcuts mode). */
function isRayTracingEnabled(): boolean {
  const t = getRenderingType();
  return t === 'raytrace' || t === 'lightcuts' || t === 'stochastic_lightcuts';
}

/** Get the current canvas content as a data URL. */
function getCanvasDataURL(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/png');
}

interface FullLightsTrainingOptions {
  onImage?: (index: number, dataUrl: string, timeMs: number) => void;
  forceRayTracing?: boolean;
}

/**
 * Run full-lights training: generate numImages with camera on random north hemisphere.
 */
async function runFullLightsTraining(
  app: GPUApp,
  scene: Scene,
  numImages: number,
  options: FullLightsTrainingOptions = {},
): Promise<{ times: number[] }> {
  const { onImage = () => { }, forceRayTracing = true } = options;
  console.log('[FullLights] runFullLightsTraining started, numImages =', numImages);
  const renderingSelect = document.getElementById('rendering_type_select') as HTMLSelectElement | null;
  const wasType = getRenderingType();
  if (forceRayTracing && renderingSelect) {
    renderingSelect.value = 'raytrace';
  }

  const times: number[] = [];
  for (let i = 0; i < numImages; i++) {
    setCameraRandomNorthHemisphere(scene);
    const timeMs = await renderScene(app, scene);
    times.push(timeMs);
    const dataUrl = getCanvasDataURL(app.canvas);
    onImage(i, dataUrl, timeMs);
  }

  if (forceRayTracing && renderingSelect) {
    renderingSelect.value = wasType;
  }
  return { times };
}

function initEvents(app: GPUApp, scene: Scene, renderCallback: () => void): void {
  app.canvas.addEventListener('mousedown', (e: MouseEvent) => {
    scene.camera.lastX = e.clientX;
    scene.camera.lastY = e.clientY;
    if (e.button === 0) scene.camera.dragging = true;
    if (e.button === 1 || e.button === 2) scene.camera.panning = true;
  });
  window.addEventListener('mouseup', () => {
    scene.camera.dragging = false;
    scene.camera.panning = false;
  });
  app.canvas.addEventListener('mousemove', (e: MouseEvent) => {
    const dx = e.clientX - (scene.camera.lastX ?? 0);
    const dy = e.clientY - (scene.camera.lastY ?? 0);
    scene.camera.lastX = e.clientX;
    scene.camera.lastY = e.clientY;
    if (scene.camera.dragging) {
      scene.camera.yaw -= dx * scene.camera.rotateSpeed;
      scene.camera.pitch += dy * scene.camera.rotateSpeed;
      const maxPitch = Math.PI / 2 - 0.01;
      scene.camera.pitch = Math.max(-maxPitch, Math.min(maxPitch, scene.camera.pitch));
      renderCallback();
    }
    if (scene.camera.panning) {
      pan(scene.camera, dx, -dy);
      renderCallback();
    }
  });
  app.canvas.addEventListener('wheel', (e: WheelEvent) => {
    e.preventDefault();
    scene.camera.radius *= 1 + e.deltaY * scene.camera.zoomSpeed;
    scene.camera.radius = Math.max(scene.camera.minRadius, Math.min(scene.camera.maxRadius, scene.camera.radius));
    renderCallback();
  }, { passive: false });
  app.canvas.addEventListener('contextmenu', (e: Event) => e.preventDefault());
}

async function renderScene(app: GPUApp, scene: Scene): Promise<number> {
  const useRayTracing = isRayTracingEnabled();
  const method = getSelectedRenderMethod();

  if (!useRayTracing) {
    return renderSceneRaster(app, scene);
  }

  if (method === 'accumulation') {
    return renderSceneAccumulation(app, scene);
  } else if (method === 'oneshot') {
    return renderSceneOneShot(app, scene);
  } else {
    return renderSceneTiles(app, scene);
  }
}

/** Rasterization path. */
async function renderSceneRaster(app: GPUApp, scene: Scene): Promise<number> {
  const start = performance.now();
  updateUniforms(app, scene);
  updateDebugUniform(app);
  updateMaterialBuffer(app, scene.materials);
  updateLightSourceBuffer(app, scene.lightSources);
  const encoder = app.device.createCommandEncoder();
  const renderPass = encoder.beginRenderPass({
    label: 'Raster pass',
    colorAttachments: [{
      view: app.context.getCurrentTexture().createView(),
      loadOp: 'clear' as const,
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store' as const,
    }],
    depthStencilAttachment: {
      view: app.depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: 'clear' as const,
      depthStoreOp: 'store' as const,
    },
  });
  renderPass.setPipeline(app.rasterizationPipeline);
  renderPass.setBindGroup(0, app.bindGroup);
  for (let i = 0; i < scene.meshes.length; i++) {
    renderPass.draw(scene.meshes[i]!.indices.length, 1, 0, i);
  }
  renderPass.end();
  app.device.queue.submit([encoder.finish()]);
  await app.device.queue.onSubmittedWorkDone();
  const end = performance.now();
  const frameMs = end - start;
  const label = document.getElementById('render_time_label');
  if (label) label.textContent = `${frameMs.toFixed(3)} ms`;
  return frameMs;
}

/** One-shot RT: render the full screen in a single dispatch. */
async function renderSceneOneShot(app: GPUApp, scene: Scene): Promise<number> {
  const start = performance.now();
  console.log('[Render] Starting image (one-shot RT)', scene.lightSources?.length ?? 0, 'lights');
  updateUniforms(app, scene);
  updateDebugUniform(app);
  updateMaterialBuffer(app, scene.materials);
  updateLightSourceBuffer(app, scene.lightSources);

  const offscreenView = app.offscreenColorTexture.createView();
  const depthView = app.depthTexture.createView();

  const encoder = app.device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    label: 'One-shot RT',
    colorAttachments: [{
      view: offscreenView,
      loadOp: 'clear' as const,
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store' as const,
    }],
    depthStencilAttachment: {
      view: depthView,
      depthClearValue: 1.0,
      depthLoadOp: 'clear' as const,
      depthStoreOp: 'store' as const,
    },
  });
  pass.setPipeline(app.rayTracingPipeline);
  pass.setBindGroup(0, app.bindGroup);
  pass.draw(6);
  pass.end();
  app.device.queue.submit([encoder.finish()]);
  await app.device.queue.onSubmittedWorkDone();

  // Blit to swap chain
  const blitEncoder = app.device.createCommandEncoder();
  const blitPass = blitEncoder.beginRenderPass({
    label: 'Blit to canvas',
    colorAttachments: [{
      view: app.context.getCurrentTexture().createView(),
      loadOp: 'clear' as const,
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store' as const,
    }],
  });
  blitPass.setPipeline(app.blitPipeline);
  blitPass.setBindGroup(0, app.blitBindGroup);
  blitPass.draw(6);
  blitPass.end();
  app.device.queue.submit([blitEncoder.finish()]);
  await app.device.queue.onSubmittedWorkDone();

  const end = performance.now();
  const frameMs = end - start;
  console.log('[Render] One-shot frame in', frameMs.toFixed(3), 'ms');
  const label = document.getElementById('render_time_label');
  if (label) label.textContent = `${frameMs.toFixed(3)} ms`;
  return frameMs;
}

/** Tiled RT: split canvas into tiles, render each sequentially. */
async function renderSceneTiles(app: GPUApp, scene: Scene): Promise<number> {
  const start = performance.now();
  console.log('[Render] Starting image (tiled RT)', scene.lightSources?.length ?? 0, 'lights');
  updateUniforms(app, scene);
  updateDebugUniform(app);
  updateMaterialBuffer(app, scene.materials);
  updateLightSourceBuffer(app, scene.lightSources);

  const numLights = Math.max(1, scene.lightSources?.length ?? 1);
  const desiredTileSize = TILERATIO / numLights;
  const baseTileSize = Number.isFinite(desiredTileSize) && desiredTileSize > 0
    ? desiredTileSize
    : ((scene.cameraConfig && typeof scene.cameraConfig.tileSize === 'number') ? scene.cameraConfig.tileSize : 256);
  const tileSize = Math.max(32, Math.min(256, Math.round(baseTileSize)));
  const tilesX = Math.ceil(app.canvas.width / tileSize);
  const tilesY = Math.ceil(app.canvas.height / tileSize);
  const tileCount = tilesX * tilesY;

  const offscreenView = app.offscreenColorTexture.createView();
  const depthView = app.depthTexture.createView();

  // Clear offscreen
  const clearEncoder = app.device.createCommandEncoder();
  const clearPass = clearEncoder.beginRenderPass({
    label: 'Clear pass',
    colorAttachments: [{
      view: offscreenView,
      loadOp: 'clear' as const,
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store' as const,
    }],
    depthStencilAttachment: {
      view: depthView,
      depthClearValue: 1.0,
      depthLoadOp: 'clear' as const,
      depthStoreOp: 'store' as const,
    },
  });
  clearPass.end();
  app.device.queue.submit([clearEncoder.finish()]);
  await app.device.queue.onSubmittedWorkDone();

  const tileTimes: number[] = [];
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const tileStart = performance.now();
      const x = tx * tileSize;
      const y = ty * tileSize;
      const w = Math.min(tileSize, app.canvas.width - x);
      const h = Math.min(tileSize, app.canvas.height - y);

      const tileEncoder = app.device.createCommandEncoder();
      const tilePass = tileEncoder.beginRenderPass({
        label: `Tile ${tx},${ty}`,
        colorAttachments: [{
          view: offscreenView,
          loadOp: 'load' as const,
          storeOp: 'store' as const,
        }],
        depthStencilAttachment: {
          view: depthView,
          depthLoadOp: 'load' as const,
          depthStoreOp: 'store' as const,
        },
      });
      tilePass.setPipeline(app.rayTracingPipeline);
      tilePass.setBindGroup(0, app.bindGroup);
      tilePass.setViewport(x, y, w, h, 0.0, 1.0);
      tilePass.setScissorRect(x, y, w, h);
      tilePass.draw(6);
      tilePass.end();
      app.device.queue.submit([tileEncoder.finish()]);
      await app.device.queue.onSubmittedWorkDone();

      {
        const tileEnd = performance.now();
        tileTimes.push(tileEnd - tileStart);
      }
      if ((tx + ty * tilesX) % 10 === 0) {
        await new Promise<void>(resolve => setTimeout(resolve, 0));
      }
    }
  }

  // Blit offscreen to swap chain
  const blitEncoder = app.device.createCommandEncoder();
  const blitPass = blitEncoder.beginRenderPass({
    label: 'Blit to canvas',
    colorAttachments: [{
      view: app.context.getCurrentTexture().createView(),
      loadOp: 'clear' as const,
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store' as const,
    }],
  });
  blitPass.setPipeline(app.blitPipeline);
  blitPass.setBindGroup(0, app.blitBindGroup);
  blitPass.draw(6);
  blitPass.end();
  app.device.queue.submit([blitEncoder.finish()]);
  await app.device.queue.onSubmittedWorkDone();

  if (tileTimes.length > 0) {
    const totalTileMs = tileTimes.reduce((acc, t) => acc + t, 0);
    const meanTileMs = totalTileMs / tileCount;
    let variance = 0;
    for (let i = 0; i < tileCount; i++) {
      const d = (tileTimes[i] ?? 0) - meanTileMs;
      variance += d * d;
    }
    variance /= tileCount;
    console.log('[Render][Tiles] total time =', totalTileMs.toFixed(3), 'ms, tiles =', tileCount, ', mean =', meanTileMs.toFixed(3), 'ms, variance =', variance.toFixed(3), 'ms^2');
  }

  const end = performance.now();
  const frameMs = end - start;
  console.log('[Render] Tiled frame in', frameMs.toFixed(3), 'ms');
  const label = document.getElementById('render_time_label');
  if (label) label.textContent = `${frameMs.toFixed(3)} ms`;
  return frameMs;
}

/**
 * Accumulation RT: render multiple passes, each with K lights, accumulate additively,
 * then final-blit with 1/N division.
 */
async function renderSceneAccumulation(app: GPUApp, scene: Scene): Promise<number> {
  const start = performance.now();
  const totalLights = scene.lightSources?.length ?? 0;
  const numPasses = Math.max(1, Math.ceil(totalLights / LIGHTS_PER_PASS));
  console.log('[Render] Starting accumulation RT:', totalLights, 'lights,', numPasses, 'passes of', LIGHTS_PER_PASS);

  updateUniforms(app, scene);
  updateDebugUniform(app);
  updateMaterialBuffer(app, scene.materials);
  updateLightSourceBuffer(app, scene.lightSources);

  const offscreenView = app.offscreenColorTexture.createView();
  const depthView = app.depthTexture.createView();
  const accumView = app.accumTexture.createView();

  // Clear accumulation texture to black
  {
    const enc = app.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      label: 'Clear accum',
      colorAttachments: [{
        view: accumView,
        loadOp: 'clear' as const,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        storeOp: 'store' as const,
      }],
    });
    pass.end();
    app.device.queue.submit([enc.finish()]);
    await app.device.queue.onSubmittedWorkDone();
  }

  for (let p = 0; p < numPasses; p++) {
    const lightStart = p * LIGHTS_PER_PASS;
    const lightEnd = Math.min(lightStart + LIGHTS_PER_PASS, totalLights);

    updateLightRange(app, lightStart, lightEnd);

    // Render full screen to offscreen texture
    {
      const enc = app.device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        label: `Accum pass ${p}`,
        colorAttachments: [{
          view: offscreenView,
          loadOp: 'clear' as const,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: 'store' as const,
        }],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear' as const,
          depthStoreOp: 'store' as const,
        },
      });
      pass.setPipeline(app.rayTracingPipeline);
      pass.setBindGroup(0, app.bindGroup);
      pass.draw(6);
      pass.end();
      app.device.queue.submit([enc.finish()]);
      await app.device.queue.onSubmittedWorkDone();
    }

    // Additive blit: offscreen → accumulation texture
    {
      const enc = app.device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        label: `Accum blit ${p}`,
        colorAttachments: [{
          view: accumView,
          loadOp: 'load' as const,
          storeOp: 'store' as const,
        }],
      });
      pass.setPipeline(app.accumBlitPipeline);
      pass.setBindGroup(0, app.accumBlitBindGroup);
      pass.draw(6);
      pass.end();
      app.device.queue.submit([enc.finish()]);
      await app.device.queue.onSubmittedWorkDone();
    }

    if (p % 5 === 4) {
      await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
  }

  // Final blit: accumulation texture → swap chain, dividing by numPasses
  updateAccumFinalPassCount(app, numPasses);
  {
    const enc = app.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      label: 'Accum final blit',
      colorAttachments: [{
        view: app.context.getCurrentTexture().createView(),
        loadOp: 'clear' as const,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store' as const,
      }],
    });
    pass.setPipeline(app.accumFinalPipeline);
    pass.setBindGroup(0, app.accumFinalBindGroup);
    pass.draw(6);
    pass.end();
    app.device.queue.submit([enc.finish()]);
    await app.device.queue.onSubmittedWorkDone();
  }

  const end = performance.now();
  const frameMs = end - start;
  console.log('[Render] Accumulation frame in', frameMs.toFixed(3), 'ms (', numPasses, 'passes)');
  const label = document.getElementById('render_time_label');
  if (label) label.textContent = `${frameMs.toFixed(3)} ms`;
  return frameMs;
}

async function main(): Promise<void> {
  console.log('[Main] Starting application');
  try {
    const app = await createGPUApp();
    const camAspect = app.canvas.width / app.canvas.height;
    console.log('[Main] Canvas aspect ratio =', camAspect);
    const shaderResponse = await fetch('shaders.wgsl');
    console.log('[Main] shaders.wgsl HTTP status =', shaderResponse.status);
    const shaderCode = await shaderResponse.text();
    initRenderPipeline(app, shaderCode);
    initAccumulationResources(app);

    const sceneSelect = document.getElementById('scene_select') as HTMLSelectElement | null;
    const getSceneName = (): string => (sceneSelect && sceneSelect.value) || 'ram';
    let scene = await createScene(camAspect, getSceneName());
    console.log('[Main] Scene ready, meshes:', scene.meshes.length, 'lights:', scene.lightSources.length);
    let animationFrameId: number | null = null;
    let isRendering = false;

    async function renderLoop(): Promise<void> {
      if (isRendering) return;
      const useRayTracing = isRayTracingEnabled();

      if (!useRayTracing) {
        isRendering = true;
        await renderScene(app, scene);
        isRendering = false;
        animationFrameId = requestAnimationFrame(renderLoop);
      } else {
        animationFrameId = null;
      }
    }

    function triggerRender(): void {
      const useRayTracing = isRayTracingEnabled();
      if (!useRayTracing && !isRendering) {
        if (!animationFrameId) {
          renderLoop();
        }
      }
    }

    initEvents(app, scene, triggerRender);
    initGPUBuffers(app, scene);
    console.log('[Main] GPU buffers initialized');

    await renderScene(app, scene);

    const renderingTypeSelect = document.getElementById('rendering_type_select') as HTMLSelectElement | null;
    function onRenderingTypeChange(): void {
      const useRayTracing = isRayTracingEnabled();
      if (!useRayTracing && !animationFrameId) {
        renderLoop();
      } else if (useRayTracing && animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }
    }
    if (renderingTypeSelect) {
      renderingTypeSelect.addEventListener('change', onRenderingTypeChange);
    }
    if (!isRayTracingEnabled()) {
      renderLoop();
    }

    const renderButton = document.getElementById('render_button') as HTMLButtonElement | null;
    if (renderButton) {
      renderButton.addEventListener('click', async () => {
        if (!isRendering) {
          isRendering = true;
          await renderScene(app, scene);
          isRendering = false;
        }
      });
    }

    if (sceneSelect) {
      sceneSelect.addEventListener('change', async () => {
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
          animationFrameId = null;
        }
        const sceneName = getSceneName();
        try {
          scene = await createScene(camAspect, sceneName);
          initGPUBuffers(app, scene);
          await renderScene(app, scene);
          if (!isRayTracingEnabled()) renderLoop();
        } catch (e) {
          console.error('[Main] Scene load failed:', e);
        }
      });
    }

    // Tabs: switch panel and sidebar visibility
    document.querySelectorAll('.tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        const id = tab.getAttribute('data-tab');
        document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        document.querySelectorAll('.tab-panel').forEach((panel) => {
          const panelEl = panel as HTMLElement;
          const panelId = panelEl.id.replace('panel-', '');
          const isActive = panelId === id;
          panelEl.classList.toggle('active', isActive);
          panelEl.hidden = !isActive;
        });
        const sidebarEl = document.getElementById('playground_sidebar');
        if (sidebarEl) sidebarEl.style.display = (id === 'playground' || id === 'testing') ? 'flex' : 'none';

        // Stop the render loop when leaving Playground, restart when coming back
        if (id !== 'playground') {
          if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
          }
        } else if (!isRayTracingEnabled() && !animationFrameId) {
          renderLoop();
        }
      });
    });

    // ─── Lightcut Tree tab ─────────────────────────────────────────────────
    let lightcutTree: LightcutNode | null = null;

    /**
     * Lightweight GPU context for the lightcut visualization canvas.
     * Shares {device, shaderModule, bindGroupLayout, rasterizationPipeline} with the main GPUApp.
     */
    interface LightcutGPUCtx {
      canvas: HTMLCanvasElement;
      device: GPUDevice;
      context: GPUCanvasContext;
      canvasFormat: GPUTextureFormat;
      shaderModule: GPUShaderModule;
      bindGroupLayout: GPUBindGroupLayout;
      rasterizationPipeline: GPURenderPipeline;
      depthTexture: GPUTexture;
      uniformBuffer: GPUBuffer | null;
      debugUniformBuffer: GPUBuffer | null;
      uniformData: Float32Array;
      debugUniformData: Uint32Array;
      bindGroup: GPUBindGroup | null;
      meshBuffers?: MeshBuffers;
      materialBuffer?: GPUBuffer;
      lightSourceBuffer?: GPUBuffer;
    }

    let lightcutGPU: LightcutGPUCtx | null = null;

    async function initLightcutGPU(): Promise<LightcutGPUCtx | null> {
      if (lightcutGPU) return lightcutGPU;
      const lcCanvas = document.getElementById('lightcut_canvas') as HTMLCanvasElement | null;
      if (!lcCanvas) return null;
      const lcContext = lcCanvas.getContext('webgpu')!;
      const lcFormat = navigator.gpu.getPreferredCanvasFormat();
      lcContext.configure({ device: app.device, format: lcFormat, alphaMode: 'opaque' });
      lightcutGPU = {
        canvas: lcCanvas,
        device: app.device,
        context: lcContext,
        canvasFormat: lcFormat,
        shaderModule: app.shaderModule,
        bindGroupLayout: app.bindGroupLayout,
        rasterizationPipeline: app.rasterizationPipeline,
        depthTexture: app.device.createTexture({
          size: [lcCanvas.width, lcCanvas.height],
          format: 'depth24plus',
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        }),
        uniformBuffer: null,
        debugUniformBuffer: null,
        uniformData: new Float32Array(96),
        debugUniformData: new Uint32Array(8),
        bindGroup: null,
      };
      return lightcutGPU;
    }

    /** Build a scene variant with solid bounding boxes for the lightcut viz. */
    function buildLightcutScene(baseScene: Scene, nodes: LightcutNode[]): Scene {
      const vizScene: Scene = {
        camera: baseScene.camera,
        meshes: [...baseScene.meshes.slice(0, baseScene.baseMeshCount ?? baseScene.meshes.length)],
        materials: [...baseScene.materials],
        lightSources: [],
        baseMeshCount: baseScene.baseMeshCount ?? baseScene.meshes.length,
      };

      // Create fill lights from 6 directions
      interface Bounds { minX: number; minY: number; minZ: number; maxX: number; maxY: number; maxZ: number }
      const bounds: Bounds = { minX: Infinity, minY: Infinity, minZ: Infinity, maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity };
      for (const m of vizScene.meshes) {
        if (!m.positions) continue;
        for (let i = 0; i < m.positions.length; i += 3) {
          bounds.minX = Math.min(bounds.minX, m.positions[i]!);
          bounds.minY = Math.min(bounds.minY, m.positions[i + 1]!);
          bounds.minZ = Math.min(bounds.minZ, m.positions[i + 2]!);
          bounds.maxX = Math.max(bounds.maxX, m.positions[i]!);
          bounds.maxY = Math.max(bounds.maxY, m.positions[i + 1]!);
          bounds.maxZ = Math.max(bounds.maxZ, m.positions[i + 2]!);
        }
      }
      const cx = (bounds.minX + bounds.maxX) * 0.5;
      const cy = (bounds.minY + bounds.maxY) * 0.5;
      const cz = (bounds.minZ + bounds.maxZ) * 0.5;
      const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ) * 1.5;
      const center: [number, number, number] = [cx, cy, cz];
      const fillLight = (pos: [number, number, number]): LightSource => ({
        position: pos,
        intensity: 0.3,
        color: [1, 1, 1],
        spot: center,
        angle: 0.0,
        useRaytracedShadows: false,
      });
      vizScene.lightSources = [
        fillLight([cx, cy + span, cz]),
        fillLight([cx, cy - span, cz]),
        fillLight([cx + span, cy, cz]),
        fillLight([cx - span, cy, cz]),
        fillLight([cx, cy, cz + span]),
        fillLight([cx, cy, cz - span]),
      ];

      const baseMaterialIndex = vizScene.materials.length;
      const nodeMaterials = createIntensityMaterials(nodes);
      for (const mat of nodeMaterials) {
        vizScene.materials.push(mat);
      }

      const boxMeshes = createBBoxMeshes(nodes, baseMaterialIndex);
      for (const mesh of boxMeshes) {
        vizScene.meshes.push(mesh);
      }

      return vizScene;
    }

    /** Render the lightcut visualization on the dedicated canvas. */
    async function renderLightcutViz(depth: number): Promise<void> {
      if (!lightcutTree) {
        console.warn('[Lightcut] No tree built yet');
        return;
      }
      const lcGPU = await initLightcutGPU();
      if (!lcGPU) return;

      const maxDepth = getTreeMaxDepth(lightcutTree);
      const nodes = getNodesAtDepth(lightcutTree, depth);

      const nodeCountEl = document.getElementById('lightcut_node_count');
      if (nodeCountEl) nodeCountEl.textContent = String(nodes.length);

      const vizScene = buildLightcutScene(scene, nodes);

      // Build a temporary GPUApp-like object for buffer creation helpers
      const tempApp = { device: lcGPU.device, materialStagingBuffer: new Float32Array(0), lightSourceStagingBuffer: new Float32Array(0) } as GPUApp;
      lcGPU.meshBuffers = createMeshBuffers(tempApp, vizScene.meshes);
      lcGPU.materialBuffer = createMaterialBuffer(tempApp, vizScene.materials);
      lcGPU.lightSourceBuffer = createLightSourceBuffer(tempApp, vizScene.lightSources);
      lcGPU.uniformBuffer = createGPUBuffer(lcGPU.device, lcGPU.uniformData as ArrayBufferView, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      lcGPU.debugUniformBuffer = createGPUBuffer(lcGPU.device, lcGPU.debugUniformData as ArrayBufferView, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      lcGPU.bindGroup = lcGPU.device.createBindGroup({
        layout: lcGPU.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: lcGPU.uniformBuffer } },
          { binding: 1, resource: { buffer: lcGPU.meshBuffers.positionBuffer } },
          { binding: 2, resource: { buffer: lcGPU.meshBuffers.normalBuffer } },
          { binding: 3, resource: { buffer: lcGPU.meshBuffers.indexBuffer } },
          { binding: 4, resource: { buffer: lcGPU.meshBuffers.meshBuffer } },
          { binding: 5, resource: { buffer: lcGPU.materialBuffer } },
          { binding: 6, resource: { buffer: lcGPU.lightSourceBuffer } },
          { binding: 7, resource: { buffer: lcGPU.debugUniformBuffer } },
        ],
      });

      // Update uniforms with the visualization scene
      updateCamera(vizScene.camera);
      lcGPU.uniformData.set(vizScene.camera.modelMat, 0);
      lcGPU.uniformData.set(vizScene.camera.viewMat, 16);
      const invViewMat = mat4Invert(vizScene.camera.viewMat);
      lcGPU.uniformData.set(invViewMat, 32);
      lcGPU.uniformData.set(mat4Transpose(invViewMat), 48);
      lcGPU.uniformData.set(vizScene.camera.projMat, 64);
      lcGPU.uniformData[80] = vizScene.camera.fov;
      lcGPU.uniformData[81] = vizScene.camera.aspect;
      lcGPU.uniformData[84] = vizScene.meshes.length;
      lcGPU.uniformData[85] = vizScene.lightSources.length;
      lcGPU.uniformData[86] = 0;
      lcGPU.uniformData[87] = vizScene.lightSources.length;
      lcGPU.uniformData[88] = lcGPU.canvas.width;
      lcGPU.uniformData[89] = lcGPU.canvas.height;
      lcGPU.device.queue.writeBuffer(lcGPU.uniformBuffer, 0, lcGPU.uniformData as unknown as BufferSource);

      // Debug uniform: mode 2 = raw albedo
      lcGPU.debugUniformData[0] = 2;
      lcGPU.device.queue.writeBuffer(lcGPU.debugUniformBuffer, 0, lcGPU.debugUniformData as unknown as BufferSource);

      // Render pass
      const encoder = lcGPU.device.createCommandEncoder();
      const renderPass = encoder.beginRenderPass({
        label: 'Lightcut raster pass',
        colorAttachments: [{
          view: lcGPU.context.getCurrentTexture().createView(),
          loadOp: 'clear' as const,
          clearValue: { r: 0.06, g: 0.06, b: 0.07, a: 1 },
          storeOp: 'store' as const,
        }],
        depthStencilAttachment: {
          view: lcGPU.depthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: 'clear' as const,
          depthStoreOp: 'store' as const,
        },
      });
      renderPass.setPipeline(lcGPU.rasterizationPipeline);
      renderPass.setBindGroup(0, lcGPU.bindGroup);
      for (let i = 0; i < vizScene.meshes.length; i++) {
        renderPass.draw(vizScene.meshes[i]!.indices.length, 1, 0, i);
      }
      renderPass.end();
      lcGPU.device.queue.submit([encoder.finish()]);
      await lcGPU.device.queue.onSubmittedWorkDone();
    }

    // Build button
    const lightcutBuildBtn = document.getElementById('lightcut_build_btn') as HTMLButtonElement | null;
    const lightcutMethodSelect = document.getElementById('lightcut_method_select') as HTMLSelectElement | null;
    const lightcutDepthSlider = document.getElementById('lightcut_depth_slider') as HTMLInputElement | null;
    const lightcutDepthValue = document.getElementById('lightcut_depth_value');
    const lightcutMaxDepthEl = document.getElementById('lightcut_max_depth');
    const lightcutTotalLightsEl = document.getElementById('lightcut_total_lights');
    const lightcutBuildTimeEl = document.getElementById('lightcut_build_time');

    if (lightcutBuildBtn) {
      lightcutBuildBtn.addEventListener('click', async () => {
        if (!scene.lightSources || scene.lightSources.length === 0) {
          console.warn('[Lightcut] No lights in scene');
          return;
        }
        lightcutBuildBtn.disabled = true;
        lightcutBuildBtn.textContent = 'Building…';

        await new Promise<void>(r => setTimeout(r, 10));

        const method = lightcutMethodSelect ? lightcutMethodSelect.value : 'kdtree-spatial';
        const buildStart = performance.now();

        if (method === 'kdtree-spatial') {
          lightcutTree = buildLightcutTreeKDTree(scene.lightSources, 'spatial');
        } else if (method === 'kdtree-median') {
          lightcutTree = buildLightcutTreeKDTree(scene.lightSources, 'median');
        } else {
          lightcutTree = buildLightcutTreeBruteForce(scene.lightSources);
        }

        const buildTime = performance.now() - buildStart;
        const maxDepth = getTreeMaxDepth(lightcutTree);

        if (lightcutMaxDepthEl) lightcutMaxDepthEl.textContent = String(maxDepth);
        if (lightcutTotalLightsEl) lightcutTotalLightsEl.textContent = String(scene.lightSources.length);
        if (lightcutBuildTimeEl) lightcutBuildTimeEl.textContent = buildTime.toFixed(1) + ' ms';

        if (lightcutDepthSlider) {
          lightcutDepthSlider.max = String(maxDepth);
          lightcutDepthSlider.value = '0';
          lightcutDepthSlider.disabled = false;
        }
        if (lightcutDepthValue) lightcutDepthValue.textContent = '0';

        lightcutBuildBtn.disabled = false;
        lightcutBuildBtn.textContent = 'Build Tree';

        console.log(`[Lightcut] Built ${method} tree: maxDepth=${maxDepth}, lights=${scene.lightSources.length}, time=${buildTime.toFixed(1)}ms`);

        await renderLightcutViz(0);
      });
    }

    // Depth slider
    if (lightcutDepthSlider) {
      lightcutDepthSlider.addEventListener('input', async () => {
        const depth = parseInt(lightcutDepthSlider.value, 10);
        if (lightcutDepthValue) lightcutDepthValue.textContent = String(depth);
        await renderLightcutViz(depth);
      });
    }

    // Camera controls for lightcut canvas
    const lcCanvas = document.getElementById('lightcut_canvas') as HTMLCanvasElement | null;
    if (lcCanvas) {
      lcCanvas.addEventListener('mousedown', (e: MouseEvent) => {
        scene.camera.lastX = e.clientX;
        scene.camera.lastY = e.clientY;
        if (e.button === 0) scene.camera._lcDragging = true;
        if (e.button === 1 || e.button === 2) scene.camera._lcPanning = true;
      });
      window.addEventListener('mouseup', () => {
        scene.camera._lcDragging = false;
        scene.camera._lcPanning = false;
      });
      lcCanvas.addEventListener('mousemove', async (e: MouseEvent) => {
        const dx = e.clientX - (scene.camera.lastX ?? 0);
        const dy = e.clientY - (scene.camera.lastY ?? 0);
        scene.camera.lastX = e.clientX;
        scene.camera.lastY = e.clientY;
        if (scene.camera._lcDragging) {
          scene.camera.yaw -= dx * scene.camera.rotateSpeed;
          scene.camera.pitch += dy * scene.camera.rotateSpeed;
          const maxPitch = Math.PI / 2 - 0.01;
          scene.camera.pitch = Math.max(-maxPitch, Math.min(maxPitch, scene.camera.pitch));
          if (lightcutTree) {
            const depth = parseInt(lightcutDepthSlider?.value || '0', 10);
            await renderLightcutViz(depth);
          }
        }
        if (scene.camera._lcPanning) {
          pan(scene.camera, dx, -dy);
          if (lightcutTree) {
            const depth = parseInt(lightcutDepthSlider?.value || '0', 10);
            await renderLightcutViz(depth);
          }
        }
      });
      lcCanvas.addEventListener('wheel', async (e: WheelEvent) => {
        e.preventDefault();
        scene.camera.radius *= 1 + e.deltaY * scene.camera.zoomSpeed;
        scene.camera.radius = Math.max(scene.camera.minRadius, Math.min(scene.camera.maxRadius, scene.camera.radius));
        if (lightcutTree) {
          const depth = parseInt(lightcutDepthSlider?.value || '0', 10);
          await renderLightcutViz(depth);
        }
      }, { passive: false });
      lcCanvas.addEventListener('contextmenu', (e: Event) => e.preventDefault());
    }
    const playgroundSidebar = document.getElementById('playground_sidebar');
    if (playgroundSidebar) playgroundSidebar.style.display = 'flex';

    // Full-lights training
    const fullLightsRunBtn = document.getElementById('full_lights_run_btn') as HTMLButtonElement | null;
    const fullLightsNumImagesSlider = document.getElementById('full_lights_num_images') as HTMLInputElement | null;
    const fullLightsNumImagesValue = document.getElementById('full_lights_num_images_value');
    const fullLightsAvgMs = document.getElementById('full_lights_avg_ms');
    const fullLightsLastRun = document.getElementById('full_lights_last_run');
    const fullLightsGrid = document.getElementById('full_lights_grid');
    if (fullLightsRunBtn && fullLightsGrid) {
      if (fullLightsNumImagesSlider && fullLightsNumImagesValue) {
        fullLightsNumImagesSlider.addEventListener('input', () => {
          fullLightsNumImagesValue.textContent = fullLightsNumImagesSlider.value;
        });
        fullLightsNumImagesValue.textContent = fullLightsNumImagesSlider.value;
      }
      fullLightsRunBtn.addEventListener('click', async () => {
        // Stop any running render loop first
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
          animationFrameId = null;
        }
        // Wait for any in-flight render to finish
        if (isRendering) {
          console.log('[Testing] Waiting for current render to finish…');
          await new Promise<void>(resolve => {
            const check = () => { if (!isRendering) resolve(); else setTimeout(check, 50); };
            check();
          });
        }
        const numImages = Math.max(1, parseInt(fullLightsNumImagesSlider?.value || '10', 10));
        console.log('[Testing] Starting generation of', numImages, 'images');
        isRendering = true;
        fullLightsRunBtn.disabled = true;
        fullLightsGrid.innerHTML = '';
        if (fullLightsLastRun) fullLightsLastRun.textContent = 'Running…';

        try {
          const { times } = await runFullLightsTraining(app, scene, numImages, {
            onImage(index: number, dataUrl: string) {
              const img = document.createElement('img');
              img.src = dataUrl;
              img.alt = `Frame ${index + 1}`;
              img.className = 'training-thumb';
              fullLightsGrid.appendChild(img);
            },
          });

          const avg = times.length ? times.reduce((a, b) => a + b, 0) / times.length : 0;
          if (fullLightsAvgMs) fullLightsAvgMs.textContent = avg.toFixed(2);
          if (fullLightsLastRun) fullLightsLastRun.textContent = times.map((t) => t.toFixed(0) + ' ms').join(', ');
          console.log('[FullLights] Done. Average time:', avg.toFixed(2), 'ms');
        } catch (err) {
          console.error('[FullLights] Error:', err);
          if (fullLightsLastRun) fullLightsLastRun.textContent = 'Error: ' + ((err as Error).message || String(err));
        } finally {
          isRendering = false;
          fullLightsRunBtn.disabled = false;
        }
      });
    }
  } catch (err) {
    console.error('[Main] Fatal error during initialization:', err);
  }
}

main();
