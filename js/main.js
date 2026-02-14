import { createScene, setCameraTopDown, setCameraRandomNorthHemisphere } from './scene.js';
import { createGPUApp, initRenderPipeline, initGPUBuffers, updateUniforms, updateMaterialBuffer, updateLightSourceBuffer, updateDebugUniform, initAccumulationResources, updateLightRange, updateAccumFinalPassCount } from './gpu.js';
import { pan } from './camera.js';
import { buildLightcutTreeBruteForce, buildLightcutTreeKDTree, getNodesAtDepth, getTreeMaxDepth } from './lightcutTree.js';
import { createBBoxMeshes, createIntensityMaterials } from './lightcutViz.js';

// Tile ratio controlling ray-tracing tile size vs. light count:
// tileSize ≈ TILERATIO / numLights, clamped to [32, 256].
// Example: with TILERATIO = 12800 → 100 lights → 128px tiles, 400 lights → 32px tiles.
const TILERATIO = 128000;

// Number of lights rendered per pass in accumulation mode.
const LIGHTS_PER_PASS = 10;

/** Read the current render method from the UI selector (context-aware: Playground vs Training). */
function getSelectedRenderMethod() {
  const fullLightsPanel = document.getElementById('panel-full-lights');
  if (fullLightsPanel && !fullLightsPanel.hidden && fullLightsPanel.classList.contains('active')) {
    const selTrain = document.getElementById('render_method_select_training');
    return selTrain ? selTrain.value : 'tiles';
  }
  const sel = document.getElementById('render_method_select');
  return sel ? sel.value : 'tiles'; // fallback to tiles
}

/** Get the current rendering type from the dropdown. */
function getRenderingType() {
  const sel = document.getElementById('rendering_type_select');
  return sel ? sel.value : 'raytrace';
}

/** Whether ray tracing is enabled (raytrace or lightcuts mode). */
function isRayTracingEnabled() {
  const t = getRenderingType();
  return t === 'raytrace' || t === 'lightcuts';
}

/** Get the current canvas content as a data URL (reusable for any canvas). */
function getCanvasDataURL(canvas) {
  return canvas.toDataURL('image/png');
}

/**
 * Run full-lights training: generate numImages with camera on random north hemisphere.
 * Reuses renderScene; calls onImage(index, dataUrl, timeMs) for each image as it completes.
 * Optionally forces ray tracing on for the run and restores after.
 */
async function runFullLightsTraining(GPUApp, scene, numImages, options = {}) {
  const { onImage = () => { }, forceRayTracing = true } = options;
  console.log('[FullLights] runFullLightsTraining started, numImages =', numImages);
  const renderingSelect = document.getElementById('rendering_type_select');
  const wasType = getRenderingType();
  if (forceRayTracing && renderingSelect) {
    renderingSelect.value = 'raytrace';
  }

  const times = [];
  for (let i = 0; i < numImages; i++) {
    setCameraRandomNorthHemisphere(scene);
    const timeMs = await renderScene(GPUApp, scene);
    times.push(timeMs);
    const dataUrl = getCanvasDataURL(GPUApp.canvas);
    onImage(i, dataUrl, timeMs);
  }

  if (forceRayTracing && renderingSelect) {
    renderingSelect.value = wasType;
  }
  return { times };
}

function initEvents(GPUApp, scene, renderCallback) {
  GPUApp.canvas.addEventListener('mousedown', (e) => {
    scene.camera.lastX = e.clientX;
    scene.camera.lastY = e.clientY;
    if (e.button === 0) scene.camera.dragging = true;
    if (e.button === 1 || e.button === 2) scene.camera.panning = true;
  });
  window.addEventListener('mouseup', () => {
    scene.camera.dragging = false;
    scene.camera.panning = false;
  });
  GPUApp.canvas.addEventListener('mousemove', (e) => {
    const dx = e.clientX - scene.camera.lastX;
    const dy = e.clientY - scene.camera.lastY;
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
  GPUApp.canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    scene.camera.radius *= 1 + e.deltaY * scene.camera.zoomSpeed;
    scene.camera.radius = Math.max(scene.camera.minRadius, Math.min(scene.camera.maxRadius, scene.camera.radius));
    renderCallback();
  }, { passive: false });
  GPUApp.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
}

async function renderScene(GPUApp, scene) {
  const useRayTracing = isRayTracingEnabled();
  const method = getSelectedRenderMethod();

  // For non-ray-tracing, always use the standard raster path regardless of method.
  if (!useRayTracing) {
    return renderSceneRaster(GPUApp, scene);
  }

  // Ray tracing dispatch based on method.
  if (method === 'accumulation') {
    return renderSceneAccumulation(GPUApp, scene);
  } else if (method === 'oneshot') {
    return renderSceneOneShot(GPUApp, scene);
  } else {
    return renderSceneTiles(GPUApp, scene);
  }
}

/** Rasterization path (unchanged from original logic). */
async function renderSceneRaster(GPUApp, scene) {
  const start = performance.now();
  updateUniforms(GPUApp, scene);
  updateDebugUniform(GPUApp);
  updateMaterialBuffer(GPUApp, scene.materials);
  updateLightSourceBuffer(GPUApp, scene.lightSources);
  const encoder = GPUApp.device.createCommandEncoder();
  const renderPass = encoder.beginRenderPass({
    label: 'Raster pass',
    sampleCount: 1,
    colorAttachments: [{
      view: GPUApp.context.getCurrentTexture().createView(),
      loadOp: 'clear',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store',
    }],
    depthStencilAttachment: {
      view: GPUApp.depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  });
  renderPass.setPipeline(GPUApp.rasterizationPipeline);
  renderPass.setBindGroup(0, GPUApp.bindGroup);
  // Always draw all meshes including debug light markers
  for (let i = 0; i < scene.meshes.length; i++) {
    renderPass.draw(scene.meshes[i].indices.length, 1, 0, i);
  }
  renderPass.end();
  GPUApp.device.queue.submit([encoder.finish()]);
  await GPUApp.device.queue.onSubmittedWorkDone();
  const end = performance.now();
  const frameMs = end - start;
  const label = document.getElementById('render_time_label');
  if (label) label.textContent = `${frameMs.toFixed(3)} ms`;
  return frameMs;
}

/** One-shot RT: render the full screen in a single dispatch (all lights, no tiling). */
async function renderSceneOneShot(GPUApp, scene) {
  const start = performance.now();
  console.log('[Render] Starting image (one-shot RT)', scene.lightSources?.length ?? 0, 'lights');
  updateUniforms(GPUApp, scene);
  updateDebugUniform(GPUApp);
  updateMaterialBuffer(GPUApp, scene.materials);
  updateLightSourceBuffer(GPUApp, scene.lightSources);

  const offscreenView = GPUApp.offscreenColorTexture.createView();
  const depthView = GPUApp.depthTexture.createView();

  // Clear + render in one pass
  const encoder = GPUApp.device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    label: 'One-shot RT',
    sampleCount: 1,
    colorAttachments: [{
      view: offscreenView,
      loadOp: 'clear',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store',
    }],
    depthStencilAttachment: {
      view: depthView,
      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  });
  pass.setPipeline(GPUApp.rayTracingPipeline);
  pass.setBindGroup(0, GPUApp.bindGroup);
  pass.draw(6);
  pass.end();
  GPUApp.device.queue.submit([encoder.finish()]);
  await GPUApp.device.queue.onSubmittedWorkDone();

  // Blit to swap chain
  const blitEncoder = GPUApp.device.createCommandEncoder();
  const blitPass = blitEncoder.beginRenderPass({
    label: 'Blit to canvas',
    sampleCount: 1,
    colorAttachments: [{
      view: GPUApp.context.getCurrentTexture().createView(),
      loadOp: 'clear',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store',
    }],
  });
  blitPass.setPipeline(GPUApp.blitPipeline);
  blitPass.setBindGroup(0, GPUApp.blitBindGroup);
  blitPass.draw(6);
  blitPass.end();
  GPUApp.device.queue.submit([blitEncoder.finish()]);
  await GPUApp.device.queue.onSubmittedWorkDone();

  const end = performance.now();
  const frameMs = end - start;
  console.log('[Render] One-shot frame in', frameMs.toFixed(3), 'ms');
  const label = document.getElementById('render_time_label');
  if (label) label.textContent = `${frameMs.toFixed(3)} ms`;
  return frameMs;
}

/** Tiled RT: split canvas into tiles, render each sequentially (original approach). */
async function renderSceneTiles(GPUApp, scene) {
  const start = performance.now();
  console.log('[Render] Starting image (tiled RT)', scene.lightSources?.length ?? 0, 'lights');
  updateUniforms(GPUApp, scene);
  updateDebugUniform(GPUApp);
  updateMaterialBuffer(GPUApp, scene.materials);
  updateLightSourceBuffer(GPUApp, scene.lightSources);

  const numLights = Math.max(1, scene.lightSources?.length ?? 1);
  const desiredTileSize = TILERATIO / numLights;
  const baseTileSize = Number.isFinite(desiredTileSize) && desiredTileSize > 0
    ? desiredTileSize
    : ((scene.cameraConfig && typeof scene.cameraConfig.tileSize === 'number') ? scene.cameraConfig.tileSize : 256);
  const tileSize = Math.max(32, Math.min(256, Math.round(baseTileSize)));
  const tilesX = Math.ceil(GPUApp.canvas.width / tileSize);
  const tilesY = Math.ceil(GPUApp.canvas.height / tileSize);
  const tileCount = tilesX * tilesY;

  const offscreenView = GPUApp.offscreenColorTexture.createView();
  const depthView = GPUApp.depthTexture.createView();

  // Clear offscreen
  const clearEncoder = GPUApp.device.createCommandEncoder();
  const clearPass = clearEncoder.beginRenderPass({
    label: 'Clear pass',
    sampleCount: 1,
    colorAttachments: [{
      view: offscreenView,
      loadOp: 'clear',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store',
    }],
    depthStencilAttachment: {
      view: depthView,
      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  });
  clearPass.end();
  GPUApp.device.queue.submit([clearEncoder.finish()]);
  await GPUApp.device.queue.onSubmittedWorkDone();

  const tileTimes = [];
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const tileStart = performance.now();
      const x = tx * tileSize;
      const y = ty * tileSize;
      const w = Math.min(tileSize, GPUApp.canvas.width - x);
      const h = Math.min(tileSize, GPUApp.canvas.height - y);

      const tileEncoder = GPUApp.device.createCommandEncoder();
      const tilePass = tileEncoder.beginRenderPass({
        label: `Tile ${tx},${ty}`,
        sampleCount: 1,
        colorAttachments: [{
          view: offscreenView,
          loadOp: 'load',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: depthView,
          depthLoadOp: 'load',
          depthStoreOp: 'store',
        },
      });
      tilePass.setPipeline(GPUApp.rayTracingPipeline);
      tilePass.setBindGroup(0, GPUApp.bindGroup);
      tilePass.setViewport(x, y, w, h, 0.0, 1.0);
      tilePass.setScissorRect(x, y, w, h);
      tilePass.draw(6);
      tilePass.end();
      GPUApp.device.queue.submit([tileEncoder.finish()]);
      await GPUApp.device.queue.onSubmittedWorkDone();

      {
        const tileEnd = performance.now();
        tileTimes.push(tileEnd - tileStart);
      }
      if ((tx + ty * tilesX) % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
  }

  // Blit offscreen to swap chain
  const blitEncoder = GPUApp.device.createCommandEncoder();
  const blitPass = blitEncoder.beginRenderPass({
    label: 'Blit to canvas',
    sampleCount: 1,
    colorAttachments: [{
      view: GPUApp.context.getCurrentTexture().createView(),
      loadOp: 'clear',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: 'store',
    }],
  });
  blitPass.setPipeline(GPUApp.blitPipeline);
  blitPass.setBindGroup(0, GPUApp.blitBindGroup);
  blitPass.draw(6);
  blitPass.end();
  GPUApp.device.queue.submit([blitEncoder.finish()]);
  await GPUApp.device.queue.onSubmittedWorkDone();

  if (tileTimes.length > 0) {
    const totalTileMs = tileTimes.reduce((acc, t) => acc + t, 0);
    const meanTileMs = totalTileMs / tileCount;
    let variance = 0;
    for (let i = 0; i < tileCount; i++) {
      const d = tileTimes[i] - meanTileMs;
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
async function renderSceneAccumulation(GPUApp, scene) {
  const start = performance.now();
  const totalLights = scene.lightSources?.length ?? 0;
  const numPasses = Math.max(1, Math.ceil(totalLights / LIGHTS_PER_PASS));
  console.log('[Render] Starting accumulation RT:', totalLights, 'lights,', numPasses, 'passes of', LIGHTS_PER_PASS);

  updateUniforms(GPUApp, scene);
  updateDebugUniform(GPUApp);
  updateMaterialBuffer(GPUApp, scene.materials);
  updateLightSourceBuffer(GPUApp, scene.lightSources);

  const offscreenView = GPUApp.offscreenColorTexture.createView();
  const depthView = GPUApp.depthTexture.createView();
  const accumView = GPUApp.accumTexture.createView();

  // Clear accumulation texture to black
  {
    const enc = GPUApp.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      label: 'Clear accum',
      sampleCount: 1,
      colorAttachments: [{
        view: accumView,
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 0 }, // Transparent black
        storeOp: 'store',
      }],
    });
    pass.end();
    GPUApp.device.queue.submit([enc.finish()]);
    await GPUApp.device.queue.onSubmittedWorkDone();
  }

  for (let p = 0; p < numPasses; p++) {
    const lightStart = p * LIGHTS_PER_PASS;
    const lightEnd = Math.min(lightStart + LIGHTS_PER_PASS, totalLights);

    // Set the light range for this pass
    updateLightRange(GPUApp, lightStart, lightEnd);

    // Render full screen to offscreen texture (clear each pass)
    {
      const enc = GPUApp.device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        label: `Accum pass ${p}`,
        sampleCount: 1,
        colorAttachments: [{
          view: offscreenView,
          loadOp: 'clear',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      pass.setPipeline(GPUApp.rayTracingPipeline);
      pass.setBindGroup(0, GPUApp.bindGroup);
      pass.draw(6);
      pass.end();
      GPUApp.device.queue.submit([enc.finish()]);
      await GPUApp.device.queue.onSubmittedWorkDone();
    }

    // Additive blit: offscreen → accumulation texture (blend ONE+ONE)
    {
      const enc = GPUApp.device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        label: `Accum blit ${p}`,
        sampleCount: 1,
        colorAttachments: [{
          view: accumView,
          loadOp: 'load',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(GPUApp.accumBlitPipeline);
      pass.setBindGroup(0, GPUApp.accumBlitBindGroup);
      pass.draw(6);
      pass.end();
      GPUApp.device.queue.submit([enc.finish()]);
      await GPUApp.device.queue.onSubmittedWorkDone();
    }

    // Yield to browser periodically
    if (p % 5 === 4) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  // Final blit: accumulation texture → swap chain, dividing by numPasses
  updateAccumFinalPassCount(GPUApp, numPasses);
  {
    const enc = GPUApp.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      label: 'Accum final blit',
      sampleCount: 1,
      colorAttachments: [{
        view: GPUApp.context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store',
      }],
    });
    pass.setPipeline(GPUApp.accumFinalPipeline);
    pass.setBindGroup(0, GPUApp.accumFinalBindGroup);
    pass.draw(6);
    pass.end();
    GPUApp.device.queue.submit([enc.finish()]);
    await GPUApp.device.queue.onSubmittedWorkDone();
  }

  const end = performance.now();
  const frameMs = end - start;
  console.log('[Render] Accumulation frame in', frameMs.toFixed(3), 'ms (', numPasses, 'passes)');
  const label = document.getElementById('render_time_label');
  if (label) label.textContent = `${frameMs.toFixed(3)} ms`;
  return frameMs;
}

async function main() {
  console.log('[Main] Starting application');
  try {
    const GPUApp = await createGPUApp();
    const camAspect = GPUApp.canvas.width / GPUApp.canvas.height;
    console.log('[Main] Canvas aspect ratio =', camAspect);
    const shaderResponse = await fetch('shaders.wgsl');
    console.log('[Main] shaders.wgsl HTTP status =', shaderResponse.status);
    const shaderCode = await shaderResponse.text();
    initRenderPipeline(GPUApp, shaderCode);
    initAccumulationResources(GPUApp); // Create accum texture & pipelines

    const sceneSelect = document.getElementById('scene_select');
    const getSceneName = () => (sceneSelect && sceneSelect.value) || 'ram';
    let scene = await createScene(camAspect, getSceneName());
    console.log('[Main] Scene ready, meshes:', scene.meshes.length, 'lights:', scene.lightSources.length);
    let animationFrameId = null;
    let isRendering = false;

    async function renderLoop() {
      if (isRendering) return;
      const useRayTracing = isRayTracingEnabled();

      // Only continuously render when NOT using ray tracing
      if (!useRayTracing) {
        isRendering = true;
        await renderScene(GPUApp, scene);
        isRendering = false;
        animationFrameId = requestAnimationFrame(renderLoop);
      } else {
        animationFrameId = null;
      }
    }

    // Function to trigger a render (used by camera events)
    function triggerRender() {
      const useRayTracing = isRayTracingEnabled();
      if (!useRayTracing && !isRendering) {
        // For non-raytracing, the render loop will handle it
        // But we can also trigger immediately if needed
        if (!animationFrameId) {
          renderLoop();
        }
      }
    }

    initEvents(GPUApp, scene, triggerRender);
    initGPUBuffers(GPUApp, scene);
    console.log('[Main] GPU buffers initialized');

    // Initial render
    await renderScene(GPUApp, scene);

    // Start continuous rendering loop for non-raytracing mode
    const renderingTypeSelect = document.getElementById('rendering_type_select');
    function onRenderingTypeChange() {
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

    // Manual "Generate image" button (for ray tracing mode).
    const renderButton = document.getElementById('render_button');
    if (renderButton) {
      renderButton.addEventListener('click', async () => {
        if (!isRendering) {
          isRendering = true;
          await renderScene(GPUApp, scene);
          isRendering = false;
        }
      });
    }

    // Scene dropdown: reload scene and buffers when changed
    if (sceneSelect) {
      sceneSelect.addEventListener('change', async () => {
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
          animationFrameId = null;
        }
        const sceneName = getSceneName();
        try {
          scene = await createScene(camAspect, sceneName);
          initGPUBuffers(GPUApp, scene);
          await renderScene(GPUApp, scene);
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
          const panelId = panel.id.replace('panel-', '');
          const isActive = panelId === id;
          panel.classList.toggle('active', isActive);
          panel.hidden = !isActive;
        });
        const sidebarEl = document.getElementById('playground_sidebar');
        if (sidebarEl) sidebarEl.style.display = id === 'playground' ? 'flex' : 'none';
      });
    });
    const sidebarEl = document.getElementById('playground_sidebar');

    // ─── Lightcut Tree tab ─────────────────────────────────────────────────
    let lightcutTree = null;
    let lightcutGPU = null; // Separate GPU context for lightcut canvas

    async function initLightcutGPU() {
      if (lightcutGPU) return lightcutGPU;
      const lcCanvas = document.getElementById('lightcut_canvas');
      if (!lcCanvas) return null;
      // Share the same adapter & device with the main GPUApp
      const lcContext = lcCanvas.getContext('webgpu');
      const lcFormat = navigator.gpu.getPreferredCanvasFormat();
      lcContext.configure({ device: GPUApp.device, format: lcFormat, alphaMode: 'opaque' });
      lightcutGPU = {
        canvas: lcCanvas,
        device: GPUApp.device,
        context: lcContext,
        canvasFormat: lcFormat,
        shaderModule: GPUApp.shaderModule,
        bindGroupLayout: GPUApp.bindGroupLayout,
        rasterizationPipeline: GPUApp.rasterizationPipeline,
        depthTexture: GPUApp.device.createTexture({
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
    function buildLightcutScene(baseScene, nodes) {
      // Clone the essential parts
      const vizScene = {
        camera: baseScene.camera,
        meshes: [...baseScene.meshes.slice(0, baseScene.baseMeshCount || baseScene.meshes.length)],
        materials: [...baseScene.materials],
        lightSources: [], // will be replaced with fill lights below
        baseMeshCount: baseScene.baseMeshCount || baseScene.meshes.length,
      };

      // Create fill lights from 6 directions so all box faces are visible.
      // Use wide cones (angle=0) and moderate intensity.
      const bounds = { minX: Infinity, minY: Infinity, minZ: Infinity, maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity };
      for (const m of vizScene.meshes) {
        if (!m.positions) continue;
        for (let i = 0; i < m.positions.length; i += 3) {
          bounds.minX = Math.min(bounds.minX, m.positions[i]);
          bounds.minY = Math.min(bounds.minY, m.positions[i + 1]);
          bounds.minZ = Math.min(bounds.minZ, m.positions[i + 2]);
          bounds.maxX = Math.max(bounds.maxX, m.positions[i]);
          bounds.maxY = Math.max(bounds.maxY, m.positions[i + 1]);
          bounds.maxZ = Math.max(bounds.maxZ, m.positions[i + 2]);
        }
      }
      const cx = (bounds.minX + bounds.maxX) * 0.5;
      const cy = (bounds.minY + bounds.maxY) * 0.5;
      const cz = (bounds.minZ + bounds.maxZ) * 0.5;
      const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ) * 1.5;
      const center = [cx, cy, cz];
      const fillLight = (pos) => ({
        position: pos,
        intensity: 0.3,
        color: [1, 1, 1],
        spot: center,
        angle: 0.0, // wide cone — no angular cutoff
        useRaytracedShadows: false,
      });
      vizScene.lightSources = [
        fillLight([cx, cy + span, cz]),   // above
        fillLight([cx, cy - span, cz]),   // below
        fillLight([cx + span, cy, cz]),   // right
        fillLight([cx - span, cy, cz]),   // left
        fillLight([cx, cy, cz + span]),   // front
        fillLight([cx, cy, cz - span]),   // back
      ];

      // Create per-node materials colored by intensity (red → green)
      const baseMaterialIndex = vizScene.materials.length;
      const nodeMaterials = createIntensityMaterials(nodes);
      for (const mat of nodeMaterials) {
        vizScene.materials.push(mat);
      }

      // Create and add solid box meshes (each node has its own material)
      const boxMeshes = createBBoxMeshes(nodes, baseMaterialIndex);
      for (const mesh of boxMeshes) {
        vizScene.meshes.push(mesh);
      }

      return vizScene;
    }

    /** Render the lightcut visualization on the dedicated canvas. */
    async function renderLightcutViz(depth) {
      if (!lightcutTree) {
        console.warn('[Lightcut] No tree built yet');
        return;
      }
      const lcGPU = await initLightcutGPU();
      if (!lcGPU) return;

      const maxDepth = getTreeMaxDepth(lightcutTree);
      const nodes = getNodesAtDepth(lightcutTree, depth);

      // Update stats UI
      const nodeCountEl = document.getElementById('lightcut_node_count');
      if (nodeCountEl) nodeCountEl.textContent = nodes.length;

      // Build the visualization scene
      const vizScene = buildLightcutScene(scene, nodes);

      // Init GPU buffers for the lightcut context
      const { createMeshBuffers, createMaterialBuffer, createLightSourceBuffer, createGPUBuffer } = await import('./gpu.js');
      lcGPU.meshBuffers = createMeshBuffers(lcGPU, vizScene.meshes);
      lcGPU.materialBuffer = createMaterialBuffer(lcGPU, vizScene.materials);
      lcGPU.lightSourceBuffer = createLightSourceBuffer(lcGPU, vizScene.lightSources);
      lcGPU.uniformBuffer = createGPUBuffer(lcGPU.device, lcGPU.uniformData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      lcGPU.debugUniformBuffer = createGPUBuffer(lcGPU.device, lcGPU.debugUniformData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
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
      const { updateCamera } = await import('./camera.js');
      const { mat4Invert, mat4Transpose } = await import('./math.js');
      updateCamera(vizScene.camera);
      lcGPU.uniformData.set(vizScene.camera.modelMat, 0);
      lcGPU.uniformData.set(vizScene.camera.viewMat, 16);
      const invViewMat = mat4Invert(vizScene.camera.viewMat);
      lcGPU.uniformData.set(invViewMat, 32);
      lcGPU.uniformData.set(mat4Transpose(invViewMat), 48);
      lcGPU.uniformData.set(vizScene.camera.projMat, 64);
      lcGPU.uniformData[80] = vizScene.camera.fov;
      lcGPU.uniformData[81] = vizScene.camera.aspect;
      lcGPU.uniformData[84] = vizScene.meshes.length; // render ALL meshes including boxes
      lcGPU.uniformData[85] = vizScene.lightSources.length;
      lcGPU.uniformData[86] = 0;
      lcGPU.uniformData[87] = vizScene.lightSources.length;
      lcGPU.uniformData[88] = lcGPU.canvas.width;
      lcGPU.uniformData[89] = lcGPU.canvas.height;
      lcGPU.device.queue.writeBuffer(lcGPU.uniformBuffer, 0, lcGPU.uniformData.buffer, lcGPU.uniformData.byteOffset, lcGPU.uniformData.byteLength);

      // Debug uniform: mode 2 = raw albedo (unlit — just show material color)
      lcGPU.debugUniformData[0] = 2;
      lcGPU.device.queue.writeBuffer(lcGPU.debugUniformBuffer, 0, lcGPU.debugUniformData.buffer, lcGPU.debugUniformData.byteOffset, lcGPU.debugUniformData.byteLength);

      // Fill light source staging
      const { fillLightSourceStagingBuffer } = await import('./gpu.js');
      // Use the shared light source buffer — already created above

      // Render pass
      const encoder = lcGPU.device.createCommandEncoder();
      const renderPass = encoder.beginRenderPass({
        label: 'Lightcut raster pass',
        sampleCount: 1,
        colorAttachments: [{
          view: lcGPU.context.getCurrentTexture().createView(),
          loadOp: 'clear',
          clearValue: { r: 0.06, g: 0.06, b: 0.07, a: 1 },
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: lcGPU.depthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      renderPass.setPipeline(lcGPU.rasterizationPipeline);
      renderPass.setBindGroup(0, lcGPU.bindGroup);
      for (let i = 0; i < vizScene.meshes.length; i++) {
        renderPass.draw(vizScene.meshes[i].indices.length, 1, 0, i);
      }
      renderPass.end();
      lcGPU.device.queue.submit([encoder.finish()]);
      await lcGPU.device.queue.onSubmittedWorkDone();

      //console.log('[Lightcut] Rendered', nodes.length, 'bounding boxes at depth', depth);
    }

    // Build button
    const lightcutBuildBtn = document.getElementById('lightcut_build_btn');
    const lightcutMethodSelect = document.getElementById('lightcut_method_select');
    const lightcutDepthSlider = document.getElementById('lightcut_depth_slider');
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

        // Use setTimeout to let the UI update
        await new Promise(r => setTimeout(r, 10));

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

        // Update UI
        if (lightcutMaxDepthEl) lightcutMaxDepthEl.textContent = maxDepth;
        if (lightcutTotalLightsEl) lightcutTotalLightsEl.textContent = scene.lightSources.length;
        if (lightcutBuildTimeEl) lightcutBuildTimeEl.textContent = buildTime.toFixed(1) + ' ms';

        // Configure depth slider
        if (lightcutDepthSlider) {
          lightcutDepthSlider.max = maxDepth;
          lightcutDepthSlider.value = 0;
          lightcutDepthSlider.disabled = false;
        }
        if (lightcutDepthValue) lightcutDepthValue.textContent = '0';

        lightcutBuildBtn.disabled = false;
        lightcutBuildBtn.textContent = 'Build Tree';

        console.log(`[Lightcut] Built ${method} tree: maxDepth=${maxDepth}, lights=${scene.lightSources.length}, time=${buildTime.toFixed(1)}ms`);

        // Render at depth 0
        await renderLightcutViz(0);
      });
    }

    // Depth slider
    if (lightcutDepthSlider) {
      lightcutDepthSlider.addEventListener('input', async () => {
        const depth = parseInt(lightcutDepthSlider.value, 10);
        if (lightcutDepthValue) lightcutDepthValue.textContent = depth;
        await renderLightcutViz(depth);
      });
    }

    // Camera controls for lightcut canvas
    const lcCanvas = document.getElementById('lightcut_canvas');
    if (lcCanvas) {
      lcCanvas.addEventListener('mousedown', (e) => {
        scene.camera.lastX = e.clientX;
        scene.camera.lastY = e.clientY;
        if (e.button === 0) scene.camera._lcDragging = true;
        if (e.button === 1 || e.button === 2) scene.camera._lcPanning = true;
      });
      window.addEventListener('mouseup', () => {
        scene.camera._lcDragging = false;
        scene.camera._lcPanning = false;
      });
      lcCanvas.addEventListener('mousemove', async (e) => {
        const dx = e.clientX - scene.camera.lastX;
        const dy = e.clientY - scene.camera.lastY;
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
      lcCanvas.addEventListener('wheel', async (e) => {
        e.preventDefault();
        scene.camera.radius *= 1 + e.deltaY * scene.camera.zoomSpeed;
        scene.camera.radius = Math.max(scene.camera.minRadius, Math.min(scene.camera.maxRadius, scene.camera.radius));
        if (lightcutTree) {
          const depth = parseInt(lightcutDepthSlider?.value || '0', 10);
          await renderLightcutViz(depth);
        }
      }, { passive: false });
      lcCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }
    if (sidebarEl) sidebarEl.style.display = 'flex';

    // Full-lights training: run N images (slider) with random north-hemisphere cameras; images appear one by one
    const fullLightsRunBtn = document.getElementById('full_lights_run_btn');
    const fullLightsNumImagesSlider = document.getElementById('full_lights_num_images');
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
        if (isRendering) {
          console.log('[FullLights] Ignored click: already rendering');
          return;
        }
        const numImages = Math.max(1, parseInt(fullLightsNumImagesSlider?.value || '10', 10));
        console.log('[FullLights] Starting generation of', numImages, 'images');
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
          animationFrameId = null;
        }
        isRendering = true;
        fullLightsRunBtn.disabled = true;
        fullLightsGrid.innerHTML = '';
        if (fullLightsLastRun) fullLightsLastRun.textContent = 'Running…';

        try {
          const { times } = await runFullLightsTraining(GPUApp, scene, numImages, {
            onImage(index, dataUrl) {
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
          if (fullLightsLastRun) fullLightsLastRun.textContent = 'Error: ' + (err.message || String(err));
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
