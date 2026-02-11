import { createScene, setCameraTopDown, setCameraRandomNorthHemisphere } from './scene.js';
import { createGPUApp, initRenderPipeline, initGPUBuffers, updateUniforms, updateMaterialBuffer, updateLightSourceBuffer, updateDebugUniform } from './gpu.js';
import { pan } from './camera.js';

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
  const { onImage = () => {}, forceRayTracing = true, syncIntensitySliderToMode } = options;
  console.log('[FullLights] runFullLightsTraining started, numImages =', numImages);
  const rtCheckbox = document.getElementById('raytracingCheckbox');
  const wasRT = rtCheckbox && rtCheckbox.checked;
  if (forceRayTracing && rtCheckbox) rtCheckbox.checked = true;
  if (typeof syncIntensitySliderToMode === 'function') syncIntensitySliderToMode();

  const times = [];
  for (let i = 0; i < numImages; i++) {
    setCameraRandomNorthHemisphere(scene);
    const timeMs = await renderScene(GPUApp, scene);
    times.push(timeMs);
    const dataUrl = getCanvasDataURL(GPUApp.canvas);
    onImage(i, dataUrl, timeMs);
  }

  if (forceRayTracing && rtCheckbox && !wasRT) rtCheckbox.checked = false;
  if (typeof syncIntensitySliderToMode === 'function') syncIntensitySliderToMode();
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
  const useRayTracing = document.getElementById('raytracingCheckbox').checked;
  const start = performance.now();
  if (useRayTracing) {
    console.log('[Render] Starting image (ray tracing)', scene.lightSources?.length ?? 0, 'lights');
  }
  const debugRenderCheckbox = document.getElementById('debug_render_checkbox');
  const debugRenderEnabled = debugRenderCheckbox && debugRenderCheckbox.checked;
  let tileCount = 0;

  updateUniforms(GPUApp, scene);
  updateDebugUniform(GPUApp);
  updateMaterialBuffer(GPUApp, scene.materials);
  updateLightSourceBuffer(GPUApp, scene.lightSources);
  const encoder = GPUApp.device.createCommandEncoder();
  const renderPass = encoder.beginRenderPass({
    label: 'Main rendering pass',
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
  renderPass.setPipeline(useRayTracing ? GPUApp.rayTracingPipeline : GPUApp.rasterizationPipeline);
  renderPass.setBindGroup(0, GPUApp.bindGroup);
  if (!useRayTracing) {
    const debugCheckbox = document.getElementById('debug_lights_checkbox');
    const debugOn = debugCheckbox && debugCheckbox.checked;
    const baseMeshCount = typeof scene.baseMeshCount === 'number' ? scene.baseMeshCount : scene.meshes.length;

    // Draw main scene meshes.
    for (let i = 0; i < baseMeshCount; i++) {
      renderPass.draw(scene.meshes[i].indices.length, 1, 0, i);
    }
    // Optionally draw debug light marker meshes.
    if (debugOn) {
      for (let i = baseMeshCount; i < scene.meshes.length; i++) {
        renderPass.draw(scene.meshes[i].indices.length, 1, 0, i);
      }
    }
  } else {
    // Ray tracing path: render to an offscreen texture (tiles accumulate), then copy to canvas once.
    // This avoids swapchain reuse and ensures we see the full image. Tile size from scene camera.txt.
    const tileSize = Math.max(32, Math.min(256, (scene.cameraConfig && typeof scene.cameraConfig.tileSize === 'number') ? scene.cameraConfig.tileSize : 256));
    const tilesX = Math.ceil(GPUApp.canvas.width / tileSize);
    const tilesY = Math.ceil(GPUApp.canvas.height / tileSize);
    tileCount = tilesX * tilesY;

    const offscreenView = GPUApp.offscreenColorTexture.createView();
    const depthView = GPUApp.depthTexture.createView();

    // Clear offscreen (our texture; safe to reuse view across submits)
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
    
    // All tiles render to offscreen (same view); submit + wait each so tiles run one-after-the-other (TDR).
    const tileTimes = debugRenderEnabled ? [] : null;
    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const tileStart = debugRenderEnabled ? performance.now() : 0;
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
        
        if (debugRenderEnabled) {
          const tileEnd = performance.now();
          tileTimes.push(tileEnd - tileStart);
        }
      }
    }
    
    // Blit offscreen to swap chain (render pass; swap chain has no COPY_DST)
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
    
    // Log tile statistics if debug is enabled
    if (debugRenderEnabled && tileTimes && tileTimes.length > 0) {
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
  }
  
  // For non-ray-tracing path, end the render pass and submit
  if (!useRayTracing) {
    renderPass.end();
    GPUApp.device.queue.submit([encoder.finish()]);
    // Wait until GPU work for this frame is actually finished.
    await GPUApp.device.queue.onSubmittedWorkDone();
  }

  const end = performance.now();
  const frameMs = end - start;
  const ms = frameMs.toFixed(3);
  if (debugRenderEnabled && useRayTracing) {
    console.log('[Render] Frame generated in', ms, 'ms using', scene.lightSources.length, 'lights');
  }
  const label = document.getElementById('render_time_label');
  if (label) {
    label.textContent = `${ms} ms`;
  }
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

    const sceneSelect = document.getElementById('scene_select');
    const getSceneName = () => (sceneSelect && sceneSelect.value) || 'ram';
    let scene = await createScene(camAspect, getSceneName());
    console.log('[Main] Scene ready, meshes:', scene.meshes.length, 'lights:', scene.lightSources.length);
    let animationFrameId = null;
    let isRendering = false;

    async function renderLoop() {
      if (isRendering) return;
      const useRayTracing = document.getElementById('raytracingCheckbox').checked;
      
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
      const useRayTracing = document.getElementById('raytracingCheckbox').checked;
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

    // Intensity slider: different range per mode. Raster = 0–3 (default 1); RT = 0–0.5 (default 0.3).
    const intensitySlider = document.getElementById('intensity_slider');
    const intensityValue = document.getElementById('intensity_value');
    function syncIntensitySliderToMode() {
      if (!intensitySlider || !intensityValue) return;
      const useRT = document.getElementById('raytracingCheckbox').checked;
      if (useRT) {
        intensitySlider.min = '0';
        intensitySlider.max = '0.5';
        intensitySlider.step = '0.01';
        intensitySlider.value = '0.3';
      } else {
        intensitySlider.min = '0';
        intensitySlider.max = '3';
        intensitySlider.step = '0.1';
        intensitySlider.value = '1';
      }
      intensityValue.textContent = intensitySlider.value;
    }

    // Start continuous rendering loop for non-raytracing mode
    const raytracingCheckbox = document.getElementById('raytracingCheckbox');
    if (raytracingCheckbox) {
      syncIntensitySliderToMode();
      raytracingCheckbox.addEventListener('change', () => {
        const useRayTracing = raytracingCheckbox.checked;
        syncIntensitySliderToMode();
        if (!useRayTracing && !animationFrameId) {
          // Start continuous rendering when ray tracing is turned off
          renderLoop();
        } else if (useRayTracing && animationFrameId) {
          // Stop continuous rendering when ray tracing is turned on
          cancelAnimationFrame(animationFrameId);
          animationFrameId = null;
        }
      });
      
      // Start loop if ray tracing is initially off
      if (!raytracingCheckbox.checked) {
        renderLoop();
      }
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

    // Intensity slider: update label and trigger render.
    if (intensitySlider && intensityValue) {
      intensitySlider.addEventListener('input', () => {
        intensityValue.textContent = intensitySlider.value;
        const useRT = document.getElementById('raytracingCheckbox').checked;
        if (!useRT && !isRendering && !animationFrameId) renderLoop();
      });
      intensitySlider.addEventListener('change', () => {
        intensityValue.textContent = intensitySlider.value;
        const useRT = document.getElementById('raytracingCheckbox').checked;
        if (useRT && !isRendering) {
          isRendering = true;
          renderScene(GPUApp, scene).then(() => { isRendering = false; });
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
          syncIntensitySliderToMode();
          await renderScene(GPUApp, scene);
          if (raytracingCheckbox && !raytracingCheckbox.checked) renderLoop();
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
            syncIntensitySliderToMode,
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
