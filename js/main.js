import { createScene } from './scene.js';
import { createGPUApp, initRenderPipeline, initGPUBuffers, updateUniforms, updateMaterialBuffer, updateLightSourceBuffer } from './gpu.js';
import { pan } from './camera.js';

function initEvents(GPUApp, scene) {
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
    }
    if (scene.camera.panning) pan(scene.camera, dx, -dy);
  });
  GPUApp.canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    scene.camera.radius *= 1 + e.deltaY * scene.camera.zoomSpeed;
    scene.camera.radius = Math.max(scene.camera.minRadius, Math.min(scene.camera.maxRadius, scene.camera.radius));
  }, { passive: false });
  GPUApp.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
}

function renderScene(GPUApp, scene) {
  const useRayTracing = document.getElementById('raytracingCheckbox').checked;
  const start = performance.now();

  updateUniforms(GPUApp, scene);
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
    for (let i = 0; i < scene.meshes.length; i++)
      renderPass.draw(scene.meshes[i].indices.length, 1, 0, i);
  } else {
    renderPass.draw(6);
  }
  renderPass.end();
  GPUApp.device.queue.submit([encoder.finish()]);

  const end = performance.now();
  const ms = (end - start).toFixed(3);

  const debugRenderCheckbox = document.getElementById('debug_render_checkbox');
  const debugRenderEnabled = debugRenderCheckbox && debugRenderCheckbox.checked;
  if (debugRenderEnabled) {
    console.log('[Render] Frame generated in', ms, 'ms using', scene.lightSources.length, 'lights');
  }
  const label = document.getElementById('render_time_label');
  if (label) {
    label.textContent = `${ms} ms`;
  }
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
    const scene = await createScene(camAspect);
    console.log('[Main] Scene ready, meshes:', scene.meshes.length, 'lights:', scene.lightSources.length);
    initEvents(GPUApp, scene);
    initGPUBuffers(GPUApp, scene);
    console.log('[Main] GPU buffers initialized');

    // Initial image.
    renderScene(GPUApp, scene);

    // Manual "Generate image" button.
    const renderButton = document.getElementById('render_button');
    if (renderButton) {
      renderButton.addEventListener('click', () => {
        renderScene(GPUApp, scene);
      });
    }
  } catch (err) {
    console.error('[Main] Fatal error during initialization:', err);
  }
}

main();
