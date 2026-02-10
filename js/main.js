import { createScene } from './scene.js';
import { createGPUApp, initRenderPipeline, initGPUBuffers, updateUniforms, updateMaterialBuffer, updateLightSourceBuffer } from './gpu.js';
import { pan } from './camera.js';
import { srgbToHex, hexToSrgb, linearVec3ToSrgb, srgbVec3ToLinear } from './color.js';

function syncUI(scene) {
  const roughnessSlider = document.getElementById('roughness_slider');
  const roughnessLabel = document.getElementById('roughness_label');
  const mat = scene.materials[scene.materials.length - 1];
  roughnessSlider.value = mat.roughness;
  roughnessLabel.textContent = mat.roughness.toFixed(2);
}

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

  const material = scene.materials[scene.materials.length - 1];
  const roughnessSlider = document.getElementById('roughness_slider');
  roughnessSlider.value = material.roughness;
  roughnessSlider.addEventListener('input', () => { material.roughness = parseFloat(roughnessSlider.value); });
  const roughnessLabel = document.getElementById('roughness_label');
  roughnessLabel.textContent = material.roughness.toFixed(2);

  const metalnessSlider = document.getElementById('metalness_slider');
  metalnessSlider.value = material.metalness;
  metalnessSlider.addEventListener('input', () => { material.metalness = parseFloat(metalnessSlider.value); });
  const metalnessLabel = document.getElementById('metalness_label');
  metalnessLabel.textContent = material.metalness.toFixed(2);

  const albedoPicker = document.getElementById('albedo_picker');
  albedoPicker.value = srgbToHex(linearVec3ToSrgb(material.albedo));
  albedoPicker.addEventListener('input', () => {
    material.albedo = srgbVec3ToLinear(hexToSrgb(albedoPicker.value));
  });
}

function animate(scene, time) {
  scene.time = time;
  const t = time * 0.05;
  const angle = t / 40.0;
  scene.lightSources[scene.lightSources.length - 1].position = [0.5 * Math.cos(angle), 0.9, 0.5 * Math.sin(angle)];
}

function renderFrame(GPUApp, scene, time) {
  const useRayTracing = document.getElementById('raytracingCheckbox').checked;
  if (document.getElementById('animateCheckbox').checked) animate(scene, time);
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
  requestAnimationFrame((t) => renderFrame(GPUApp, scene, t));
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
    syncUI(scene);
    initGPUBuffers(GPUApp, scene);
    console.log('[Main] GPU buffers initialized, starting render loop');
    renderFrame(GPUApp, scene);
  } catch (err) {
    console.error('[Main] Fatal error during initialization:', err);
  }
}

main();
