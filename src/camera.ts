import type { Camera, Mat4, Vec3 } from './types.ts';
import { mat4Invert, mat4Transpose, mat4Perspective } from './math.ts';

export function getCameraPosition(camera: Camera): Vec3 {
  return [
    camera.target[0] + camera.radius * Math.cos(camera.pitch) * Math.sin(camera.yaw),
    camera.target[1] + camera.radius * Math.sin(camera.pitch),
    camera.target[2] + camera.radius * Math.cos(camera.pitch) * Math.cos(camera.yaw),
  ];
}

export function pan(camera: Camera, dx: number, dy: number): void {
  const cosYaw = Math.cos(camera.yaw), sinYaw = Math.sin(camera.yaw);
  const rightX = cosYaw, rightZ = -sinYaw;
  const scale = camera.radius * camera.panSpeed;
  camera.target[0] += (-rightX * dx + 0 * dy) * scale;
  camera.target[1] += (0 * dx + 1 * dy) * scale;
  camera.target[2] += (-rightZ * dx + 0 * dy) * scale;
}

export function lookAt(out: Mat4, eye: Vec3, target: Vec3, up: Vec3): void {
  const zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
  let len = 1 / Math.sqrt(zx * zx + zy * zy + zz * zz);
  const z0 = zx * len, z1 = zy * len, z2 = zz * len;
  let xx = up[1] * z2 - up[2] * z1, xy = up[2] * z0 - up[0] * z2, xz = up[0] * z1 - up[1] * z0;
  len = 1 / Math.sqrt(xx * xx + xy * xy + xz * xz);
  const x0 = xx * len, x1 = xy * len, x2 = xz * len;
  const y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
  out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
  out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
  out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
  out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
  out[15] = 1;
}

export function updateCamera(camera: Camera): void {
  camera.viewMat = new Float32Array(16);
  const eye = getCameraPosition(camera);
  lookAt(camera.viewMat, eye, camera.target, [0, 1, 0]);
  camera.invViewMat = mat4Invert(camera.viewMat);
  camera.transInvViewMat = mat4Transpose(camera.invViewMat);
  camera.projMat = mat4Perspective(camera.fov, camera.aspect, camera.near, camera.far);
  camera.modelMat = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

export function createCamera(aspect: number): Camera {
  const camera: Camera = {
    viewMat: new Float32Array(16),
    invViewMat: new Float32Array(16),
    transInvViewMat: new Float32Array(16),
    projMat: new Float32Array(16),
    modelMat: new Float32Array(16),
    fov: Math.PI / 4,
    aspect,
    near: 0.1,
    far: 100,
    // orbit parameters
    yaw: 0,
    pitch: 0,
    radius: 3,
    target: [0, 0, 0],
    rotateSpeed: 0.005,
    panSpeed: 0.001,
    zoomSpeed: 0.001,
    minRadius: 0.5,
    maxRadius: 20,
  };
  updateCamera(camera);
  return camera;
}
