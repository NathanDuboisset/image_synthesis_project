import type { Vec3 } from './types.ts';

export function hexToSrgb(hex: string): Vec3 {
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ];
}

export function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export function srgbVec3ToLinear(rgb: Vec3): Vec3 {
  return rgb.map(srgbToLinear) as Vec3;
}

export function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

export function linearVec3ToSrgb(rgb: Vec3): Vec3 {
  return rgb.map(linearToSrgb) as Vec3;
}

export function srgbToHex(rgb: Vec3): string {
  const toByte = (c: number): number => Math.min(255, Math.max(0, Math.round(c * 255)));
  return '#' + rgb.map(toByte).map(n => n.toString(16).padStart(2, '0')).join('');
}
