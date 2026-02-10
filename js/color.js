export function hexToSrgb(hex) {
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ];
}

export function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export function srgbVec3ToLinear(rgb) {
  return rgb.map(srgbToLinear);
}

export function linearToSrgb(c) {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1/2.4) - 0.055;
}

export function linearVec3ToSrgb(rgb) {
  return rgb.map(linearToSrgb);
}

export function srgbToHex(rgb) {
  const toByte = c => Math.min(255, Math.max(0, Math.round(c * 255)));
  return '#' + rgb.map(toByte).map(n => n.toString(16).padStart(2, '0')).join('');
}
