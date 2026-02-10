import { vec3Sub, vec3Cross, vec3Normalize } from './math.js';

export function computeNormals(mesh) {
  const numOfTri = mesh.indices.length / 3;
  const length = mesh.normals.length;
  for (let i = 0; i < length; ++i) mesh.normals[i] = 0.0;
  for (let i = 0; i < numOfTri; ++i) {
    const v0 = mesh.indices[3*i], v1 = mesh.indices[3*i+1], v2 = mesh.indices[3*i+2];
    const p0 = [mesh.positions[3*v0], mesh.positions[3*v0+1], mesh.positions[3*v0+2]];
    const p1 = [mesh.positions[3*v1], mesh.positions[3*v1+1], mesh.positions[3*v1+2]];
    const p2 = [mesh.positions[3*v2], mesh.positions[3*v2+1], mesh.positions[3*v2+2]];
    const e01 = vec3Sub(p1, p0);
    const e12 = vec3Sub(p2, p1);
    const nt = vec3Normalize(vec3Cross(e01, e12));
    for (const v of [v0, v1, v2]) {
      mesh.normals[3*v] += nt[0]; mesh.normals[3*v+1] += nt[1]; mesh.normals[3*v+2] += nt[2];
    }
  }
  for (let i = 0; i < length/3; ++i) {
    const ni = [mesh.normals[3*i], mesh.normals[3*i+1], mesh.normals[3*i+2]];
    const nni = vec3Normalize(ni);
    mesh.normals[3*i] = nni[0]; mesh.normals[3*i+1] = nni[1]; mesh.normals[3*i+2] = nni[2];
  }
}

export function createQuad(origin, edge0, edge1) {
  const vec3Add = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
  const a = vec3Add(origin, edge0);
  const b = vec3Add(a, edge1);
  const c = vec3Add(origin, edge1);
  const positions = origin.concat(a).concat(b).concat(c);
  const n = vec3Cross(edge0, edge1);
  const normals = n.concat(n).concat(n).concat(n);
  const mesh = {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
  return mesh;
}

export function createBox(width, height, length) {
  const vec3Length = (x) => Math.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  const w = width/2, h = height/2, l = length/2;
  const positions = [-w,-h,l, w,-h,l, w,h,l, -w,h,l, -w,-h,-l, w,-h,-l, w,h,-l, -w,h,-l];
  const den = vec3Length([w,h,l]);
  const normals = [-w/den,-h/den,l/den, w/den,-h/den,l/den, w/den,h/den,l/den, -w/den,h/den,l/den,
    -w/den,-h/den,-l/den, w/den,-h/den,-l/den, w/den,h/den,-l/den, -w/den,h/den,-l/den];
  const indices = [0,1,2,0,2,3, 1,5,6,1,6,2, 5,4,7,5,7,6, 4,0,3,4,3,7, 3,2,6,3,6,7, 4,5,1,4,1,0];
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: positions.length/3 > 65535 ? new Uint32Array(indices) : new Uint16Array(indices),
  };
}

export function createCube(size) {
  return createBox(size, size, size);
}

export function createSphere(radius, latitudeRes, longitudeRes) {
  const positions = [], normals = [], indices = [];
  for (let lat = 0; lat <= latitudeRes; lat++) {
    const theta = lat * Math.PI / latitudeRes;
    const sinTheta = Math.sin(theta), cosTheta = Math.cos(theta);
    for (let lon = 0; lon <= longitudeRes; lon++) {
      const phi = lon * 2 * Math.PI / longitudeRes;
      const x = Math.cos(phi) * sinTheta, y = cosTheta, z = Math.sin(phi) * sinTheta;
      positions.push(radius*x, radius*y, radius*z);
      normals.push(x, y, z);
    }
  }
  for (let lat = 0; lat < latitudeRes; lat++) {
    for (let lon = 0; lon < longitudeRes; lon++) {
      const first = lat * (longitudeRes + 1) + lon, second = first + longitudeRes + 1;
      indices.push(first, first+1, second, second, first+1, second+1);
    }
  }
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices),
  };
}

/** Build mesh from ram-mesh.json data (positions + indices). */
export function createRamFromData(data) {
  const mesh = {
    positions: new Float32Array(data.positions),
    normals: new Float32Array(data.positions.length),
    indices: new Uint32Array(data.indices),
  };
  computeNormals(mesh);
  return mesh;
}
