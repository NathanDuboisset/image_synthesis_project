const PI = 3.14159265358979323846;
const INV_PI = 1.0/3.14159265358979323846;
const EPSILON = 1e-6;

struct LightSource {
  position: vec3<f32>,
  intensity: f32,
  color: vec3<f32>,
  angle: f32,
  spot: vec3<f32>,
  rayTracedShadows: u32,
};

struct Material {
  albedo: vec3<f32>,
  roughness: f32,
  metalness: f32,
  _pad2: vec2<f32> // Required for uniform buffer alignment
};

struct Camera {
  modelMat: mat4x4<f32>,
  viewMat: mat4x4<f32>,
  invViewMat: mat4x4<f32>,
  transInvViewMat: mat4x4<f32>,
  projMat: mat4x4<f32>,
  fov: f32,
  aspectRatio: f32,
  _pad: vec2<f32> // Required for uniform buffer alignment
};

struct Mesh {
    posOffset: u32, // in vertices, used for all attribute buffer for now
    triOffset: u32, // in triangles
    numOfTriangles: u32,
    materialIndex: u32, // index over the material buffer
};

struct Scene {
  camera: Camera,
  numOfMeshes: f32,
  numOfLightSources: f32,
  lightStartIndex: f32,
  lightEndIndex: f32,
  screenWidth: f32,
  screenHeight: f32,
};

struct DebugParams {
  mode: u32,
  lightcutNodeCount: u32,
  maxCutSize: u32,
  _pad: u32,
};

@group(0) @binding(0)
var<uniform> scene : Scene; // The only uniform buffer, as the camera paramters change frequently

@group(0) @binding(1)
var<storage, read> positions : array<f32>; // Packed positions for all meshes

@group(0) @binding(2)
var<storage, read> normals : array<f32>; // Packed normals for all meshes

@group(0) @binding(3)
var<storage, read> triangles : array<u32>; // Packed triangles for all meshes

@group(0) @binding(4)
var<storage, read> meshes : array<Mesh>; // The scene's meshes

@group(0) @binding(5)
var<storage, read> materials : array<Material>; // matei

@group(0) @binding(6)
var<storage, read> lightSources : array<LightSource>;

@group(0) @binding(7)
var<uniform> debugParams : DebugParams;

// ─── Lightcut tree (GPU storage) ──────────────────────────────────────────

struct LightcutGPUNode {
  position: vec3<f32>,     // representative position (world space)
  intensity: f32,          // cluster totalIntensity
  color: vec3<f32>,        // representative color
  lightCount: f32,         // number of leaves under this node
  aabbMin: vec3<f32>,
  leftChild: f32,          // BFS index, -1 = leaf
  aabbMax: vec3<f32>,
  rightChild: f32,         // BFS index, -1 = leaf
};

@group(0) @binding(8)
var<storage, read> lightcutTree : array<LightcutGPUNode>;

struct RasterVertexInput {
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) meshIndex: u32
};

struct RasterVertexOutput {
  @builtin(position) builtInPos : vec4f,
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) materialIndex: u32,
};

fn getVertPos(vertIndex: u32) -> vec3f {
  return vec3f (positions[3*vertIndex], positions[3*vertIndex+1], positions[3*vertIndex+2]);
}

fn getVertNormal(vertIndex: u32) -> vec3f {
  return vec3f (normals[3*vertIndex], normals[3*vertIndex+1], normals[3*vertIndex+2]);
}

fn getTriangle(triIndex: u32) -> vec3u {
  return vec3u (triangles[3*triIndex], triangles[3*triIndex+1], triangles[3*triIndex+2]);
}

fn sqr(x: f32) -> f32 {
  return x*x;
}

fn attenuation(dist: f32, coneDecay: f32) -> f32 {
  return coneDecay * (1.0 / sqr(dist));
  //return coneDecay / (1.0 + 0.02 * sqr(dist));
}

fn TrowbridgeReitzNDF(wh : vec3f, n : vec3f, roughness: f32) -> f32 {
  let alpha2 = sqr(roughness);
  return alpha2 / (PI * sqr(1.0 + (alpha2 - 1.0) * sqr(dot (n, wh))));
}

fn SchlickFresnel(wi: vec3f, wh: vec3f, F0: vec3f) -> vec3f {
    return F0 + (1.0 - F0) * pow(1.0 - max(0.0, dot(wi, wh)), 5.0);
}

fn SmithG1(w: vec3f, n : vec3f, roughness: f32) -> f32 {
  let NdotW = dot(n,w);
  let alpha2 = sqr (roughness);
  return (2.0 * NdotW) / (NdotW + sqrt(alpha2 + (1-alpha2)*sqr(NdotW)));
}

fn SmithGGX(wi : vec3f, wo : vec3f, n: vec3f, roughness : f32) -> f32 {
  return SmithG1(wi, n, roughness) * SmithG1(wo, n, roughness);
}

fn BRDF(
  wi: vec3f,
  wo: vec3f,
  n: vec3f,
  albedo: vec3f,
  roughness: f32,
  metalness: f32
) -> vec3f {
  let diffuseColor = albedo * (1.0 - metalness);
  let specularColor = mix(vec3f(0.08), albedo, metalness);
  let alpha = roughness * roughness; // To approach a linear behavior
  let NdotL = max(0.0, dot(n, wi));
  let NdotV = max(0.0, dot(n, wo));

  if (NdotL <= 0.0) { // Not in the reflection hemisphere
    return vec3f (0.0);
  }

  let wh = normalize(wi + wo);
  let NdotH = max(0.0, dot(n, wh));
  let VdotH = max(0.0, dot(wo, wh));

  // Normal distribution function, sometimes coined "GGX"
  let D = TrowbridgeReitzNDF(wh, n, alpha);

  // Schlick approximation to the Fresnel term
  let F = SchlickFresnel(wi, wh, specularColor);

  // Masking-Shadowing term
  let G = SmithGGX(wi, wo, n, alpha);

  // "A" reasonable combination of diffuse and specular responses
  let fd = diffuseColor * (vec3f(1.0) - specularColor) / PI;
  let fs = F * D * G / (4.0);

  return (fd + fs);
}

// Operate in view space i.e., in the local frame of the camera
fn lightShade(position: vec3f, normal: vec3f, materialIndex: u32, lightSourceIndex: u32, wo: vec3f) -> vec3f {
  let light = lightSources[lightSourceIndex];
  let cam = scene.camera;
  let viewLightPos = cam.viewMat * vec4f(light.position, 1.0);
  let viewLightTarget = cam.viewMat * vec4f(light.spot, 1.0);
  let viewLightDir = normalize(viewLightTarget.xyz - viewLightPos.xyz);
  var wi = viewLightPos.xyz - position;
  let di = length(wi);
  wi = normalize(wi);
  var spotConeDecay = dot(-wi, viewLightDir) - light.angle;
  if (spotConeDecay <= 0.0) {
    return vec3f(0.0); // Out of spot light cone
  }
  let att = attenuation(di, spotConeDecay);
  let ir = light.color * light.intensity * att;
  let m = materials[materialIndex];
  let fr = BRDF(wi, wo, normal, m.albedo, m.roughness, m.metalness);
  let colorResponse = ir * fr * max (0.0, dot (wi, normal));
  return colorResponse;
}

fn computeRadiance(position: vec3f, normal: vec3f, materialIndex: u32, wo: vec3f) -> vec3f {
  var colorResponse = vec3f (0.0);
  let startIdx = u32(scene.lightStartIndex);
  let endIdx = u32(scene.lightEndIndex);
  for (var lightSourceIndex = startIdx; lightSourceIndex < endIdx; lightSourceIndex++) {
    colorResponse += lightShade(position, normal, materialIndex, lightSourceIndex, wo);
  }
  return colorResponse;
}

// ─── Lightcuts: per-pixel tree traversal ─────────────────────────────────

const LC_MAX_CUT = 32u;   // hard upper bound on cut array size

/// Shade a single lightcut cluster representative at a world-space shading point
/// using the same PBR + shadow-ray logic as the normal light loop.
fn shadeLightcutNode(
  node: LightcutGPUNode,
  worldPos: vec3f,
  worldNormal: vec3f,
  viewPos: vec3f,
  viewNormal: vec3f,
  materialIndex: u32,
  wo: vec3f
) -> vec3f {
  let L = node.position - worldPos;
  let dist = length(L);
  let dir = normalize(L);

  // Shadow ray (same as normal RT path)
  var shadowRay: Ray;
  shadowRay.origin = worldPos;
  shadowRay.direction = dir;
  var shadowHit: Hit;
  if (rayTrace(shadowRay, dist - 0.01, true, &shadowHit)) {
    return vec3f(0.0);
  }

  // PBR shading
  let att = 10.0 / (dist * dist + 0.1);
  let radiance = node.color * node.intensity * att;
  let cam = scene.camera;
  let viewLightPos = (cam.viewMat * vec4f(node.position, 1.0)).xyz;
  let wi = normalize(viewLightPos - viewPos);
  let m = materials[materialIndex];
  let fr = BRDF(wi, wo, viewNormal, m.albedo, m.roughness, m.metalness);
  return radiance * fr * max(0.0, dot(wi, viewNormal));
}

/// Error bound for a lightcut node at a shading point.
/// Captures both intensity importance and geometric spread of the cluster.
fn lightcutErrorBound(node: LightcutGPUNode, worldPos: vec3f) -> f32 {
  let L = node.position - worldPos;
  let distSq = dot(L, L) + 0.001;
  let dist = sqrt(distSq);
  let diag = length(node.aabbMax - node.aabbMin);
  return node.intensity / distSq * diag / dist;
}

/// Per-pixel greedy lightcut traversal.
/// Starts with the root; iteratively splits the node with the highest error
/// bound until the cut reaches maxCutSize or all entries are leaves.
fn computeRadianceLightcuts(
  worldPos: vec3f,
  worldNormal: vec3f,
  viewPos: vec3f,
  viewNormal: vec3f,
  materialIndex: u32,
  wo: vec3f
) -> vec3f {
  let nodeCount = debugParams.lightcutNodeCount;
  let maxCut = min(debugParams.maxCutSize, LC_MAX_CUT);

  // Cut stored as fixed-size arrays (WGSL requires compile-time size)
  var cutIdx:   array<u32, 32>;   // node indices
  var cutError: array<f32, 32>;   // cached error bounds
  var cutSize: u32 = 0u;

  if (nodeCount == 0u) {
    return vec3f(0.0);
  }

  // Seed with root node (index 0)
  cutIdx[0]   = 0u;
  cutError[0] = lightcutErrorBound(lightcutTree[0], worldPos);
  cutSize = 1u;

  // Greedy refinement
  for (var iter = 0u; iter < 256u; iter++) {
    if (cutSize >= maxCut) { break; }

    // Find the cut entry with the highest error bound
    var worstIdx = 0u;
    var worstErr = -1.0;
    for (var c = 0u; c < cutSize; c++) {
      if (cutError[c] > worstErr) {
        worstErr = cutError[c];
        worstIdx = c;
      }
    }

    // If worst entry is a leaf, we can't split further — done
    let nodeId = cutIdx[worstIdx];
    let node = lightcutTree[nodeId];
    let leftI  = i32(node.leftChild);
    let rightI = i32(node.rightChild);
    if (leftI < 0 && rightI < 0) { break; }

    // Replace worst entry with left child
    if (leftI >= 0 && u32(leftI) < nodeCount) {
      cutIdx[worstIdx]   = u32(leftI);
      cutError[worstIdx] = lightcutErrorBound(lightcutTree[u32(leftI)], worldPos);
    } else {
      // No left child — keep as leaf with zero error
      cutError[worstIdx] = 0.0;
    }

    // Append right child
    if (rightI >= 0 && u32(rightI) < nodeCount && cutSize < LC_MAX_CUT) {
      cutIdx[cutSize]   = u32(rightI);
      cutError[cutSize] = lightcutErrorBound(lightcutTree[u32(rightI)], worldPos);
      cutSize += 1u;
    }
  }

  // Shade each representative in the final cut
  var color = vec3f(0.0);
  for (var c = 0u; c < cutSize; c++) {
    let n = lightcutTree[cutIdx[c]];
    color += shadeLightcutNode(n, worldPos, worldNormal, viewPos, viewNormal, materialIndex, wo);
  }
  return color;
}


//-----------------------------------------------------------------------
// Rasterization shaders
//-----------------------------------------------------------------------

@vertex
  fn rasterVertexMain(input: RasterVertexInput) -> RasterVertexOutput {
    let cam = scene.camera;
    var mesh = meshes[input.meshIndex];
    let vID = input.vertexIndex;

    // Recovering triangle and vertex from the draw index
    let triIndex = vID / 3u;
    let triVertIndex = vID % 3u;
    let triangle = getTriangle(mesh.triOffset + triIndex);
    let vertIndex = mesh.posOffset + triangle[triVertIndex];

    var output: RasterVertexOutput;
    let p = cam.viewMat * cam.modelMat * vec4f(getVertPos(vertIndex), 1.0);
    output.builtInPos = cam.projMat * p; // Fires rasterization
    output.position = p.xyz;
    let n = cam.transInvViewMat * vec4f(getVertNormal(vertIndex), 1.0);
    output.normal = normalize(n.xyz);
    output.materialIndex = mesh.materialIndex;
    return output;
  }

@fragment
  fn rasterFragmentMain(input: RasterVertexOutput) -> @location(0) vec4f {
    // Debug mode 2: raw albedo — output material color directly, no lighting.
    let m = materials[input.materialIndex];
    if (debugParams.mode == 2u) {
      return vec4f(m.albedo, 1.0);
    }
    let position = input.position;
    let normal = normalize(input.normal);
    let wo = normalize(-position);
    let colorResponse = computeRadiance(position, normal, input.materialIndex, wo);
    return vec4f(colorResponse, 1.0);
  }

//-----------------------------------------------------------------------
// Ray tracing pipeline shaders
//-----------------------------------------------------------------------

struct RayVertexInput {
  @builtin(vertex_index) vertexIndex: u32
};

struct RayVertexOutput {
  @builtin(position) pos : vec4f,
};

struct RayFragmentInput {
  @builtin(position) fragPos : vec4f,
};

struct Ray {
  origin: vec3f,
  direction: vec3f,
};

struct Hit{
  meshIndex: u32,
  triIndex: u32,
  u: f32, // barycentric coordinates of the intersection
  v: f32,
  t: f32, // distance to ray's origin of the intersection
};

fn interpolate(x0: vec3f, x1: vec3f, x2: vec3f, uvw: vec3f) -> vec3f {
  return uvw.z * x0 + uvw.x * x1 + uvw.y * x2;
}

fn rayAt(uv: vec2f, camera : Camera) -> Ray {
  var ray : Ray;
  let viewRight = normalize(camera.invViewMat[0].xyz);
  let viewUp = normalize(camera.invViewMat[1].xyz);
  let viewDir = -normalize(camera.invViewMat[2].xyz);
  let eye = camera.invViewMat[3].xyz;
  let w = 2.0 * tan(0.5 * camera.fov);
  ray.origin = eye;
  ray.direction = normalize(viewDir + ((uv.x - 0.5) * camera.aspectRatio * w) * viewRight + ((uv.y) - 0.5) * w * viewUp);
  return ray;
}

fn intersectTriangle(
  ray: Ray,
  p0: vec3f,
  p1: vec3f,
  p2: vec3f,
  backFaceCulling: bool,
  tMin: f32,
  tMax: f32,
  hit: ptr<function, Hit>
) -> bool {
  const EPSILON = 1e-6;
  let e1 = p1 - p0;
  let e2 = p2 - p0;
  let dxe2 = cross(ray.direction, e2);
  let det = dot(e1, dxe2);
  if ((backFaceCulling && det < EPSILON) || (!backFaceCulling && abs(det) < EPSILON)) {
    return false;
  }
  let invDet = 1.0 / det;
  let op0 = ray.origin - p0;
  (*hit).u = dot(op0, dxe2) * invDet;
  if ((*hit).u < 0.0 || (*hit).u > 1.0) {
    return false;
  }
  let op0xe1 = cross(op0, e1);
  (*hit).t = dot(e2, op0xe1) * invDet;
  if ((*hit).t < tMin || (*hit).t > tMax) {
    return false;
  }
  (*hit).v = dot(ray.direction, op0xe1) * invDet;
  if ((*hit).v >= 0.0 && (*hit).u + (*hit).v <= 1.0) {
    return true;
  }
  return false;
}

fn rayTrace(
  ray: Ray,
  maxDistance: f32, // Ignore intersections found further away
  anyHit: bool, // Return as soon as an intersection is found if true
  hit: ptr<function, Hit> // Filled only if an intersection is found and anyHit is false
) -> bool {
  var intersectionFound = false;
  let numOfMeshes = u32(scene.numOfMeshes);
  for (var meshIndex = 0u; meshIndex < numOfMeshes; meshIndex++) {
    let mesh = meshes[meshIndex];
    for (var triIndex = 0u; triIndex < mesh.numOfTriangles; triIndex++) {
      let triangle = getTriangle(mesh.triOffset + triIndex);
      var triHit: Hit;
      triHit.meshIndex = meshIndex;
      triHit.triIndex = triIndex;
      let p0 = getVertPos(mesh.posOffset + triangle.x);
      let p1 = getVertPos(mesh.posOffset + triangle.y);
      let p2 = getVertPos(mesh.posOffset + triangle.z);
      if (intersectTriangle(ray, p0, p1, p2, true, 0.0, maxDistance, &triHit) == true) {
        if (!intersectionFound || (intersectionFound && triHit.t < (*hit).t)) {
          if (anyHit == true) {
            return true;
          }
          *hit = triHit;
          intersectionFound = true;
        }
      }
    }
  }
  return intersectionFound;
}

// MODES:
// 0 = Normal PBR
// 1 = Visible Light Count (Heatmap)
// 2 = Raw Albedo (Texture Color)
// 3 = World Normals
fn shadeRT(hit: Hit) -> vec4f {
  // Debug mode is driven by a small uniform set from the UI.
  let debugMode = debugParams.mode;

  let mesh = meshes[hit.meshIndex];
  let tri = getTriangle(mesh.triOffset + hit.triIndex);
  let uvw = vec3f(hit.u, hit.v, 1.0 - hit.u - hit.v);

  // Reconstruct World Data
  let p0 = getVertPos(mesh.posOffset + tri.x);
  let p1 = getVertPos(mesh.posOffset + tri.y);
  let p2 = getVertPos(mesh.posOffset + tri.z);
  let worldPos = uvw.z * p0 + uvw.x * p1 + uvw.y * p2;
  
  let n0 = getVertNormal(mesh.posOffset + tri.x);
  let n1 = getVertNormal(mesh.posOffset + tri.y);
  let n2 = getVertNormal(mesh.posOffset + tri.z);
  let worldNormal = normalize(uvw.z * n0 + uvw.x * n1 + uvw.y * n2);

  // DEBUG 3: Show Normals
  if (debugMode == 3u) {
      return vec4f(worldNormal * 0.5 + 0.5, 1.0);
  }

  // DEBUG 2: Show Albedo
  let m = materials[mesh.materialIndex];
  if (debugMode == 2u) {
      return vec4f(m.albedo, 1.0);
  }

  let cam = scene.camera;
  let viewPos = (cam.viewMat * cam.modelMat * vec4f(worldPos, 1.0)).xyz;
  let viewNormal = normalize((cam.transInvViewMat * vec4f(worldNormal, 1.0)).xyz);
  let wo = normalize(-viewPos);

  var outputColor = vec3f(0.0);
  var visibleCount = 0.0;

  // ── Lightcuts path: per-pixel tree traversal ──────────────────────
  if (debugParams.lightcutNodeCount > 0u) {
    outputColor = computeRadianceLightcuts(
      worldPos, worldNormal, viewPos, viewNormal, mesh.materialIndex, wo
    );
    return vec4f(outputColor, 1.0);
  }

  // ── Normal light loop ─────────────────────────────────────────────
  // Use light range from uniforms (supports accumulation passes)
  let startIdx = u32(scene.lightStartIndex);
  let endIdx = u32(scene.lightEndIndex);

  for (var i = startIdx; i < endIdx; i++) {
    let l = lightSources[i];
    let L = l.position - worldPos;
    let dist = length(L);
    let dir = normalize(L);
    
    var visible = 1.0;
    if (bool(l.rayTracedShadows)) {
       var shadowRay: Ray;
       // Bias along the light direction instead of the surface normal.
       // This avoids bogus self-shadowing when normals are flipped or noisy,
       // which was making the floor and RAM top report zero visible lights.
       shadowRay.origin = worldPos;
       shadowRay.direction = dir;
       
       var shadowHit: Hit;
       if (rayTrace(shadowRay, dist - 0.01, true, &shadowHit)) {
         visible = 0.0;
       }
    }

    if (visible > 0.0) {
       // Accumulate for heatmap
       visibleCount += 1.0;

       // Normal shading math
       if (debugMode == 0u) {
           let att = 10.0 / (dist * dist + 0.1); 
           let radiance = l.color * l.intensity * att;
           let wi = normalize((cam.viewMat * vec4f(l.position, 1.0)).xyz - viewPos);
           let fr = BRDF(wi, wo, viewNormal, m.albedo, m.roughness, m.metalness);
           outputColor += radiance * fr * max(0.0, dot(wi, viewNormal));
       }
    }
  }

 // DEBUG 1: Heatmap (White = 50 lights visible, Black = 0)
 if (debugMode == 1u) {
  // Keep some base visibility so geometry is never pure black.
  let base = 0.2;
  let heat = clamp(visibleCount / 100.0, 0.0, 1.0);
  let v = base + (1.0 - base) * heat;
  return vec4f(vec3f(v), 1.0);
}

  // DEBUG 0: Normal PBR
  return vec4f(outputColor, 1.0);
}

@vertex
  fn rayVertexMain(input: RayVertexInput) -> RayVertexOutput {
    var output: RayVertexOutput;
    const screenPos = array<vec2<f32>, 6>(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),
    );
    output.pos = vec4f(screenPos[input.vertexIndex], 0.0, 1.0);
    return output;
  }

@fragment
  fn rayFragmentMain(input: RayFragmentInput) -> @location(0) vec4f {
    const MAX_DISTANCE = 1e8;
    // Use actual canvas resolution from the uniform instead of hardcoded values.
    let coord = vec2f(
      input.fragPos.x / scene.screenWidth,
      1.0 - input.fragPos.y / scene.screenHeight
    );
    let ray = rayAt(coord, scene.camera);
    var colorResponse = vec4f(0.0, 0.0, 0.0, 1.0);
    var hit: Hit;
    if (rayTrace(ray, MAX_DISTANCE, false, &hit) == true) {
      colorResponse = shadeRT(hit);
    }
    return colorResponse;
  }

// Blit offscreen texture to swap chain (fullscreen quad = 2 triangles, 6 vertices). Uses group 0.
@group(0) @binding(0) var blitTex: texture_2d<f32>;
@group(0) @binding(1) var blitSampler: sampler;

struct BlitVertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn blitVertexMain(@builtin(vertex_index) vi: u32) -> BlitVertexOutput {
  var out: BlitVertexOutput;
  // Fullscreen quad: 6 vertices = two triangles (0,1,2) and (2,1,3). Y up in NDC.
  let corners = array<vec2f, 4>(
    vec2f(-1.0, 1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, -1.0)
  );
  let uvs = array<vec2f, 4>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 1.0)
  );
  let idx = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
  let i = idx[vi];
  out.position = vec4f(corners[i], 0.0, 1.0);
  out.uv = uvs[i];
  return out;
}

@fragment
fn blitFragmentMain(input: BlitVertexOutput) -> @location(0) vec4f {
  return textureSample(blitTex, blitSampler, input.uv);
}

// -----------------------------------------------------------------------
// Accumulation blit shaders
// -----------------------------------------------------------------------

// Accumulation additive pass: reads from the new-pass texture and adds to the
// accumulation texture. Uses alpha blending (ONE, ONE) on the GPU side, so
// this shader simply outputs the source color.
@group(0) @binding(0) var accumSrcTex: texture_2d<f32>;
@group(0) @binding(1) var accumSrcSampler: sampler;

@vertex
fn accumBlitVertexMain(@builtin(vertex_index) vi: u32) -> BlitVertexOutput {
  var out: BlitVertexOutput;
  let corners = array<vec2f, 4>(
    vec2f(-1.0, 1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, -1.0)
  );
  let uvs = array<vec2f, 4>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 1.0)
  );
  let idx = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
  let i = idx[vi];
  out.position = vec4f(corners[i], 0.0, 1.0);
  out.uv = uvs[i];
  return out;
}

@fragment
fn accumBlitFragmentMain(input: BlitVertexOutput) -> @location(0) vec4f {
  return textureSample(accumSrcTex, accumSrcSampler, input.uv);
}

// Final blit: divide accumulated color by pass count.
// passCount is stored as a uniform.
struct AccumFinalParams {
  invPassCount: f32,
  _pad: vec3<f32>,
}

@group(0) @binding(0) var finalAccumTex: texture_2d<f32>;
@group(0) @binding(1) var finalAccumSampler: sampler;
@group(0) @binding(2) var<uniform> accumFinalParams: AccumFinalParams;

@vertex
fn accumFinalVertexMain(@builtin(vertex_index) vi: u32) -> BlitVertexOutput {
  var out: BlitVertexOutput;
  let corners = array<vec2f, 4>(
    vec2f(-1.0, 1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, -1.0)
  );
  let uvs = array<vec2f, 4>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(1.0, 1.0)
  );
  let idx = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
  let i = idx[vi];
  out.position = vec4f(corners[i], 0.0, 1.0);
  out.uv = uvs[i];
  return out;
}

@fragment
fn accumFinalFragmentMain(input: BlitVertexOutput) -> @location(0) vec4f {
  let color = textureSample(finalAccumTex, finalAccumSampler, input.uv);
  return vec4f(color.rgb * accumFinalParams.invPassCount, 1.0);
}
