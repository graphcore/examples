// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <light/src/vector.hpp>
#include <light/src/light.hpp>
#include <light/src/sdf.hpp>
#include <light/src/ArrayStack.hpp>

#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

// Because intrinsic/vectorised code can not be used with CPU
// or IpuModel targets we need to guard IPU optimised parts of
// the code so we can still support those:
#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#endif // __IPU__

using namespace poplar;
using Vec = light::Vector;

/// Codelet which generates all outgoing (primary) camera rays for
/// a tile. Anti-aliasing noise is added to the rays using random
/// numbers that were generated external to this codelet. Because
/// they are close to normalised the camera rays can be safely
/// stored at half precision which reduces memory requirements.
///
/// This is a multi-vertex that decides how to distribute work
/// over the hardware worker threads inside the compute method
/// itself.
class GenerateCameraRays : public MultiVertex {

public:
  Input<Vector<half>> antiAliasNoise;
  Output<Vector<half>> rays;
  Input<unsigned> startRow;
  Input<unsigned> startCol;
  Input<unsigned> endRow;
  Input<unsigned> endCol;
  Input<unsigned> imageWidth;
  Input<unsigned> imageHeight;
  Input<half> antiAliasScale;

  bool compute(unsigned workerId) {
    const auto workerCount = numWorkers();

    // Make a lambda to consume random numbers from the buffer.
    // Each worker consumes random numbers using its ID as an
    // offset into the noise buffer:
    std::size_t randomIndex = workerId;
    auto rng = [&] () {
      const half value = antiAliasNoise[randomIndex];
      randomIndex += workerCount;
      return value;
    };

    // Each worker will process one sixth of the rows so
    // we simply offset the start rows based on worker ID:
    const auto workerStart = startRow + workerId;
    // Similary, we must offset the output index to start of
    // appropriate row based on worker ID:
    const auto numRayComponentsInRow = (endCol - startCol) * 2;
    unsigned k = workerId * numRayComponentsInRow;
    // Outer loop is parallelised over the worker threads:
    for (unsigned r = workerStart; r < endRow; r += workerCount) {
      for (unsigned c = startCol; c < endCol; ++c) {
        // The camera rays returned from pixelToRay are in world space
        // so the anti-aliasing noise is not in units of pixels (which
        // would be more sensible):
        const Vec cam = light::pixelToRay(c, r, imageWidth, imageHeight);
        rays[k]     = cam.x + (float)(*antiAliasScale * rng());
        rays[k + 1] = cam.y + (float)(*antiAliasScale * rng());
        k += 2;
      }
      // Output index is now at start of next row but we need to jump
      // it to the start of the next row for this worker:
      k += (workerCount - 1) * numRayComponentsInRow;
    }
    return true;
  }
};

#ifdef __IPU__
/// This version of the ray-gen vertex uses intrinsics from ipu_vector_math
/// and ipu_memory_intrinsics to optimise the ray generation using half2
/// SIMD vectors and special 64-bit load/store instructions that load
/// larger chunks of data and simultaneously increment the pointer.
/// It also inlines the calculation from light::pixelToRay in order that
/// it can use SIMD ops and have constant values moved out of the inner loop.
class GenerateCameraRaysSIMD : public MultiVertex {

public:
  // In order to use special load/store instructions we need to align
  // the data appropriately. Also, we know exactly how many rays we
  // are generating so we can use a compact pointers:
  Input<Vector<half, poplar::VectorLayout::COMPACT_PTR, 8>> antiAliasNoise;
  Output<Vector<half, poplar::VectorLayout::COMPACT_PTR, 8>> rays;
  Input<unsigned> startRow;
  Input<unsigned> startCol;
  Input<unsigned> endRow;
  Input<unsigned> endCol;
  Input<unsigned> imageWidth;
  Input<unsigned> imageHeight;
  Input<half> antiAliasScale;

  bool compute(unsigned workerId) {
    const auto workerCount = numWorkers();

    // Setup pointers for accessing the anti-aliasing random samples:
    const half4* start = (half4*)&antiAliasNoise[4 * workerId];
    const half4 aaScale {*antiAliasScale, *antiAliasScale, *antiAliasScale, *antiAliasScale};
    auto randomPtr = start;

    // Each worker will process one sixth of the rows so
    // we simply offset the start rows based on worker ID:
    const auto workerStart = startRow + workerId;
    // Similary, we must offset the output pointer to start of
    // appropriate row based on worker ID:
    const auto numRayComponentsInRow = (endCol - startCol) << 1; // 2 direction components per pixel
    half4* outPtr = (half4*)&rays[workerId * numRayComponentsInRow];
    // Note: outPtr must be 8-byte aligned for every worker. Application guarantees (endCol - startCol)
    // is a multiple of 2 and numRayComponentsInRow is 2*(endCol - startCol) so
    // numRayComponentsInRow = 2*2*n for some n. This means workerId * numRayComponentsInRow
    // is workerId*4*n so always of multiple of 4*sizeof(half) = 8 bytes. In the loop outPtr is
    // always incremented as a half4* so always remains 8-byte aligned.

    // We can pre-compute everything related to field of view:
    const half2 wh {(half)imageWidth, (half)imageHeight};
    const half2 inv_wh {(half)(1.f/imageWidth), (half)(1.f/imageHeight)};
    const half2 aspect {1.f, (*imageHeight)/(half)(*imageWidth)};
    constexpr half piby4 = light::Pi / 4.f;
    const half2 fovxy = aspect * half2{piby4, piby4};
    auto tanfovxy = ipu::tan(fovxy);
    tanfovxy = tanfovxy * half2{1.f, -1.f};
    const auto proj = inv_wh * tanfovxy;
    // Casting from unsigned to half is expensive so we try to move
    // casts to outer loops and then increment values in the inner
    // loop using floating-point arithmetic but note this limits the
    // image width to 2048: the max integer exactly representable in fp16.
    // (In reality we should be using normalised camera coordinates here
    // which would avoid this issue).
    const half colInit = (half)startCol;
    const half2 unrollInc = {1.f, 0.f};

    // Outer loop is parallelised over the worker threads (as per the non-optimised version):
    for (unsigned r = workerStart; r < endRow; r += workerCount) {
      // The inner loop uses 2-way SIMD to compute the x and y components
      // for one ray simultaneously then we unroll the loop 2x so that we can
      // compute two camera rays per iteration (this allows us to read/write
      // 8-bytes with each load store instruction):
      half2 xy {colInit, (half)r};
      for (unsigned c = startCol; c < endCol; c += 2) {
        // Load 4 noise components at once:
        const auto noise = aaScale * ipu::load_postinc(&randomPtr, workerCount);

        const auto cam1 = (2.f * xy - wh) * proj;
        xy += unrollInc;
        const auto cam2 = (2.f * xy - wh) * proj;
        xy += unrollInc;
        const auto dir1 = cam1 + half2{noise[0], noise[1]};
        const auto dir2 = cam2 + half2{noise[2], noise[3]};

        // Write 2 generated camera rays (4 components) at once:
        ipu::store_postinc(&outPtr, half4{dir1[0], dir1[1], dir2[0], dir2[1]}, 1);
      }
      // Output pointer is now at start of next row but we need to jump
      // it to the start of the next row for this worker:
      outPtr += (workerCount - 1) * (numRayComponentsInRow >> 2); // Divide by 4 because its a half4 pointer
    }
    return true;
  }
};
#endif // __IPU__

/// Codelet which performs ray tracing for the tile. It knows
/// nothing about the image geometry - it just receives a flat
/// buffer of primary rays as input and stores the result of path
/// tracing for that ray in the corresponding position in the output
/// frame buffer. This codelet also receives as input a buffer of
/// uniform noise to use for all MC sampling operations during path
/// tracing.
///
/// The codelet is templated on the framebuffer type. If using half
/// precision it is the application's responsibility to avoid framebuffer
/// saturation: this avoids extra logic and computation in the codelet.
///
/// For now the scene is hard coded onto the stack of the compute()
/// function but it could be passed in as tensor data (with some extra
/// overhead of manipulating or unpacking the data structure).
template <typename FrameBufferType>
class RayTraceKernel : public Vertex {

public:
  Input<Vector<half>> cameraRays;
  Input<Vector<half>> uniform_0_1;
  InOut<Vector<FrameBufferType>> frameBuffer;
  Input<half> refractiveIndex;
  Input<half> stopProb;
  Input<unsigned short> rouletteDepth;

  bool compute() {
    const Vec zero(0.f, 0.f, 0.f);
    const Vec one(1.f, 1.f, 1.f);
    const auto X = Vec(1.f, 0.f, 0.f);
    const auto Y = Vec(0.f, 1.f, 0.f);
    const auto Z = Vec(0.f, 0.f, 1.f);

    // The scene is currently hard coded here:
    light::Sphere spheres[6] = {
      light::Sphere(Vec(-0.75f, -1.45f, -4.4f), 1.05f),
      light::Sphere(Vec(2.0f, -2.05f, -3.7f), 0.5f),
      light::Sphere(Vec(-1.75f, -1.95f, -3.1f), 0.6f),
      light::Sphere(Vec(-1.12f, -2.3f, -3.5f), 0.2f),
      light::Sphere(Vec(-0.28f, -2.34f, -3.f), 0.2f),
      light::Sphere(Vec(.58f, -2.39f, -2.6f), 0.2f)
    };
    light::Plane planes[6] = {
      light::Plane(Y, 2.5f),
      light::Plane(Z, 5.5f),
      light::Plane(X, 2.75f),
      light::Plane(-X, 2.75f),
      light::Plane(-Y, 3.f),
      light::Plane(-Z, .5f)
    };
    light::Disc discs[1] = {
      light::Disc(-Y, Vec(0.f, 2.9999f, -4.f), .7f)
    };

    const Vec lightW(8000.f, 8000.f, 8000.f);
    const Vec lightR(1000.f, 367.f, 367.f);
    const Vec lightG(467.f, 1000.f, 434.f);
    const Vec lightB(500.f, 600.f, 1000.f);
    const float colourGain = 15.f;
    const auto sphereColour = Vec(1.f, .89f, .55f) * colourGain;
    const auto wallColour1 = Vec(.98f, .76f, .66f) * colourGain;
    const auto wallColour2 = Vec(.93f, .43f, .48f) * colourGain;
    const auto wallColour3 = Vec(.27f, .31f, .38f) * colourGain;
    constexpr auto specular = light::Material::Type::specular;
    constexpr auto refractive = light::Material::Type::refractive;
    constexpr auto diffuse = light::Material::Type::diffuse;
    constexpr auto numObjects = std::size(spheres) + std::size(planes) + std::size(discs);
    light::Scene<numObjects> scene({
        light::Object(&spheres[0], Vec(4.f, 8.f, 4.f), zero, specular),
        light::Object(&spheres[1], Vec(10.f, 10.f, 1.f), zero, refractive), // Glass sphere
        light::Object(&spheres[2], sphereColour, zero, diffuse), // Diffuse sphere
        light::Object(&spheres[3], zero, lightB, specular), // Small light red
        light::Object(&spheres[4], zero, lightG, specular), // Small light green
        light::Object(&spheres[5], zero, lightR, specular), // Small light blue

        light::Object(&planes[0], wallColour1, zero, diffuse), // Bottom plane
        light::Object(&planes[1], wallColour1, zero, diffuse), // Back plane
        light::Object(&planes[2], wallColour2, zero, diffuse), // Left plane
        light::Object(&planes[3], wallColour3, zero, diffuse), // Right plane
        light::Object(&planes[4], wallColour1, zero, diffuse), // Ceiling plane
        light::Object(&planes[5], wallColour1, zero, diffuse), // Front plane

        light::Object(&discs[0], Vec(0,0,0), lightW, diffuse), // Ceiling light
    });

    constexpr std::size_t maxContributions = 20; // IPU needs a hard upper limit on number of rays per sample.
    ArrayStack<light::Contribution, maxContributions> contributions;

    // Make a lambda to consume random numbers from the buffer:
    std::size_t randomIndex = 0;
    auto rng = [&] () {
      const half value = uniform_0_1[randomIndex];
      randomIndex += 1;
      if (randomIndex == uniform_0_1.size()) {
        randomIndex = 0;
      }
      return value;
    };

    // Loop over the camera rays:
    const auto raysSize = (cameraRays.size() >> 1) << 1;
    for (auto r = 0u, f = 0u; r < raysSize; r += 2, f += 3) {
      // Unpack the camera ray directions which are stored as a
      // sequence of x, y coords with implicit z-direction of -1:
      Vec rayDir((float)cameraRays[r], (float)cameraRays[r+1], (float)-1.f);
      light::Ray ray(zero, rayDir);
      std::uint32_t depth = 0;
      bool hitEmitter = false;
      contributions.clear();

      // Trace rays through the scene, recording contribution values and type:
      while (!contributions.full()) {
        // Russian roulette ray termination:
        float rrFactor = 1.f;
        if (depth >= rouletteDepth) {
          bool stop;
          std::tie(stop, rrFactor) = light::rouletteWeight((float)rng(), (float)*stopProb);
          if (stop) { break; }
        }

        // Intersect the ray with the whole scene, advancing it to the hit point:
        const auto intersection = scene.intersect(ray);
        if (!intersection) { break; }

        if (intersection.material->emissive) {
          contributions.push_back({intersection.material->emission, rrFactor, light::Contribution::Type::EMIT});
          hitEmitter = true;
        }

        if (contributions.full()) {
          break;
        }

        // Sample a new ray based on material type:
        if (intersection.material->type == diffuse) {
          const float sample1 = (float)rng();
          const float sample2 = (float)rng();
          const auto result =
            light::diffuse(ray, intersection.normal, intersection, rrFactor, sample1, sample2);
          contributions.push_back(result);
        } else if (intersection.material->type == specular) {
          light::reflect(ray, intersection.normal);
          contributions.push_back({zero, rrFactor, light::Contribution::Type::SPECULAR});
        } else if (intersection.material->type == refractive) {
          const float ri = (float)*refractiveIndex;
          light::refract(ray, intersection.normal, ri, (float)rng());
          contributions.push_back({zero, 1.15f * rrFactor, light::Contribution::Type::REFRACT});
        }

        depth += 1;
      }

      // Rays will only have a non-zero contribution if they hit a light source at some point:
      if (hitEmitter) {
        Vec total = zero;
        while (!contributions.empty()) {
          auto c = contributions.back();
          contributions.pop_back();
          switch (c.type) {
          case light::Contribution::Type::DIFFUSE:
            // Diffuse materials modulate the colour being carried back
            // along the light path (scaled by the importance weight):
            total = total.cwiseProduct(c.clr) * c.weight;
            break;
            // Emitters add their colour to the colour being carried back
            // along the light path (scaled by the importance weight):
          case light::Contribution::Type::EMIT:
            total += c.clr * c.weight;
            break;
            // Specular reflections/refractions have no colour contribution but
            // their importance sampling weights must still be applied:
          case light::Contribution::Type::SPECULAR:
          case light::Contribution::Type::REFRACT:
            total *= c.weight;
            break;
          // Sometimes it is useful to be able to skip certain
          // contributions when debugging.
          case light::Contribution::Type::SKIP:
          default:
            break;
          }
        }

        // Store the resulting colour contribution (x ,y, z -> r, g, b):
        frameBuffer[f]     += total.x;
        frameBuffer[f + 1] += total.y;
        frameBuffer[f + 2] += total.z;
      }

    } // end loop over camera rays

    return true;
  }
};

/// The only supported frame buffer types are float and half:
template class RayTraceKernel<float>;
template class RayTraceKernel<half>;
