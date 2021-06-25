// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <light/src/vector.hpp>
#include <light/src/light.hpp>
#include <light/src/sdf.hpp>
#include <light/src/ArrayStack.hpp>

#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>

using namespace poplar;

/// Codelet which generates all outgoing (primary) camera rays for
/// a tile. Anti-aliasing noise is added to the rays using random
/// numbers that were generated external to this codelet. Because
/// they are close to normalised the camera rays can be safely
/// stored at half precision which reduces memory requirements.
class GenerateCameraRays : public Vertex {

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

  bool compute() {
    // Make a lambda to consume random numbers from the buffer:
    std::size_t randomIndex = 0;
    auto rng = [&] () {
      const half value = antiAliasNoise[randomIndex];
      randomIndex += 1;
      if (randomIndex == antiAliasNoise.size()) {
        randomIndex = 0;
      }
      return value;
    };

    using Vec = light::Vector;

    unsigned k = 0;
    for (unsigned r = startRow; r < endRow; ++r) {
      for (unsigned c = startCol; c < endCol; ++c) {
        // The camera rays returned from pixelToRay are in world space
        // so the anti-aliasing noise is not in units of pixels (which
        // would be more sensible):
        const Vec cam = light::pixelToRay(c, r, imageWidth, imageHeight);
        rays[k]     = cam.x + (float)(*antiAliasScale * rng());
        rays[k + 1] = cam.y + (float)(*antiAliasScale * rng());
        rays[k + 2] = cam.z;
        k += 3;
      }
    }
    return true;
  }
};

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
    using Vec = light::Vector;
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
    auto raysSize = (cameraRays.size() / 3) * 3;
    for (auto r = 0u; r < raysSize; r += 3) {
      // Unpack the camera rays which are stored as sequential x, y, z coords:
      Vec rayDir((float)cameraRays[r], (float)cameraRays[r+1], (float)cameraRays[r+2]);
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
        frameBuffer[r]     += total.x;
        frameBuffer[r + 1] += total.y;
        frameBuffer[r + 2] += total.z;
      }

    } // end loop over camera rays

    return true;
  }
};

/// The only supported frame buffer types are float and half:
template class RayTraceKernel<float>;
template class RayTraceKernel<half>;
