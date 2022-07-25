// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <cmath>
#include <print.h>

using namespace poplar;

static constexpr auto SPAN = poplar::VectorLayout::SPAN;

template <class T>
class ROIAlignForwardIPU : public Vertex {
public:
  Input<Vector<T, SPAN, 4>> bottom_data;
  Input<Vector<T, SPAN, 4>> bottom_rois;
  Output<Vector<T, SPAN, 4>> top_data;
  T spatial_scale;
  int height;
  int width;
  int aligned_height;
  int aligned_width;

  bool compute() {
    auto roi_start_w = bottom_rois[0] * spatial_scale-0.5f;
    auto roi_start_h = bottom_rois[1] * spatial_scale-0.5f;
    auto roi_end_w   = bottom_rois[2] * spatial_scale-0.5f;
    auto roi_end_h   = bottom_rois[3] * spatial_scale-0.5f;

    // Force malformed ROI to be 1x1
    auto roi_width  = fmaxf(roi_end_w - roi_start_w, 0.0f);
    auto roi_height = fmaxf(roi_end_h - roi_start_h, 0.0f);
    auto bin_size_h = roi_height / aligned_height;
    auto bin_size_w = roi_width  / aligned_width;

    auto i = 0u;
    auto h = roi_start_h + bin_size_h/2;
    for (auto ph = 0u; ph < aligned_height; ++ph) {
      auto clipped_h = fmaxf(h,0.0f);
      auto hstart = floorf(clipped_h);

      auto w = roi_start_w + bin_size_w/2;
      for (auto pw = 0u; pw < aligned_width; ++pw) {
        auto clipped_w = fmaxf(w,0.0f);
        auto wstart = floorf(clipped_w);

        // binliear interpolation
        if (clipped_h >= height || clipped_w >= width) {
          top_data[i] = 0.0f;
        } else {
          auto h_ratio = clipped_h - hstart;
          auto w_ratio = clipped_w - wstart;
          auto upleft = hstart * width + wstart;
          auto downleft = upleft;
          auto upright = upleft;
          auto downright = upleft;
          if (hstart==height-1){
            downleft = upleft;
          }else{
            downleft = upleft + width;
          }
          if (wstart==width-1){
            upright = upleft;
            downright = downleft;
          }else{
            upright = upleft + 1;
            downright = downleft + 1;
          }

          top_data[i] = bottom_data[upleft] * (1.0f - h_ratio) * (1.0f - w_ratio)
                      + bottom_data[upright] * (1.0f - h_ratio) * w_ratio
                      + bottom_data[downleft] * h_ratio * (1.0f - w_ratio)
                      + bottom_data[downright] * h_ratio * w_ratio;
        }

        i += 1;
        w += bin_size_w;
      }
      h += bin_size_h;
    }

    return true;
  }
};

template class ROIAlignForwardIPU<float>;
template class ROIAlignForwardIPU<half>;


//
template <class T>
class ROIAlignBackwardIPU : public Vertex {
public:
  Input<Vector<T, SPAN, 4>> top_diff;       //R C Hi Wi
  Input<Vector<T, SPAN, 4>> bottom_rois;    //R 4
  InOut<Vector<T, SPAN, 4>> bottom_diff;   //R C Hi Wi
  T spatial_scale;
  int height;
  int width;
  int aligned_height;
  int aligned_width;
  int channels;
  int num_rois;
  int mode;
  int remainder;
  //there is no channel considered

  bool compute() {

    auto roi_start_w = bottom_rois[0] * spatial_scale - 0.5f;
    auto roi_start_h = bottom_rois[1] * spatial_scale - 0.5f;
    auto roi_end_w   = bottom_rois[2] * spatial_scale - 0.5f;
    auto roi_end_h   = bottom_rois[3] * spatial_scale - 0.5f;

    // Force malformed ROI to be 1x1
    auto roi_width  = fmaxf(roi_end_w - roi_start_w, 0.0f);
    auto roi_height = fmaxf(roi_end_h - roi_start_h, 0.0f);
    auto bin_size_h = roi_height / aligned_height;
    auto bin_size_w = roi_width  / aligned_width;

    auto i = 0u;
    auto h = roi_start_h + bin_size_h/2;
    for (auto ph = 0u; ph < aligned_height; ++ph) {
      auto clipped_h = fmaxf(h,0.0f);
      auto hstart = floorf(clipped_h);

      auto w = roi_start_w + bin_size_w/2;
      for (auto pw = 0u; pw < aligned_width; ++pw) {
        auto clipped_w = fmaxf(w,0.0f);
        auto wstart = floorf(clipped_w);

        // binliear interpolation
        if (clipped_h >= height || clipped_w >= width) {
          // do nothing
        } else {
          auto h_ratio = clipped_h - hstart;
          auto w_ratio = clipped_w - wstart;
          int upleft = hstart * width + wstart;
          int downleft = upleft;
          int upright = upleft;
          int downright = upleft;
          if (hstart==height-1){
            downleft = upleft;
          }else{
            downleft = upleft + width;
          }
          if (wstart==width-1){
            upright = upleft;
            downright = downleft;
          }else{
            upright = upleft + 1;
            downright = downleft + 1;
          }

          auto w_m = 1.f - w_ratio;
          auto h_m = 1.f - h_ratio;
          auto val = top_diff[i];

          bottom_diff[upleft] += val * h_m * w_m;
          bottom_diff[upright] += val * h_m *  w_ratio;
          bottom_diff[downleft] += val * h_ratio * w_m;
          bottom_diff[downright] += val * h_ratio * w_ratio;
        }

        i += 1;
        w += bin_size_w;
      }
      h += bin_size_h;
    }
    return true;
  }
};

template class ROIAlignBackwardIPU<float>;
template class ROIAlignBackwardIPU<half>;
