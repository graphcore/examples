// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
//
// This custom op implements roi-align operation
//
//
#include <ostream>

#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>

#include <popops/ElementWise.hpp>
#include <poplar/Program.hpp>
#include <poplar/OptionFlags.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Zero.hpp>
#include <poputil/VertexTemplates.hpp>
#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>
#include <popops/Reduce.hpp>

#define PRINT_TENSOR(FLAG, PROGRAM, TENSOR, DBGMSG){\
if(FLAG)\
  PROGRAM.add(poplar::program::PrintTensor((DBGMSG), TENSOR));\
}

namespace CustomOperators {
  const popart::OperatorIdentifier roiAlign = {"ai.graphcore", "roiAlign", 1};
} // namespace CustomOperators
namespace CustomGradOperators {
const static popart::OperatorIdentifier roiAlignGrad = {"ai.graphcore", "roiAlignGrad", 1};
} // namespace CustomGradOperators

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace


// The gradient Op
// This is defined first as the CubeOp::getGradOps requires it.
class roiAlignGradOp : public popart::Op {
public:
  std::string debugStr;
  float spatial_scale;
  int batch_size;
  int num_rois;
  int height;
  int width;
  int channels;
  int aligned_height;
  int aligned_width;
public:
  roiAlignGradOp(const popart::Op &fwdOp)
      : popart::Op(CustomGradOperators::roiAlignGrad, fwdOp.getSettings()) {}
  roiAlignGradOp(const popart::Op &fwdOp,
                  const float spatial_scale_,
                  const int batch_size_,
                  const int num_rois_,
                  const int height_,
                  const int width_,
                  const int channels_,
                  const int aligned_height_,
                  const int aligned_width_)
      : popart::Op(CustomGradOperators::roiAlignGrad, fwdOp.getSettings()) {
          spatial_scale  = spatial_scale_;
          batch_size     = batch_size_;
          num_rois       = num_rois_;
          height         = height_;
          width          = width_;
          channels       = channels_;
          aligned_height = aligned_height_;
          aligned_width  = aligned_width_;
      }

  std::unique_ptr<Op> clone() const final {
    return make_unique<roiAlignGradOp>(*this);
  }

  virtual void setup() {
    // bottom_data {B, C, Hi, Wi}
    // bottom_rois {B, D, 5}
    // top_data {B, D, C, Ho, Wo}

#ifdef DEBUG
    std::cout << "entering setup()\n";
#endif

    // out shape and type info
    popart::Shape top_diff_shape = inInfo(0).shape();
    popart::Shape bottom_rois_shape = inInfo(1).shape();
#ifdef DEBUG
    std::cout << "roiAlignOpGrad::setup()::top_diff_shape:" << \
                top_diff_shape << std::endl;
    std::cout << "roiAlignOpGrad::setup()::bottom_rois_shape:" << \
                bottom_rois_shape << std::endl;
#endif

    if(top_diff_shape.size() < 4 or
      bottom_rois_shape.size() < 3) {
      throw poplar::poplar_error("Size error");
    }

    popart::Shape bottom_diff_shape;
    bottom_diff_shape.push_back(batch_size);
    bottom_diff_shape.push_back(channels);
    bottom_diff_shape.push_back(height);
    bottom_diff_shape.push_back(width);

    popart::DataType dataType = inInfo(0).dataType();
    outInfo(0) = {dataType, bottom_diff_shape};

#ifdef DEBUG
    std::cout << "roiAlignOp::setup()::bottom_diff_shape:" << \
                bottom_diff_shape << std::endl;
    std::cout << "exiting setup()\n";
#endif
}

virtual const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
  static const std::vector<popart::GradInOutMapper> inInfo = {
      {0, 0, popart::GradOpInType::GradOut},
      {1, 1, popart::GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

float getSubgraphValue() const final { return getLowSubgraphValue(); }

};

class roiAlignOp : public popart::Op {
public:
  std::string debugStr;
  float spatial_scale;
  int batch_size;
  int num_rois;
  int height;
  int width;
  int channels;
  int aligned_height;
  int aligned_width;

  roiAlignOp(const popart::OperatorIdentifier& _opid,
             const popart::Op::Settings& settings_,
             const float spatial_scale,
             const int batch_size,
             const int num_rois,
             const int height,
             const int width,
             const int channels,
             const int aligned_height,
             const int aligned_width,
             const std::string& _debugStr);

  roiAlignOp(const roiAlignOp &) = default;
  roiAlignOp &operator=(const roiAlignOp &) = delete;
  ~roiAlignOp() override                    = default;

  std::unique_ptr<popart::Op> clone() const final {
    return make_unique<roiAlignOp>(*this);
  }

  virtual void setup();

  std::vector<std::unique_ptr<popart::Op>> getGradOps() final {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new roiAlignGradOp(*this,
                                          spatial_scale,
                                          batch_size,
                                          num_rois,
                                          height,
                                          width,
                                          channels,
                                          aligned_height,
                                          aligned_width));
    return upops;
  }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};


roiAlignOp::roiAlignOp(const popart::OperatorIdentifier& _opid,
                       const popart::Op::Settings& _settings,
                       const float _spatial_scale,
                       const int _batch_size,
                       const int _num_rois,
                       const int _height,
                       const int _width,
                       const int _channels,
                       const int _aligned_height,
                       const int _aligned_width,
                       const std::string& _debugStr) :
  popart::Op(_opid, _settings) {

  spatial_scale  = _spatial_scale;
  batch_size     = _batch_size;
  num_rois       = _num_rois;
  height         = _height;
  width          = _width;
  channels       = _channels;
  aligned_height = _aligned_height;
  aligned_width  = _aligned_width;
  debugStr       = _debugStr;

#ifdef DEBUG
  std::cout << "entering roiAlignOp()\n";
  std::cout << "roiAlignOpx::roiAlignOp:" << _spatial_scale << std::endl;
  std::cout << "exiting roiAlignOp()\n";
#endif
}


std::uint64_t
getCyclesEstimateForOp(const poplar::VertexIntrospector &vertex,
                       const poplar::Target &target) {
  return 100;
}


void roiAlignOp::setup() {
  // bottom_data {B, C, Hi, Wi}
  // bottom_rois {B, D, 5}
  // top_data {B, D, C, Ho, Wo}

#ifdef DEBUG
  std::cout << "entering setup()\n";
#endif

  // out shape and type info
  popart::Shape bottom_data_shape = inInfo(0).shape();
  popart::Shape bottom_rois_shape = inInfo(1).shape();

#ifdef DEBUG
  std::cout << "roiAlignOp::setup()::bottom_data_shape:" << \
               bottom_data_shape << std::endl;
  std::cout << "roiAlignOp::setup()::bottom_rois_shape:" << \
               bottom_rois_shape << std::endl;
#endif

  if(bottom_data_shape.size() < 4 or
     bottom_rois_shape.size() < 3) {
     throw poplar::poplar_error("Size error");
  }

  popart::Shape top_data_shape;
  top_data_shape.push_back(batch_size);
  top_data_shape.push_back(num_rois);
  top_data_shape.push_back(channels);
  top_data_shape.push_back(aligned_height);
  top_data_shape.push_back(aligned_width);

  popart::DataType dataType = inInfo(0).dataType();
  outInfo(0) = {dataType, top_data_shape};

#ifdef DEBUG
  std::cout << "roiAlignOp::setup()::top_data_shape:" << \
               top_data_shape << std::endl;
  std::cout << "exiting setup()\n";
#endif
}

//register op
static popart::OpDefinition::DataTypes roiAlignOpFLOATType = {popart::DataType::FLOAT16,
                                                              popart::DataType::FLOAT };

static popart::OpDefinition
    roiAlignOpDef({
      popart::OpDefinition::Inputs
      (
        {
          {"bottom_data", roiAlignOpFLOATType},
          {"bottom_rois", roiAlignOpFLOATType}
        }
      ),
      popart::OpDefinition::Outputs
      (
        {
          {"top_data",  roiAlignOpFLOATType}
        }
      ),
      popart::OpDefinition::Attributes
      (
        {
          {"spatial_scale", {"*"}},
          {"batch_size",    {"*"}},
          {"num_rois",      {"*"}},
          {"height",        {"*"}},
          {"width",         {"*"}},
          {"channels",      {"*"}},
          {"aligned_height",{"*"}},
          {"aligned_width", {"*"}}
        }
      )
    });


static popart::OpCreator<roiAlignOp> roiAlignOpCreator(
    popart::OpDefinitions({{CustomOperators::roiAlign, roiAlignOpDef}}),
    [](const popart::OpCreatorInfo &info){
      
      auto &attr = info.attributes;
      auto &_opid = info.opid;
      auto &settings = info.settings;

      std::string debugStr = attr.getAttribute<popart::Attributes::String>("debug_str", "roi_align_debug");
      float spatial_scale = attr.getAttribute<popart::Attributes::Float>("spatial_scale", 1);
      int batch_size = attr.getAttribute<popart::Attributes::Int>("batch_size", 1);
      int num_rois = attr.getAttribute<popart::Attributes::Int>("num_rois", 5);
      int height = attr.getAttribute<popart::Attributes::Int>("height", 5);
      int width = attr.getAttribute<popart::Attributes::Int>("width", 5);
      int channels = attr.getAttribute<popart::Attributes::Int>("channels", 5);
      int aligned_height = attr.getAttribute<popart::Attributes::Int>("aligned_height", 5);
      int aligned_width = attr.getAttribute<popart::Attributes::Int>("aligned_width", 5);

#ifdef DEBUG
      std::cout << "OpDefinitions, spatial_scale:" << spatial_scale << std::endl;
      std::cout << "OpDefinitions, batch_size:" << batch_size << std::endl;
      std::cout << "OpDefinitions, num_rois:" << num_rois << std::endl;
      std::cout << "OpDefinitions, height:" << height << std::endl;
      std::cout << "OpDefinitions, width:" << width << std::endl;
      std::cout << "OpDefinitions, channels:" << channels << std::endl;
      std::cout << "OpDefinitions, aligned_height:" << aligned_height << std::endl;
      std::cout << "OpDefinitions, aligned_width:" << aligned_width << std::endl;
#endif

      return std::unique_ptr<roiAlignOp>(
        new roiAlignOp(_opid,
                       settings,
                       spatial_scale,
                       batch_size,
                       num_rois,
                       height,
                       width,
                       channels,
                       aligned_height,
                       aligned_width,
                       debugStr));
    },
    true);


class roiAlignOpx : public popart::popx::Opx {

public:
  roiAlignOpx(popart::Op *, popart::popx::Devicex *);
  ~roiAlignOpx() override = default;

  virtual void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor roiAlignImpl(poplar::program::Sequence &prog,
                              poplar::Tensor& bottom_data,
                              poplar::Tensor& bottom_rois,
                              std::string debugMsg) const;
};

roiAlignOpx::roiAlignOpx(popart::Op *op, popart::popx::Devicex *devicex) :
  popart::popx::Opx(op, devicex) {
#ifdef DEBUG
  std::cout << "entering roiAlignOpx()\n";
#endif

  graph().addCodelets("IPU/custom_ops/roi_align/build/roiAlignCodelets.gp");
  graph().registerPerfEstimator("ROIAlignForwardIPU<float>", poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  graph().registerPerfEstimator("ROIAlignForwardIPU<half>",  poplar::PerfEstimateFunc(getCyclesEstimateForOp));

#ifdef DEBUG
  std::cout << "exiting roiAlignOpx()\n";
#endif
}

void roiAlignOpx::grow(poplar::program::Sequence& prog) const {
#ifdef DEBUG
  std::cout << "roiAlignOpx::grow entering\n";
#endif
  // bottom_data  {B, C, Hi, Wi}
  // bottom_rois  {B, R, 4}

  std::string debugMsg = "roiAlignOpx::grow";
  poplar::Tensor bottom_data = getInTensor(0);
  poplar::Tensor bottom_rois = getInTensor(1);

#ifdef DEBUG
  std::cout << "roiAlignOpx::grow():: bottom_data.shape():" << bottom_data.shape() << std::endl;
  std::cout << "roiAlignOpx::grow():: bottom_rois.shape():" << bottom_rois.shape() << std::endl;
#endif

  poplar::Tensor top_data = roiAlignImpl(prog, bottom_data, bottom_rois, debugMsg);

#ifdef DEBUG
  std::cout << "roiAlignOpx::grow():: top_data.shape():" << top_data.shape() << std::endl;
#endif

  // top data
  setOutTensor(0, top_data);

#ifdef DEBUG
  std::cout << "nmsOpx::grow exiting\n";
#endif
}

int CalcGroup(int batch, int rois, int channel, int tile) {
  // the performance is best when the number of vertex of each ComputeSet is tile * 6 
  int nGroup = 6 * tile / (batch * channel);
  int resGroup = fminf(rois, nGroup);
  return resGroup;
}

poplar::Tensor roiAlignOpx::roiAlignImpl(poplar::program::Sequence &prog,
                                         poplar::Tensor& bottom_data,
                                         poplar::Tensor& bottom_rois,
                                         std::string debugMsg) const {
#ifdef DEBUG
  std::cout << "roiAlignOpx::roiAlignImpl:: entering" << std::endl;
#endif
  roiAlignOp& op = getOp<roiAlignOp>();

  /***************** initialization phase compute set *****************/
  auto spatial_scale = op.spatial_scale;
  auto batch_size = op.batch_size;
  auto channels   = op.channels;
  auto height     = op.height;
  auto width      = op.width;
  auto num_rois   = op.num_rois;
  auto aligned_height = op.aligned_height;
  auto aligned_width  = op.aligned_width;

#ifdef DEBUG
  std::cout << "roiAlignOpx::roiAlignImpl::bottom_data : "    << bottom_data.shape() << std::endl;
  std::cout << "roiAlignOpx::roiAlignImpl::spatial_scale : "  << spatial_scale << std::endl;
  std::cout << "roiAlignOpx::roiAlignImpl::batch_size : "     << batch_size << std::endl;
  std::cout << "roiAlignOpx::roiAlignImpl::num_rois : "       << num_rois << std::endl;
  std::cout << "roiAlignOpx::roiAlignImpl::channels : "       << channels << std::endl;
  std::cout << "roiAlignOpx::roiAlignImpl::height : "         << height << std::endl;
  std::cout << "roiAlignOpx::roiAlignImpl::width : "          << width << std::endl;
  std::cout << "roiAlignOpx::roiAlignImpl::aligned_height : " << aligned_height << std::endl;
  std::cout << "roiAlignOpx::roiAlignImpl::aligned_width : "  << aligned_width << std::endl;
#endif

  /***************** top data tensor *****************/
  auto dataType = bottom_data.elementType();
  poplar::Tensor top_data = graph().addVariable(dataType,
                                                { batch_size,
                                                  num_rois,
                                                  channels,
                                                  aligned_height,
                                                  aligned_width },
                                                debugMsg + "/roiAlign/top_data");

  const auto &target = graph().getTarget();
  const auto numTiles = target.getNumTiles();
  
  
  unsigned tile_index = 0;
  poplar::ComputeSet roiAlignCS = graph().addComputeSet("/roiAlign/roiAlignCS");
  for (auto b = 0u; b < batch_size; ++b) {
    for (auto i = 0u; i < num_rois; ++i) {
      for (auto c = 0u; c < channels; ++c) {
        unsigned tile = tile_index % numTiles;
        graph().setTileMapping(top_data[b][i][c], tile);
        poplar::VertexRef roiAlignVertex =
            graph().addVertex(roiAlignCS,
                              poputil::templateVertex("ROIAlignForwardIPU", dataType),
                              {
                                {"bottom_data", bottom_data[b][c].flatten()},   // Input
                                {"bottom_rois", bottom_rois[b][i].flatten()},   // Input
                                {"top_data",    top_data[b][i][c].flatten()}
                              });
        graph().setInitialValue(roiAlignVertex["spatial_scale"], spatial_scale);
        graph().setInitialValue(roiAlignVertex["height"], height);
        graph().setInitialValue(roiAlignVertex["width"], width);
        graph().setInitialValue(roiAlignVertex["aligned_height"], aligned_height);
        graph().setInitialValue(roiAlignVertex["aligned_width"], aligned_width);
        graph().setTileMapping(roiAlignVertex, tile);
        tile_index++;
      }
    }
  }
  prog.add(poplar::program::Execute(roiAlignCS));

#ifdef DEBUG
  std::cout << "roiAlignOpx::roiAlignImpl:: num tiles : " << numTilesUsed << std::endl;
  std::cout << "roiAlignOpx::roiAlignImpl:: used tiles : " << num_rois * batch_size * channels << std::endl;
  std::cout << "roiAlignOpx::roiAlignImpl:: roiAlignCS done" << std::endl;
#endif

  return top_data;
}

class roiAlignGradOpx : public popart::popx::Opx {
public:
  roiAlignGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
#ifdef DEBUG
  std::cout << "entering roiAlignGradOpx()\n";
#endif
  graph().addCodelets("IPU/custom_ops/roi_align/build/roiAlignCodelets.gp");
  graph().registerPerfEstimator("ROIAlignBackwardIPU<float>", poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  graph().registerPerfEstimator("ROIAlignBackwardIPU<half>",  poplar::PerfEstimateFunc(getCyclesEstimateForOp));

#ifdef DEBUG
  std::cout << "exiting roiAlignGradOpx()\n";
#endif
  }

  ~roiAlignGradOpx() override = default;

  virtual void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor roiAlignImpl(poplar::program::Sequence &prog,
                                poplar::Tensor& top_diff,
                                poplar::Tensor& bottom_rois,
                                std::string debugMsg) const;

};

void roiAlignGradOpx::grow(poplar::program::Sequence& prog) const {
#ifdef DEBUG
  std::cout << "roiAlignGradOpx::grow entering\n";
#endif
  // top_diff  {B, R, C, Hi, Wi}
  // bottom_rois  {B, R, 4}

  std::string debugMsg = "roiAlignGradOpx::grow";
  poplar::Tensor top_diff = getInTensor(0);
  poplar::Tensor bottom_rois = getInTensor(1);

#ifdef DEBUG
  std::cout << "roiAlignGradOpx::grow():: top_diff.shape():" << top_diff.shape() << std::endl;
  std::cout << "roiAlignGradOpx::grow():: bottom_rois.shape():" << bottom_rois.shape() << std::endl;
  PRINT_TENSOR(true, prog, top_diff, "top_diff");
  PRINT_TENSOR(true, prog, bottom_rois, "bottom_rois");
#endif
  //
  poplar::Tensor bottom_diff = roiAlignImpl(prog, top_diff, bottom_rois, debugMsg);
  //
#ifdef DEBUG
  PRINT_TENSOR(true, prog, bottom_diff, "bottom_diff");
  std::cout << "roiAlignGradOpx::grow():: bottom_diff.shape():" << bottom_diff.shape() << std::endl;
#endif

  // top data
  setOutTensor(0, bottom_diff);

#ifdef DEBUG
  std::cout << "nmsGradOpx::grow exiting\n";
#endif
}

poplar::Tensor roiAlignGradOpx::roiAlignImpl(poplar::program::Sequence &prog,
                              poplar::Tensor& top_diff,
                              poplar::Tensor& bottom_rois,
                              std::string debugMsg) const {
#ifdef DEBUG
  std::cout << "roiAlignOpx::roiAlignGradImpl:: entering" << std::endl;
#endif
  roiAlignGradOp& op = getOp<roiAlignGradOp>();

  /***************** initialization phase compute set *****************/

  auto spatial_scale = op.spatial_scale;
  auto batch_size = op.batch_size;
  auto channels   = op.channels;
  auto height     = op.height;
  auto width      = op.width;
  auto num_rois   = op.num_rois;
  auto aligned_height = op.aligned_height;
  auto aligned_width  = op.aligned_width;

#ifdef DEBUG
  std::cout << "roiAlignGradOpx::roiAlignGradImpl::top_diff : "    << top_diff.shape() << std::endl;
  std::cout << "roiAlignGradOpx::roiAlignGradImpl::spatial_scale : "  << spatial_scale << std::endl;
  std::cout << "roiAlignGradOpx::roiAlignGradImpl::batch_size : "     << batch_size << std::endl;
  std::cout << "roiAlignGradOpx::roiAlignGradImpl::num_rois : "       << num_rois << std::endl;
  std::cout << "roiAlignGradOpx::roiAlignGradImpl::channels : "       << channels << std::endl;
  std::cout << "roiAlignGradOpx::roiAlignGradImpl::height : "         << height << std::endl;
  std::cout << "roiAlignGradOpx::roiAlignGradImpl::width : "          << width << std::endl;
  std::cout << "roiAlignGradOpx::roiAlignGradImpl::aligned_height : " << aligned_height << std::endl;
  std::cout << "roiAlignGradOpx::roiAlignGradImpl::aligned_width : "  << aligned_width << std::endl;
#endif

  /***************** top data tensor *****************/
  auto dataType = top_diff.elementType();

  //B C H W
  poplar::Tensor bottom_diff = graph().addVariable(dataType,
                                                { batch_size,
                                                  channels,
                                                  height,
                                                  width });

  //prepare
  const auto &target = graph().getTarget();
  const auto numTiles = target.getNumTiles();

  int group = CalcGroup(batch_size, num_rois, channels, numTiles);
  //
  std::map<int, poplar::ComputeSet> map_cs; //rois to cs
  for(int i = 0; i <= num_rois / group; i++) {
    map_cs[i] = graph().addComputeSet("/roiAlignGrad/roiAlignGradCS_" + std::to_string(i));
  }

  std::map<unsigned, std::set<unsigned>> tile2ch;
  for (auto b = 0u; b < batch_size; ++b) {
    for (auto c = 0u; c < channels; ++c) {
      unsigned tile = (c * group) % numTiles;
      tile2ch[tile].insert(c);
      graph().setTileMapping(bottom_diff[b][c], tile);
    }
  }

#ifdef DEBUG
  for(auto it:tile2ch) {
    std::cout<<"Tile : "<<it.first<<", ch : "<<it.second.size()<<std::endl;
  }
#endif
  poplar::Tensor bottom_buff = graph().addVariable(dataType, {batch_size, channels, group, height, width}, "/roiAlign/bottom_buff");
  std::vector<long unsigned int> bottom_buff_shape = bottom_buff.shape();
  auto zero = getScalarVariable(bottom_buff.elementType(), "zero");
  graph().setInitialValue(zero, 0);
  auto bottom_buff_zero = zero;
  for (int i = 0; i < bottom_buff_shape.size(); ++i) {
    bottom_buff_zero = bottom_buff_zero.expand({0});
  }
  for (int i = 0; i < bottom_buff_shape.size(); ++i) {
    bottom_buff_zero = bottom_buff_zero.broadcast(static_cast<unsigned>(bottom_buff_shape[i]), i);
  }
  auto bottom_buff_grad = cloneNcopy(prog, bottom_buff_zero);
  for (auto b = 0u; b < batch_size; ++b) {
    for (auto c = 0u; c < channels; ++c) {
      for (int i = 0u; i < num_rois; i+=group) {
        for (int g = 0u; g < group && i + g < num_rois; g++) {
          unsigned tile = (c * group + g) % numTiles;
          graph().setTileMapping(bottom_buff[b][c][g], tile);
          graph().setTileMapping(bottom_buff_grad[b][c][g], tile);
          poplar::VertexRef roiAlignVertex =
            graph().addVertex(map_cs[i / group],
                              poputil::templateVertex("ROIAlignBackwardIPU", dataType),
                              {
                                {"top_diff", top_diff[b][i + g][c].flatten()},    // Input B R C H W
                                {"bottom_rois", bottom_rois[b][i + g].flatten()}, // Input B R 4
                                {"bottom_diff", bottom_buff_grad[b][c][g].flatten()} 
                              });
          
          graph().setInitialValue(roiAlignVertex["spatial_scale"], spatial_scale);
          graph().setInitialValue(roiAlignVertex["height"], height);
          graph().setInitialValue(roiAlignVertex["width"], width);
          graph().setInitialValue(roiAlignVertex["aligned_height"], aligned_height);
          graph().setInitialValue(roiAlignVertex["aligned_width"], aligned_width);
          graph().setInitialValue(roiAlignVertex["channels"], c);
          graph().setInitialValue(roiAlignVertex["num_rois"], 1);
          graph().setTileMapping(roiAlignVertex, tile);
        }  
      }
    }
  }

  //map
  for(auto i=0; i<map_cs.size(); i++) {
    prog.add(poplar::program::Execute(map_cs[i]));
  }
  bottom_diff = popops::reduce(graph(), bottom_buff_grad, dataType, {2}, popops::Operation::ADD, prog, "/roiAlign/roiAlignGradCS_reduce");

#ifdef DEBUG
  prog.add(poplar::program::PrintTensor("bottom_diff",bottom_diff)); //check result
  std::cout << "roiAlignGradOpx::roiAlignGradImpl:: num tiles : " << numTiles << std::endl;
  std::cout << "roiAlignGradOpx::roiAlignGradImpl:: used tiles : " << num_rois * batch_size * channels << std::endl;
  std::cout << "roiAlignGradOpx::roiAlignGradImpl:: roiAlignCS done" << std::endl;
#endif

  return bottom_diff;
}

static popart::popx::OpxCreator<roiAlignOpx> roiAlignOpxCreator(CustomOperators::roiAlign);

static popart::popx::OpxCreator<roiAlignGradOpx> roiAlignGradOpxCreator(CustomGradOperators::roiAlignGrad);

namespace ONNX_NAMESPACE{
    void roiAlignShapeInference(InferenceContext &ctx){
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      auto input_shape   = ctx.getInputType(0)->tensor_type().shape();
      auto *output_shape =
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      
      auto num_rois = getAttribute(ctx,"num_rois",300);
      auto output_h = getAttribute(ctx,"aligned_height",7);
      auto output_w = getAttribute(ctx,"aligned_width",7);

      *output_shape->add_dim() = input_shape.dim(0);
      output_shape->add_dim()->set_dim_value(num_rois);
      *output_shape->add_dim() = input_shape.dim(1);
      output_shape->add_dim()->set_dim_value(output_h);
      output_shape->add_dim()->set_dim_value(output_w);
    }

    static const char roiAlignDoc[] = "roialign doc";

    ONNX_OPERATOR_SET_SCHEMA_EX(
        roiAlign,
        AiGraphcore,
        popart::Domain::ai_graphcore,
        1,
        false,
        OpSchema()
            .SetDoc(roiAlignDoc)
            .Input(0,"bottom_data","bottom_data","T")
            .Input(1,"bottom_rois","bottom_rois","T")
            .Output(0,"top_data","Output tensor","T")
            .Attr("spatial_scale","spatial_scale",AttributeProto::FLOAT)
            .Attr("batch_size","batch_size",AttributeProto::INT)
            .Attr("num_rois","num_rois",AttributeProto::INT)
            .Attr("height","height",AttributeProto::INT)
            .Attr("width","width",AttributeProto::INT)
            .Attr("channels","channels",AttributeProto::INT)
            .Attr("aligned_height","aligned_height",AttributeProto::INT)
            .Attr("aligned_width","aligned_width",AttributeProto::INT)
            .TypeConstraint(
                "T",
                {"tensor(float)", "tensor(float16)"},
                "Constrain input and output types to signed numeric tensors."
            )
            .TypeAndShapeInferenceFunction(roiAlignShapeInference)
    );

    static bool registerOps() {

        ONNX_NAMESPACE::RegisterSchema(
            GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1,roiAlign)>());

        return true;
    }

    static bool ret = registerOps();

}// namespace ONNX_NAMESPACE
