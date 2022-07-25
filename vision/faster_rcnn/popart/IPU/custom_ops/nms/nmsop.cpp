// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <poplar/Engine.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>
#include <time.h>
#include <iostream>
#include <chrono>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <popops/Reduce.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Sort.hpp>
#include <popnn/Loss.hpp>
#include <poputil/VertexTemplates.hpp>
#include "popops/Cast.hpp"

#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>

#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>
#include <time.h>

using namespace poplar;
using namespace poplar::program;

// =========================
// popart init 
namespace CustomOperators {
    const popart::OperatorIdentifier nms = {"ai.graphcore", "nms", 1};
} // namespace CustomOperators

namespace {
// for C++11 compatibility, we don't use std::make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace

class NMSOp : public popart::Op{
  public:
  std::string debugStr;
  float threshold;
  int numDetections;
  NMSOp(const popart::OperatorIdentifier& _opid,
        const popart::Op::Settings& settings_,
        const float threshold,
        const int numDetections,
        const std::string& _debugStr);

  NMSOp(const NMSOp &) = default;
  NMSOp &operator=(const NMSOp &) = delete;
  ~NMSOp() override                  = default;

  std::unique_ptr<popart::Op> clone() const final {
    return make_unique<NMSOp>(*this);
  }

  virtual void setup();

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  float getThresh() const { return threshold; }
  int getNumDetections() const { return numDetections; }
};

NMSOp::NMSOp(const popart::OperatorIdentifier& _opid,
             const popart::Op::Settings& _settings,
             const float _threshold,
             const int _numDetections,
             const std::string& _debugStr) :
                      popart::Op(_opid, _settings) {
  debugStr      = _debugStr;
  threshold     = _threshold;
  numDetections = _numDetections;
}

std::uint64_t
getCyclesEstimateForOp(const poplar::VertexIntrospector &vertex, const poplar::Target &target){
  return 100;
}

void NMSOp::setup() {

  // out shape and type info
  popart::Shape scoresShape = inInfo(0).shape();
  popart::Shape boxesShape = inInfo(1).shape();

  size_t batchSize = (scoresShape.size() > 2) ? scoresShape[0] : 0;

  popart::Shape scoresClassesOutShape, boxesOutShape, keepOutShape;
  if(batchSize > 0) {
    boxesOutShape.push_back(batchSize);
  }
  scoresClassesOutShape.push_back(scoresShape[0]);
  scoresClassesOutShape.push_back(scoresShape[1]);
  scoresClassesOutShape.push_back(1);

  boxesOutShape.push_back(scoresShape[0]);
  boxesOutShape.push_back(numDetections);

  boxesOutShape.push_back(4);

  keepOutShape.push_back(scoresShape[0]);
  keepOutShape.push_back(numDetections);

  outInfo(0) = {inInfo(0).dataType(), scoresClassesOutShape};
  outInfo(1) = {inInfo(1).dataType(), boxesOutShape};
  outInfo(2) = {popart::DataType::INT32, keepOutShape};
}

//register op
static popart::OpDefinition::DataTypes NMSOpFLOATType = { popart::DataType::FLOAT16,
                                                            popart::DataType::FLOAT};
static popart::OpDefinition::DataTypes NMSOpINTType = { popart::DataType::INT32, 
                                                           popart::DataType::UINT32};
static popart::OpDefinition::DataTypes NMSOpBOOLType = { popart::DataType::BOOL};

static popart::OpDefinition
   NMSOpDef({
     popart::OpDefinition::Inputs
     (
       {
         {"inScores",        NMSOpFLOATType},
         {"inBoxes",         NMSOpFLOATType}
       }
     ),
     popart::OpDefinition::Outputs
     (
       {
         {"outScores",  NMSOpFLOATType},
         {"outBoxes",   NMSOpFLOATType},
         {"keep",       NMSOpINTType}
       }
     ),
     popart::OpDefinition::Attributes
     (
       {
         {"threshold",     {"*"}},
         {"numDetections", {"*"}}
       }
     )
   });

static popart::OpCreator<NMSOp> NMSOpCreator(
    popart::OpDefinitions({{CustomOperators::nms, NMSOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      auto &attr = info.attributes;

      std::string debugStr = attr.getAttribute<popart::Attributes::String>("debug_str", "nms_debug");

      float threshold = attr.getAttribute<popart::Attributes::Float>("threshold", 0.5);
      int numDetections = attr.getAttribute<popart::Attributes::Int>("numDetections", 300);

      return std::unique_ptr<NMSOp>(new NMSOp(info.opid,
                                              info.settings,
                                              threshold,
                                              numDetections,
                                              debugStr));
    },
    true);

class NMSOpx : public popart::popx::Opx {

  public:
    NMSOpx(popart::Op *, popart::popx::Devicex *);
    ~NMSOpx() override = default;
  
    virtual void grow(poplar::program::Sequence &) const final;
  
  private:
    float threshold;
    int numDetections; 
};


NMSOpx::NMSOpx(popart::Op *op, popart::popx::Devicex *devicex) :
  popart::popx::Opx(op, devicex) {
  verifyOp<NMSOp>(op, CustomOperators::nms);
  threshold = static_cast<float>(dynamic_cast<NMSOp *>(op)->getThresh());
  numDetections = static_cast<unsigned>(dynamic_cast<NMSOp *>(op)->getNumDetections());

  std::string local_path = std::string(std::getenv("NMS_CODELET_PATH"));
  graph().addCodelets(local_path.c_str());

  graph().registerPerfEstimator("NmsCoreVertex<float>",                             poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  graph().registerPerfEstimator("NmsCoreVertex<half>",                             poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  graph().registerPerfEstimator("PartialFetchBoxVertex<float>",                poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  graph().registerPerfEstimator("PartialFetchBoxVertex<half>",                poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  graph().registerPerfEstimator("UpdateStateVertex<float>",                   poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  graph().registerPerfEstimator("UpdateStateVertex<half>",                   poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  graph().registerPerfEstimator("fillTrueVertex",                     poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  graph().registerPerfEstimator("SetIthKeepVertex<float>",                     poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  graph().registerPerfEstimator("SetIthKeepVertex<half>",                     poplar::PerfEstimateFunc(getCyclesEstimateForOp));
  
}

poplar::Device getDevice() 
{
  
  auto dm = DeviceManager::createDeviceManager();
  auto devices = dm.getDevices(TargetType::IPU, 1);
  std::cout << "Found " << devices.size() << " devices ..." << std::endl;
  
  if (devices.size() > 0) 
  {
    for (auto &d : devices) 
    {
      if (d.attach()) 
      {
        std::cout << "Using hw device " << d.getId() << " ...\n";
        return std::move(d);
      }
    }
  }

  std::cerr << "Using IpuModel ...\n";
  IPUModel ipuModel;
  return ipuModel.createDevice();
}

// popart init end

//=====================================================
// poplar and poplibs src code

// nms core process func
Program create_nms_core_program(Graph& graph,
                                std::string debugMsg,
                                Tensor& Keep,          
                                Tensor& Score,         
                                Tensor& Box,                 
                                Tensor& Box_i,         
                                Tensor& Score_2D_idx,
                                float threshold,
                                int bs,
                                int vlength)
{
    program::Sequence prog;
    ComputeSet Update_CS = graph.addComputeSet(poputil::templateVertex(debugMsg+"Update_CS"));
    
    int tile = 0;
    int numTiles = graph.getTarget().getNumTiles();

    for(int sample = 0; sample < bs; sample++)
    {
      for(unsigned int idx = 0; idx < vlength; idx++)
      {
        auto v = graph.addVertex(
                        Update_CS, 
                        poputil::templateVertex("NmsCoreVertex", Box.elementType()),
                        {
                            {"idx",      idx},
                            {"nms_thresh", threshold},
                            {"sorted_index", Score_2D_idx[sample]},
                            {"keep_r"  , Keep[sample][idx]},
                            {"score_r"   , Score[sample][idx]},
                            {"box_r"   , Box[sample][idx]},
                            {"box_i"   , Box_i[sample]},
                        }
            );

        graph.setTileMapping(v, tile);
        graph.setTileMapping(Keep[sample][idx] , tile);
        graph.setTileMapping(Score[sample][idx], tile);
        graph.setTileMapping(Box[sample][idx]  , tile);
        graph.setTileMapping(Box_i[sample]  , tile);
        if(tile+1 >= numTiles)
          tile = 0;
        else 
          tile++;
      }
    
    }
    prog.add(Execute(Update_CS));
    auto Score_2D = Score.reshape({bs, vlength});
    return prog;
}


Program fetch_set_result_program(Graph& graph,
                                 std::string debugMsg,
                                 Tensor& Score,   
                                 Tensor& Box,
                                 Tensor& Keep,
                                 Tensor& Box_i,
                                 Tensor& result, 
                                 Tensor& resultbox,     
                                 Tensor& value_index,
                                 Tensor& Score_2D_idx,
                                 int bs,
                                 int vlength,
                                 int numDetections)
{
    program::Sequence prog;
    ComputeSet Set_Result_CS      = graph.addComputeSet(poputil::templateVertex(debugMsg+"SetIthKeep_CS"));
    ComputeSet Result_Index_CS      = graph.addComputeSet(poputil::templateVertex(debugMsg+"ResultIndex_CS"));
    ComputeSet Fetch_ielements_CS = graph.addComputeSet(poputil::templateVertex(debugMsg+"Fetch_ielements_CS"));
    graph.setTileMapping(Box_i, 1);

    int numTiles = graph.getTarget().getNumTiles(); 
    //reshape cause pass3 codelets require 2D tensors 
    Tensor Score_2D = Score.reshape({bs, vlength});
    Tensor Box_2D   = Box.reshape({bs, vlength*4});
    int numWorkers = graph.getTarget().getNumWorkerContexts() * graph.getTarget().getNumTiles();

    int L = vlength;
    numWorkers = 1216;
    int numRowsPerWorker = (L + numWorkers - 1) / numWorkers;
    int numVertices = L / numRowsPerWorker + 1;
    int tile = 0;

    tile = 0;
    for(int sample = 0; sample < bs; sample++)
      for(int idx = 0; idx < vlength; idx++)
      {
        graph.setTileMapping(Score[sample][idx]  , tile);
        graph.setTileMapping(Box[sample][idx]    , tile);
      
        if(tile+1 >= numTiles)
          tile = 0;
        else 
          tile++;
      }
    
    poplar::Tensor Box_dist_Tensor     = graph.addVariable(Box.elementType(), {numVertices, bs, 4}, "Box_dist_T"); 
    for (int i = 0; i < numVertices; i++) {
      int rowStart = i * numRowsPerWorker * 4;
      int rowEnd = std::min(L * 4, rowStart + numRowsPerWorker * 4);
      poplar::Tensor workerBox = Box_2D.slice(rowStart, rowEnd, 1); 
      graph.setTileMapping(workerBox, tile);
      graph.setTileMapping(Box_dist_Tensor[i], tile);
      poplar::VertexRef v3 = graph.addVertex(Fetch_ielements_CS, 
                              poputil::templateVertex("PartialFetchBoxVertex", Box.elementType()),
                              {
                                  {"in_row_start", rowStart},
                                  {"in_row_end", rowEnd},
                                  {"in_tensor", workerBox},
                                  {"sorted_index", Score_2D_idx[0]},
                                  {"out_val", Box_dist_Tensor[i]},
                                  {"batch_size", bs}
                              }
              );
      graph.setTileMapping(v3, tile);
      if(tile+1 >= numTiles)
        tile = 0;
      else 
        tile++;
    }
    prog.add(Execute(Fetch_ielements_CS));// 4 Vertices
    Box_i = popops::reduce(graph, Box_dist_Tensor, Box.elementType(), {0}, popops::Operation::ADD, prog, debugMsg+"FetchVertexOutValBox");
  
    for(int sample = 0; sample < bs; sample++)
      graph.setTileMapping(Box_i[sample], 6);

    
    resultbox = resultbox.reshape({bs, 4 * numDetections});
    for (int i = 0; i < numDetections; i++) {
      int rowStart = i;
      int rowEnd = std::min(L, rowStart + 1);
      poplar::Tensor worker_ResultBox = resultbox.slice(rowStart * 4, rowEnd * 4, 1); 
      poplar::Tensor worker_result_idx = result.slice(rowStart, rowEnd, 1); 
      graph.setTileMapping(worker_ResultBox, tile);
      graph.setTileMapping(worker_result_idx, tile);
      poplar::VertexRef v3 = graph.addVertex(Set_Result_CS, 
                              poputil::templateVertex("PartialSetIthKeepVertex", Box.elementType()),
                              {
                                  {"in_row_start", rowStart},
                                  {"in_row_end", rowEnd},
                                  {"result", worker_result_idx},
                                  {"resultbox", worker_ResultBox},
                                  {"sorted_index", Score_2D_idx},
                                  {"box_i", Box_i},
                                  {"batch_size", bs},
                                  {"index", value_index}
                              }
              );
      graph.setTileMapping(v3, tile);
      if(tile+1 >= numTiles)
        tile = 0;
      else 
        tile++;
    }
    graph.setTileMapping(value_index, 3);
    prog.add(Execute(Set_Result_CS)); // 1 Vertex)
    poplar::VertexRef value_index_add_one = graph.addVertex(Result_Index_CS, 
                              poputil::templateVertex("addOneVertex"),
                              {
                                  {"batch_size", bs},
                                  {"var", value_index}
                              }
              );
    graph.setTileMapping(value_index_add_one, tile);
    prog.add(Execute(Result_Index_CS)); 
    graph.setTileMapping(value_index, 3);
    return prog;
}

Program init_iter_round_program(Graph& graph,
                                std::string debugMsg,
                                Tensor& Keep,          
                                Tensor& Score,                    
                                Tensor& Score_2D_idx,
                                int bs,
                                int vlength)
{
    program::Sequence prog;
    Tensor Score_2D = Score.reshape({bs, vlength});
    graph.setTileMapping(Score_2D_idx, 1);
    Score_2D_idx = popnn::argMax(graph, Score_2D, prog, debugMsg+"argmax");
    Score_2D_idx = popops::cast(graph,Score_2D_idx,poplar::INT,prog);
    Score_2D_idx = Score_2D_idx.reshape({bs, 1});
    return prog;
}

void init_params(Graph& graph, 
                 std::string debugMsg,
                 Sequence& prog,
                 Tensor& value_index,
                 Tensor& keep,
                 int bs,
                 int vlength
                 ) {
  const auto initializePhaseCS = graph.addComputeSet(debugMsg + "initializePhaseCS");
  unsigned int numWorkers = graph.getTarget().getNumWorkerContexts() * graph.getTarget().getNumTiles();
  unsigned int numTiles = graph.getTarget().getNumTiles(); 
  unsigned int L = vlength * bs;
  numWorkers = 1216;
  unsigned int numRowsPerWorker = (L + numWorkers - 1) / numWorkers;
  unsigned int numVertices = L / numRowsPerWorker + 1;
  unsigned int tile = 0;
  keep = keep.reshape({bs * vlength});
  for (int i = 0; i < numVertices; ++i)
  {
      unsigned int rowStart = i * numRowsPerWorker;
      unsigned int rowEnd = std::min(L, rowStart + numRowsPerWorker);
      poplar::Tensor workerKeep = keep.slice(rowStart, rowEnd);
      graph.setTileMapping(workerKeep, tile);
      poplar::VertexRef fillTrueVertex =
          graph.addVertex(initializePhaseCS,
                        poputil::templateVertex("fillTrueVertex"),
                        {
                          {"keep", workerKeep}
                        });
      graph.setTileMapping(fillTrueVertex, tile);
      if(tile + 1 >= numTiles)
        tile = 0;
      else 
        tile++;
  }
  keep = keep.reshape({bs, vlength, 1});
  poplar::VertexRef fillZeroVertex =
    graph.addVertex(initializePhaseCS,
                      poputil::templateVertex("fillZeroVertex"),
                      {
                        {"var", value_index}
                      });
    graph.setTileMapping(fillZeroVertex, 2);
  prog.add(poplar::program::Execute(initializePhaseCS));
}

void post_process_result(Graph& graph, std::string debugMsg, Sequence& prog, int bs, Tensor& result) {
  // wq add
  // This function is used to set the index of the redundant box to -1
  const auto setValidResult = graph.addComputeSet(debugMsg + "setValidResult");
  unsigned int numTiles = graph.getTarget().getNumTiles();
  unsigned tile = 0;
  for (unsigned i = 0; i < bs; i++) {
    unsigned tileId = tile % numTiles;
    poplar::VertexRef setResultVertex = graph.addVertex(setValidResult,
                                                        poputil::templateVertex("setResultVertex"), 
                                                        {
                                                          {"res", result[i]}
                                                        });
    graph.setTileMapping(setResultVertex, tileId);
    tile++;
  }
  prog.add(poplar::program::Execute(setValidResult));
}

// the forward process 
void NMSOpx::grow(poplar::program::Sequence& prog) const {
  std::string debugMsg = "NMSOpx::grow";
  poplar::Tensor Score_Tensor = getInTensor(0);  // {B, L}
  poplar::Tensor Box_Tensor = getInTensor(1);  // {B, L, 4}

  auto score_shape = Score_Tensor.shape();
  long unsigned int Bs = score_shape[0];
  long unsigned int N = score_shape[1];
  Score_Tensor = Score_Tensor.reshape({Bs, score_shape[1], 1});
  
  poplar::Tensor Score_2D_idx          = graph().addVariable(INT     , {Bs, 1}   , "score_2d_idx");
  poplar::Tensor Keep_Tensor           = graph().addVariable(FLOAT            , {Bs, N, 1}, "Keep_T");
  poplar::Tensor result_Tensor         = graph().addVariable(INT     , {Bs, numDetections}   , "Result_T");
  poplar::Tensor resultbox_Tensor      = graph().addVariable(Box_Tensor.elementType(), {Bs, numDetections, 4}   , "ResultBox_T");
  poplar::Tensor value_index           = graph().addVariable(INT         , {Bs}      , "value_index_T");
  poplar::Tensor Box_i_Tensor          = graph().addVariable(Box_Tensor.elementType() , {Bs, 4}   , "Box_i_T");

  poplar::program::Sequence core_programs;
 
  init_params(graph(), debugMsg, prog, value_index, Keep_Tensor, Bs, N);
  
  auto iter_init_prog = init_iter_round_program(graph(),
                                                debugMsg,
                                                Keep_Tensor, Score_Tensor, Score_2D_idx,
                                                Bs, N);
                                                
  auto fetch_set_prog = fetch_set_result_program(graph(),
                                                 debugMsg,
                                                 Score_Tensor, Box_Tensor, Keep_Tensor,
                                                 Box_i_Tensor, result_Tensor, resultbox_Tensor, 
                                                 value_index, Score_2D_idx,
                                                 Bs, N, numDetections);
   
  auto nms_core_prog = create_nms_core_program(graph(),
                                               debugMsg,
                                               Keep_Tensor, Score_Tensor, Box_Tensor, 
                                               Box_i_Tensor,
                                               Score_2D_idx,
                                               threshold, Bs, N);
    

  core_programs.add(iter_init_prog);
  core_programs.add(fetch_set_prog);
  core_programs.add(nms_core_prog);
  
  prog.add(Repeat(numDetections, core_programs));
  post_process_result(graph(), debugMsg, prog, Bs, result_Tensor);

  // outScores
  setOutTensor(0, Score_Tensor);

  // outBoxes
  resultbox_Tensor = resultbox_Tensor.reshape({Bs, numDetections, 4});
  setOutTensor(1, resultbox_Tensor);

  // keep
  setOutTensor(2, result_Tensor);
}

// popart related code 
static popart::popx::OpxCreator<NMSOpx> NMSOpxCreator(CustomOperators::nms);


namespace ONNX_NAMESPACE{
    void nmsShapeInference(InferenceContext &ctx){
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      propagateElemTypeFromInputToOutput(ctx, 0, 1);
      propagateElemTypeFromInputToOutput(ctx, 0, 2);
      auto input_shape   = ctx.getInputType(0)->tensor_type().shape();
      auto *output_shape0 =
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      auto *output_shape1 =
          ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
      auto *output_shape2 =
          ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();

      auto numDetections = getAttribute(ctx,"numDetections",300);

      *output_shape0->add_dim() = input_shape.dim(0);
      *output_shape0->add_dim() = input_shape.dim(1);
      output_shape0->add_dim()->set_dim_value(1L);

      *output_shape1->add_dim() = input_shape.dim(0);
      output_shape1->add_dim()->set_dim_value(numDetections);
      output_shape1->add_dim()->set_dim_value(4L);

      *output_shape2->add_dim() = input_shape.dim(0);
      output_shape2->add_dim()->set_dim_value(numDetections);
      
    }

    static const char nmsDoc[] = "nms doc";

    ONNX_OPERATOR_SET_SCHEMA_EX(
        nms,
        AiGraphcore,
        popart::Domain::ai_graphcore,
        1,
        false,
        OpSchema()
            .SetDoc(nmsDoc)
            .Input(0,"input_scores","input_scores","T")
            .Input(1,"input_boxes","input_boxes","T")
            .Output(0,"Score_Tensor","Output tensor","T")
            .Output(1,"resultbox_Tensor","Output tensor","T")
            .Output(2,"result_Tensor","Output tensor","Tint")
            .Attr("threshold","threshold",AttributeProto::FLOAT)
            .Attr("numDetections","numDetections",AttributeProto::INT)
            .TypeConstraint(
                "T",
                {"tensor(float)", "tensor(float16)"},
                "Constrain input and output types to signed numeric tensors."
            )
            .TypeConstraint(
                "Tint",
                {"tensor(int16)", "tensor(int32)"},
                "Integer types")
            .TypeAndShapeInferenceFunction(nmsShapeInference)
    );

    static bool registerOps() {

        ONNX_NAMESPACE::RegisterSchema(
            GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1,nms)>());

        return true;
    }

    static bool ret = registerOps();

}// namespace ONNX_NAMESPACE


// -------------- cppimport --------------
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wsign-compare', '-shared']
cfg['libraries'] = ['popart', 'poplar', 'popops', 'poputil', 'popnn']
cfg['include_dirs'] = ['../include']
setup_pybind11(cfg)
%>
*/
