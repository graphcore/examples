
CXX ?= g++
CXXFLAGS =  -std=c++14 -g -fPIC -shared -I..
LDLIBS =  -lpopart -lpoplar -lpopops
ONNX_NAMESPACE = -DONNX_NAMESPACE=onnx

BUILD_DIR = utils/custom_ops/build
COPY_SOURCE = utils/custom_ops/copy_tensor/copy_custom_op.cpp
COPY_TARGET = $(BUILD_DIR)/copy_tensor_custom_op.so

NMS_DIR = utils/custom_ops/nms
NMS_SOURCE = $(NMS_DIR)/nms_custom_op.cpp
NMS_TARGET = $(BUILD_DIR)/nms_custom_op.so

all: create_build_dir copy_custom_op nms_custom_op

.PHONY: create_build_dir
create_build_dir:
	mkdir -p $(BUILD_DIR)

copy_custom_op: $(COPY_SOURCE)
	$(CXX) $? $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(COPY_TARGET)

nms_custom_op: $(NMS_SOURCE) $(NMS_DIR)/nms.cpp $(NMS_DIR)/ipu_utils.cpp
	$(CXX) $? $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(NMS_TARGET)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
