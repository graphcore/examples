
CXX ?= g++
CXXFLAGS = -std=c++14 -fPIC -g
LDLIBS = -shared -lpopart
ONNX_NAMESPACE = -DONNX_NAMESPACE=onnx

BUILD_DIR = utils/custom_ops/build
COPY_SOURCE = utils/custom_ops/copy_tensor/copy_custom_op.cpp
COPY_TARGET = $(BUILD_DIR)/copy_tensor_custom_op.so

all: create_build_dir copy_custom_op 

.PHONY: create_build_dir
create_build_dir: 
	mkdir -p $(BUILD_DIR)

copy_custom_op: utils/custom_ops/copy_tensor/copy_custom_op.cpp
	$(CXX) $(COPY_SOURCE)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(COPY_TARGET)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
