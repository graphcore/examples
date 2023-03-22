CXX ?= g++
CXXFLAGS = -std=c++11 -ggdb3 -O3
LDLIBS = -lpopart -lpoplar -lpopops
ONNX_NAMESPACE = -DONNX_NAMESPACE=onnx

BUILD_DIR = build
SOURCES = cube_fn_custom_op.cpp
TARGET = $(BUILD_DIR)/cube_fn_custom_op

all: create_build_dir $(TARGET)

.PHONY: create_build_dir
create_build_dir:
	mkdir -p $(BUILD_DIR)

$(TARGET): cube_fn_custom_op.cpp
	$(CXX) $(SOURCES)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(TARGET)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
