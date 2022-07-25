CXX ?= g++
CXXFLAGS = -std=c++14 -g -fPIC -shared
LDLIBS = -lpopart -lpoplar -lpopops
ONNX_NAMESPACE = -DONNX_NAMESPACE=onnx

BUILD_DIR = ../../build
SOURCES = exp_avg_custom_op.cpp
TARGET = $(BUILD_DIR)/exp_avg_custom_op.so

all: $(TARGET)

$(TARGET): $(SOURCES) $(MAKEFILE_LIST)
	$(CXX) $(SOURCES)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(TARGET)

.PHONY: clean
clean:
	rm -rf  $(TARGET)
