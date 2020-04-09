CXX ?= g++
CXXFLAGS = -std=c++14 -fPIC -g
LDLIBS = -shared -lpopart

BUILD_DIR = build
SOURCES = leaky_relu_custom_op.cpp
TARGET = $(BUILD_DIR)/custom_ops.so

all: create_build_dir leaky_relu_custom_op 

.PHONY: create_build_dir
create_build_dir: 
	mkdir -p $(BUILD_DIR)

leaky_relu_custom_op: leaky_relu_custom_op.cpp
	$(CXX) $(SOURCES)  $(LDLIBS) $(CXXFLAGS) -o $(TARGET)

.PHONY: clean
clean:
	rm -r  $(BUILD_DIR)
