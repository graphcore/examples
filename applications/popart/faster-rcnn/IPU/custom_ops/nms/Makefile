CXX = g++
CXXFLAGS = -std=c++14 -fPIC -O2 -DONNX_NAMESPACE=onnx -v
LDLIBS1 = -lpoplar -lpopops -lpoputil
LDLIBS2 = -lpoplar -lpopops -lpoputil -lpopart -shared
INCLUDES = -I ../include

BUILD_DIR = build

all: create_build_dir nmsop

.PHONY: create_build_dir
create_build_dir:
	mkdir -p $(BUILD_DIR)

nmsop: nmsop.cpp
	$(CXX) nmsop.cpp $(LDLIBS2) $(CXXFLAGS) $(INCLUDES) -o $(BUILD_DIR)/libnms_ops.so -g

.PHONY: clean
clean:
	rm -r  $(BUILD_DIR)
