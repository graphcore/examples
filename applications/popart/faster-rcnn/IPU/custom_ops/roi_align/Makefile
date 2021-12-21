CXX = g++
POPC = popc
CXXFLAGS = -std=c++14 -fPIC -O2 -DONNX_NAMESPACE=onnx
LDLIBS = -shared -lpopart -lpoplar -lpopops
INCLUDES = -I../include


BUILD_DIR = build
SOURCES = roi_align.cpp
TARGET = $(BUILD_DIR)/roi_align.so

all: create_build_dir codelet custom_op

create_build_dir:
	mkdir -p $(BUILD_DIR)

codelet:
	${POPC} -O2 roiAlignCodelets.cpp -o ${BUILD_DIR}/roiAlignCodelets.gp

custom_op: roi_align.cpp
	$(CXX) $(SOURCES) $(LDLIBS) $(CXXFLAGS) $(INCLUDES) -o $(TARGET)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
   
