CC = g++
CXXFLAGS_LIB = -std=c++14 -g -fPIC -shared -I.. -I/opt/graphcore/boost/include/
LDLIBS = -lpoplar -lpopops -lpoputil -lpopnn
BUILD_DIR = build
SOURCES = nms_custom_op.cpp
TARGET_LIB = $(BUILD_DIR)/nms_custom_op.so

all: create_build_dir nms_custom_op codelet.gp

.PHONY: create_build_dir
create_build_dir:
	mkdir -p $(BUILD_DIR)

codelet.gp: ../codelet.cpp
	popc $? -O2 -o $@


nms_custom_op: nms_custom_op.cpp ../nms.cpp ../ipu_utils.cpp
	$(CXX) $?  $(LDLIBS) $(CXXFLAGS_LIB) $(ONNX_NAMESPACE)  -Wl,-soname,$@ -o $(TARGET_LIB)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
