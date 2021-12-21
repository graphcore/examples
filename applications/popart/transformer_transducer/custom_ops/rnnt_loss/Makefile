CXX ?= g++
CXXFLAGS = -std=c++14 -g -fPIC -shared -DONNX_NAMESPACE=onnx -Wall -Wno-sign-compare -Wno-switch -I./
LDLIBS = -lpopart -lpoplar -lpopops -lpoputil

BUILD_DIR = ../../build
RNNT_LOSS_SOURCES = rnnt_loss.cpp rnnt_utils.cpp ipu_utils.cpp
RNNT_LOSS_TARGET = $(BUILD_DIR)/rnnt_loss.so
SPARSE_LOGSOFTMAX_SOURCES = sparse_logsoftmax.cpp rnnt_utils.cpp ipu_utils.cpp logsoftmax.cpp
SPARSE_LOGSOFTMAX_TARGET = $(BUILD_DIR)/sparse_logsoftmax.so

all: $(RNNT_LOSS_TARGET) $(SPARSE_LOGSOFTMAX_TARGET)

$(RNNT_LOSS_TARGET): $(RNNT_LOSS_SOURCES) $(MAKEFILE_LIST)
	$(CXX) $(RNNT_LOSS_SOURCES)  $(LDLIBS) $(CXXFLAGS) -o $(RNNT_LOSS_TARGET)

$(SPARSE_LOGSOFTMAX_TARGET): $(SPARSE_LOGSOFTMAX_SOURCES) $(MAKEFILE_LIST)
	$(CXX) $(SPARSE_LOGSOFTMAX_SOURCES) $(LDLIBS) $(CXXFLAGS) -o $(SPARSE_LOGSOFTMAX_TARGET)

.PHONY: clean
clean:
	rm -rf  $(RNNT_LOSS_TARGET) $(SPARSE_LOGSOFTMAX_TARGET)
