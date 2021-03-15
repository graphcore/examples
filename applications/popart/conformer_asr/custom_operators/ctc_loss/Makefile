CXX ?= g++
CXXFLAGS = -std=c++14 -g -fPIC -shared -DONNX_NAMESPACE=onnx
LDLIBS = -lpopart -lpoplar -lpopops -lpoputil -lpopnn

BUILD_DIR = build
SOURCE_DIR = .
CTC_LOSS = $(BUILD_DIR)/ctc_loss.so
all: create_build_dir ctc_loss

.PHONY: create_build_dir
create_build_dir: 
	mkdir -p $(BUILD_DIR)

ctc_loss: $(SOURCE_DIR)/ctc_loss.cpp $(SOURCE_DIR)/codelet.cpp
	$(CXX) $ $(SOURCE_DIR)/ctcloss_utils.cpp $(SOURCE_DIR)/ctc_loss.cpp $(LDLIBS) $(CXXFLAGS) -o $(CTC_LOSS)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
