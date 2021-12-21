CXX ?= g++
CXXFLAGS = -std=c++17 -g -fPIC -Wall

BUILD_DIR = ../../build

FA_SRCS = feat_proc.cpp np_fp16_convert.cpp torch_random.cpp feat_proc_async.cpp
FA_LIB = feat_proc.so
FA_TARGET = $(BUILD_DIR)/$(FA_LIB)
FA_INCLUDES = -I./
FA_LDFLAGS = -lpthread

all: release

trace: CXXFLAGS += -DTRACE
trace: debug

debug: CXXFLAGS += -DDEBUG -O0
debug: targets

release: CXXFLAGS += -O3
release: targets

targets: $(FA_TARGET)

$(FA_TARGET): $(FA_SRCS) $(MAKEFILE_LIST)
	$(CXX) $(FA_SRCS) $(CXXFLAGS) $(FA_INCLUDES) $(FA_LDFLAGS) -shared -o $(FA_TARGET)

.PHONY: clean
clean:
	rm -rf  $(FA_TARGET)
