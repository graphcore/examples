# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
CXX = g++

CXXFLAGS = -std=c++17 -fPIC

all: release

debug: CXXFLAGS += -DDEBUG -g -O0
debug: targets

release: CXXFLAGS += -O2
release: targets

POPLAR_LIBS = -lpoplar -lpoputil -lpopops -lpoplin -lpopsparse -lpopnn

targets: custom_ops/plugin_version libblock_sparse.so

custom_ops/plugin_version: custom_ops/compile_time_version.*
	g++ custom_ops/compile_time_version.cpp -o custom_ops/plugin_version -l poplar

lib%.so: custom_ops/plugin_version custom_ops/*.cpp custom_ops/*.hpp Makefile
	$(CXX) $(CXXFLAGS) -DSTATIC_VERSION=\"${shell ./custom_ops/plugin_version}\" \
                custom_ops/bsmatmul.cpp \
                custom_ops/bssoftmax.cpp \
                custom_ops/bsutils.cpp \
                $(POPLAR_LIBS) \
                -shared \
                -o $@

.PHONY : clean
clean:
	-rm libblock_sparse.so custom_ops/plugin_version 
