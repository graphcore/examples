# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

all: custom_ops/plugin_version custom_ops.so

custom_ops/plugin_version: custom_ops/compile_time_version.*
	g++ custom_ops/compile_time_version.cpp -o custom_ops/plugin_version -l poplar

custom_ops.so: custom_ops/plugin_version custom_ops/*.cpp custom_ops/*.hpp Makefile
	g++ -std=c++14 -fPIC -g -DSTATIC_VERSION=\"${shell ./custom_ops/plugin_version}\" \
                -DONNX_NAMESPACE=onnx\
                custom_ops/block_sparse_ops.cpp \
                -shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand -lpopsparse \
                -o custom_ops.so

.PHONY : clean
clean:
	-rm custom_ops.so custom_ops/plugin_version 

