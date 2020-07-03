all: custom_ops/plugin_version custom_ops.so

custom_ops/plugin_version: custom_ops/compile_time_version.*
	g++ custom_ops/compile_time_version.cpp -o custom_ops/plugin_version -l poplar

custom_ops.so: custom_ops/plugin_version custom_ops/*.cpp
	g++ -std=c++14 -fPIC \
		-DSTATIC_VERSION=\"${shell ./custom_ops/plugin_version}\" \
		-DONNX_NAMESPACE=onnx \
		custom_ops/detach.cpp \
		custom_ops/disable_attn_dropout_bwd_pattern.cpp \
		custom_ops/sparse_sgd1_accumulate.cpp \
		custom_ops/sparse_sgd1_pattern.cpp \
		custom_ops/tied_gather.cpp \
		custom_ops/tied_gather_pattern.cpp \
		custom_ops/outline_dropout_pattern.cpp \
		custom_ops/seed_modify.cpp \
		-shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand \
		-o custom_ops.so

.PHONY : clean
clean:
	-rm custom_ops.so custom_ops/plugin_version 
