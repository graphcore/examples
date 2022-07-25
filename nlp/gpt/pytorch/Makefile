all: custom_ops/plugin_version custom_ops.so

custom_ops/plugin_version: custom_ops/compile_time_version.*
	g++ custom_ops/compile_time_version.cpp -o custom_ops/plugin_version -l poplar

custom_ops.so: custom_ops/plugin_version custom_ops/*.cpp custom_ops/workarounds/*.cpp
	g++ -std=c++14 -fPIC \
		-DSTATIC_VERSION=\"${shell ./custom_ops/plugin_version}\" \
		-DONNX_NAMESPACE=onnx \
		custom_ops/workarounds/residual_add_inplace_pattern.cpp \
		custom_ops/remap_tensor_ce.cpp \
		-shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand \
		-o custom_ops.so

.PHONY : clean
clean:
	-rm custom_ops.so custom_ops/plugin_version 
