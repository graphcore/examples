all: custom_ops.so

custom_ops.so:
	g++ -std=c++14 -fPIC -O2 \
		-DONNX_NAMESPACE=onnx \
		custom_ops/remap_tensor_ce.cpp \
		custom_ops/TileMappingCommon.cpp \
		-shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand \
		-o custom_ops.so

.PHONY : clean
clean:
	-rm custom_ops.so
