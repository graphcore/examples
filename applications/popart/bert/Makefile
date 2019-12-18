all: custom_ops/plugin_version custom_ops.so

custom_ops/plugin_version: custom_ops/compile_time_version.*
	g++ custom_ops/compile_time_version.cpp -o custom_ops/plugin_version -l poplar

custom_ops.so: custom_ops/plugin_version custom_ops/*.cpp
	g++ -std=c++14 -fPIC -g -DSTATIC_VERSION=\"${shell ./custom_ops/plugin_version}\" \
		custom_ops/attention.cpp \
		custom_ops/embeddingGather.cpp \
		custom_ops/detach.cpp \
		-shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand \
		-o custom_ops.so
