custom_ops.so: custom_ops/*
	g++ -std=c++14 -fPIC -g \
		custom_ops/attention.cpp \
		custom_ops/embeddingGather.cpp \
		custom_ops/detach.cpp \
		-shared -lpopart -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -lpoprand \
		-o custom_ops.so
