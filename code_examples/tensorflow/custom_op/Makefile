CC = g++

all: libcustom_op.so

custom_codelet.gp: custom_codelet.cpp
	popc custom_codelet.cpp -o $@

libcustom_op.so: custom_codelet.gp poplar_code.cpp
	$(CC) poplar_code.cpp -shared -fpic -Wl,-soname,$@ -o $@ -lpoplar -lpoputil

clean:
	rm libcustom_op.so custom_codelet.gp
