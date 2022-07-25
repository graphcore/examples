CC = g++

all: libcustom_op.so

libcustom_op.so: custom_crop_op.cpp crop.cpp
	$(CC) custom_crop_op.cpp crop.cpp -shared -fpic -Wl,-soname,$@ -o $@ -lpoplar -lpoputil -lpopops

clean:
	rm libcustom_op.so
