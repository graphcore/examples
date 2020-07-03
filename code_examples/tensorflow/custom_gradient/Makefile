CC = g++

all: libproduct.so

libproduct.so: product.cpp
	$(CC) product.cpp -shared -fpic -Wl,-soname,$@ -o $@ -lpoplar -lpoplin -lpopops

clean:
	rm libproduct.so