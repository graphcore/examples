CC = g++ -std=c++17 -O2

PYINC := $(shell python -m pybind11 --includes)
PYSO := $(shell python-config --extension-suffix)
POPLAR_LIBS = -lpoplar -lpoputil -lpopops -lpoplin -lpopsparse -lpopnn

all: libsparse_matmul.so host_utils$(PYSO)

host_utils$(PYSO): host_utils.cpp utils.hpp
	$(CC) host_utils.cpp -shared -fpic -Wl,-soname,$@ $(PYINC) \
	-o host_utils$(PYSO) $(POPLAR_LIBS)

lib%.so: %.cpp utils.hpp
	$(CC) $< -shared -fpic -Wl,-soname,$@ -o $@ $(POPLAR_LIBS)

.PHONY : clean
clean:
	rm *.so
