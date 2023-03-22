CC = g++
PY3 = $(shell python3 -c 'import sys; print("python" + str(sys.version_info.major) + "." + str(sys.version_info.minor))')

all: path_algorithms.so

path_algorithms.so:
	cython -3 path_algorithms.pyx -o path_algorithms.c
	$(CC) path_algorithms.c \
		-shared \
		-fPIC \
		-I/usr/include/$(PY3) \
		-I/usr/local/lib/$(PY3)/dist-packages/numpy/core/include/ \
		-I/usr/lib/$(PY3)/site-packages/numpy/core/include/ \
		-D NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
		-o path_algorithms.so
	rm path_algorithms.c

clean:
	rm path_algorithms.so
