CC = g++ -std=c++17 -O2 -Werror

PYINC := $(shell python3 -m pybind11 --includes)
PYSO := $(shell python3-config --extension-suffix)
POPLAR_LIBS = -lpoplar -lpoputil -lpopops -lpoplin -lpopsparse -lpopnn
OBJECTS = utils.o
STATIC_BLOCK_SPARSE_DIR = ../../../../sparsity/block_sparse/tensorflow1/custom_ops
STATIC_BLOCK_SPARSE_SRC := $(wildcard $(STATIC_BLOCK_SPARSE_DIR)/bs*.cpp)
STATIC_BLOCK_SPARSE_INC := $(wildcard $(STATIC_BLOCK_SPARSE_DIR)/bs*.hpp)

all: libsparse_matmul.so libsparse_embedding.so host_utils$(PYSO) tests/test_matmul_serialization libstatic_block_sparse.so

# Build all the object files for files which have
# matching cpp and hpp file names:
%.o: %.cpp %.hpp
	$(CC) $< -c -fpic -o $@ $(POPLAR_LIBS)

# Build the host utilities pybind module:
host_utils$(PYSO): host_utils.cpp $(OBJECTS)
	$(CC) host_utils.cpp $(OBJECTS) -shared -fpic -Wl,-soname,$@ $(PYINC) \
	-o host_utils$(PYSO) $(POPLAR_LIBS)

# Build the tensorflow custom ops listed under all:
lib%.so: %.cpp $(OBJECTS)
	$(CC) $< $(OBJECTS) -shared -fpic -Wl,-soname,$@ -o $@ $(POPLAR_LIBS)

# Build the serialization test program:
tests/test_matmul_serialization: tools/matmul_serialization.cpp $(OBJECTS)
	$(CC) tools/matmul_serialization.cpp $(OBJECTS) -o tests/test_matmul_serialization $(POPLAR_LIBS)

# Build the the static block-sparse library used in block-sparse autoregressive attention:
libstatic_block_sparse.so: $(STATIC_BLOCK_SPARSE_SRC) $(STATIC_BLOCK_SPARSE_INC)
	$(CC) $(STATIC_BLOCK_SPARSE_SRC) -shared -fpic $(POPLAR_LIBS) -o $@ 

.PHONY : clean
clean:
	$(RM) *.so
	$(RM) $(OBJECTS)
