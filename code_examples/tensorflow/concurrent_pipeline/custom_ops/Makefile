CC = g++ -std=c++17 -O2 -Werror -Wno-error=deprecated-declarations

POPLAR_LIBS = -lpoplar -lpoputil -lpopops -lpoplin -lpopnn -lgcl_ct
OBJECTS = utils.o common.o sharded_matmul.o sharded_losses.o sharded_utils.o sharded_embedding.o

all: libconcurrent_ops.so

lib%.so: $(OBJECTS)
	$(CC) $(OBJECTS) -shared -fpic -Wl,-soname,$@ -o $@ $(POPLAR_LIBS)

# Build all the object files that have matching cpp and hpp file names:
%.o: %.cpp %.hpp
	$(CC) $< -c -fpic -o $@

.PHONY : clean
clean:
	$(RM) *.so
	$(RM) $(OBJECTS)
