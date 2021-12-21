CC = g++
LDLIBS = -shared -lpopart -lpoplar -lpopops -lpoplin

all: liblengthRegulator.so

lengthRegulator.gp: lengthRegulatorCodelets.cpp
	popc -O2 lengthRegulatorCodelets.cpp -o $@

liblengthRegulator.so: lengthRegulator.gp poplar_code.cpp
	$(CC) poplar_code.cpp $(LDLIBS) -shared -fpic -Wl,-soname,$@ -o $@ -lpoplar -lpoputil

clean:
	rm liblengthRegulator.so lengthRegulator.gp
