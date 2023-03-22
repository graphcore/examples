CXX ?= g++
all: tut5 tut5_ipu

tut5: tut5_complete.cpp
	$(CXX) -std=c++11 tut5_complete.cpp -lpoplar -lpoputil -o tut5

tut5_ipu: tut5_ipu_hardware_complete.cpp
	$(CXX) -std=c++11 tut5_ipu_hardware_complete.cpp -lpoplar -lpoputil -o tut5_ipu

clean:
	rm -f tut5
	rm -f tut5_ipu
