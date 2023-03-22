CXX ?= c++

all: regression-demo

regression-demo: regression-demo.cpp mnist.cpp mnist.h
	$(CXX) -std=c++11 regression-demo.cpp mnist.cpp -lpopnn -lpoplin -lpopops -lpoputil -lpoplar -o regression-demo

clean:
	rm -f regression-demo
