CC = g++

all: codelets.gp example

codelets.gp: codelets.cpp
	popc codelets.cpp -o codelets.gp

example: utils.h main.cpp
	$(CC) --std=c++11 main.cpp -o example -lpoplar -lpopops -lboost_program_options

clean:
	rm codelets.gp example
