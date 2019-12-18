CC = g++

all: example

codelets.gp: codelets.cpp
	popc codelets.cpp -o codelets.gp

example: utils.h main.cpp codelets.gp
	$(CC) main.cpp -o example -lpoplar -lpopops -lboost_program_options

clean:
	rm codelets.gp example
