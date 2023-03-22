CC = g++

all: example_1 example_2 example_3 example_3_a

example_1: ../examples/example_1.cpp
	$(CC) --std=c++14 $(realpath ..)/examples/example_1.cpp -lpoplar -lpva -o example_1

example_2: ../examples/example_2.cpp
	$(CC) --std=c++14 $(realpath ..)/examples/example_2.cpp -lpoplar -lpva -o example_2

example_3: ../examples/example_3.cpp
	$(CC) --std=c++14 $(realpath ..)/examples/example_3.cpp -lpoplar -lpva -o example_3

example_3_a: ../examples/example_3_a.cpp
	$(CC) --std=c++14 $(realpath ..)/examples/example_3_a.cpp -lpoplar -lpva -o example_3_a

clean:
	rm -f example_1 example_2 example_3 example_3_a
