CC = g++

all: custom_grouped_gather_scatter.so

custom_grouped_gather_scatter.so: custom_grouped_gather_scatter.cpp
	$(CC) -std=c++17 -fpic -Wall -Wextra -Werror -g -O2 \
		$< \
		-shared -lpoplar -lpopops -lpoputil \
		-o $@

clean:
	rm custom_grouped_gather_scatter.so
