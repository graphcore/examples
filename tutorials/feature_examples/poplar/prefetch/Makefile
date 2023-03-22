all: prefetch

%: %.cpp
	g++ -std=c++11 $^ -lpoplar -o $@

clean:
	rm -f prefetch
