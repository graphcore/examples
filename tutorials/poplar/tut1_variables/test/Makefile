CC = g++

all: tut1_ipu_model_complete tut1_ipu_hardware_complete tut1_start_here

tut1_start_here:
	$(CC) --std=c++11 ../start_here/tut1.cpp -lpoplar -o tut1_start_here

tut1_ipu_model_complete:
	$(CC) --std=c++11 ../complete/tut1_ipu_model_complete.cpp -lpoplar -o tut1_ipu_model_complete

tut1_ipu_hardware_complete:
	$(CC) --std=c++11 ../complete/tut1_ipu_hardware_complete.cpp -lpoplar -o tut1_ipu_hardware_complete

clean:
	rm -f tut1_ipu_model_complete tut1_ipu_hardware_complete tut1_start_here