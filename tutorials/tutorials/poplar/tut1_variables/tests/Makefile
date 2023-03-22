CC = g++

all: tut1_ipu_model_complete tut1_ipu_hardware_complete

tut1_ipu_model_complete: ../complete/tut1_ipu_model_complete.cpp
	$(CC) --std=c++11 ../complete/tut1_ipu_model_complete.cpp -lpoplar -o tut1_ipu_model_complete

tut1_ipu_hardware_complete: ../complete/tut1_ipu_hardware_complete.cpp
	$(CC) --std=c++11 ../complete/tut1_ipu_hardware_complete.cpp -lpoplar -o tut1_ipu_hardware_complete

clean:
	rm -f tut1_ipu_model_complete tut1_ipu_hardware_complete
