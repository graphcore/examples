# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


CC = g++ -std=c++11

all: libmatrix_solve_ce_op.so

matrix_solve_codelets.gp: matrix_solve_codelets.cpp
	popc --target=ipu1,ipu2 -O2 matrix_solve_codelets.cpp -o $@

libmatrix_solve_ce_op.so: matrix_solve_codelets.gp main.cpp
	$(CC) -O3 main.cpp -shared -fpic -Wl,-soname,$@ -o $@ -lpoplar -lpoputil

clean:
	rm libmatrix_solve_ce_op.so matrix_solve_codelets.gp
