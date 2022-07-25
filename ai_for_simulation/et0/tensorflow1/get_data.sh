# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

weather_data="dummy_data/dummy_raw_data"
file_dem="dummy_data/dummy_resource"
file_latit="dummy_data/dummy_latitude"
so_file="matrix_solve/matrix_solve_ce_op/libmatrix_solve_ce_op.so"
gp_file="matrix_solve/matrix_solve_ce_op/matrix_solve_codelets.gp"
save_folder="test_result"

git clone https://github.com/WangQiangItachi/dummy_data.git

if [ ! -x "$weather_data" ] || [ ! -x "$file_dem" ] || [ ! -x "$file_latit" ]; then
  cd dummy_data
  unzip dummy_data.zip
  cd ../
fi

if [ ! -f "$so_file" ] || [ ! -f "$gp_file" ]; then
  cd matrix_solve/matrix_solve_ce_op
  make
  cd ../../
fi

if [ ! -x "$save_folder" ]; then
  mkdir "$save_folder"
fi

echo "Unpacking dummy data of et0 and make the custom operaton"
