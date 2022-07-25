#!/bin/bash

# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
# This script shows a simple technique
# for distributing rendering over a 16 IPUs.
# However the technique could be scaled to
# many more IPUs.

if [ "$#" -ne 1 ]; then
  echo "Expected one arguments: <output-image>"
  exit -1
fi

if which oidnDenoise; then
  echo "Open Image Denoise found."
else
  echo "Error: 'oidnDenoise' not found."
  echo "You must install Open Image Denoise and source it to use this script (see README.md)."
  exit 1
fi

DIR="distributed_render_output"
mkdir -p $DIR
OUT=$DIR/$1
EXE_NAME=$DIR/distributed_render_example
SAMPLES_PER_DEVICE=2400

CMD=(./build/ipu_trace -w 1920 -h 2160 --tile-width 48 --tile-height 30 --ipus 2 \
--samples-per-step 200 --refractive-index 1.5 --roulette-depth 5 --stop-prob 0.35 \
--exe-name "${EXE_NAME}")

# First we compile a dual IPU path tracer using the --compile-only option
BUILD_CMD=( "${CMD[@]}" )
BUILD_CMD+=(--save-exe --compile-only --codelet-path build --outfile none.png -s 100)
"${BUILD_CMD[@]}"

# Next launch jobs using the compiled graph onto all pairs of devices
# in (e.g.) a POD16. We need a different seed for every job:
SEEDS=(72985 46889 36007 76770 57686 68201 47968 50564 4458 9048 750 535 4131 7377 3693 3040)
for i in {1..8}
do
RUN_CMD=( "${CMD[@]}" )
RUN_CMD+=(--load-exe --outfile "${DIR}/image${i}.png" -s "${SAMPLES_PER_DEVICE}" --save-interval 1000 --seed "${SEEDS[i-1]}")
"${RUN_CMD[@]}" &
done

wait

# Because this renderer uses unbiased Monte Carlo, once the
# final images have been produced we can just average them:
./build/exrtool -i $DIR/image1.png.exr $DIR/image2.png.exr $DIR/image3.png.exr $DIR/image4.png.exr \
$DIR/image5.png.exr $DIR/image6.png.exr $DIR/image7.png.exr $DIR/image8.png.exr \
-o $DIR/dre_combined.exr --op mean

echo "Path tracing complete"

# Finally we apply tone mapping using PFSTools and
# denoising using Intel's Open image denoiser:
pfsin $DIR/dre_combined.exr | pfstmo_mai11 | pfsout $DIR/dre_tmp.pfm
oidnDenoise --hdr $DIR/dre_tmp.pfm  -o $DIR/dre_tmp.pfm

# Convert to output file format using imagemagik:
/usr/bin/convert -verbose $DIR/dre_tmp.pfm $OUT
rm $DIR/dre_tmp.pfm

echo "Rendering complete"
