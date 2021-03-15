
# Benchmarking on IPUs

This README describes how to run the LSTM kernel throughput benchmark on the Mk2 IPU.

## TensorFlow kernel benchmarks

Follow the installation instructions in code_examples/tensorflow/kernel_benchmarks/README.md

Run the following command line from inside the code_examples/tensorflow/kernel_benchmarks directory.

## LSTM 2-layer

1 x IPU

```
python3 lstm_multi.py --batch-size 1 --use-generated-data --input-size 16 --num-layers 2 --hidden-size 256 --timesteps 200
```

Change the --batch-size argument to be one of 1, 2, 4, 8, 16, 32, 64, 128, 256, 512

Latency in ms is calculated as 1000 * batch_size / throughput
where throughput is in items/second.




