## Graphcore benchmarks: MCMC with TFP

This README describes how to run the TensorFlow MCMC sampling examples with TFP model for throughput benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in code_examples/tensorflow/mcmc/README.md

## Training

Run the following command line from inside the code_examples/tensorflow/mcmc directory.

If you wish to share the dataset with other users, move the file
`returns_and_features_for_mcmc.txt` to `$DATASETS_DIR/mcmc/` after downloading it.
Otherwise you can omit the second argument.

### MCMC TFP

#### 1 x IPU-M2000

Command:
```console
python3 mcmc_tfp.py --num_ipus=4 --dataset_dir $DATASETS_DIR/mcmc/
```
