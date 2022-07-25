# Graphcore

## CTC Loss

This CTC loss operator enables Connectionist Temporal Classification described in this paper:

[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with 
Recurrent Neural Networks.](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

### How to build the CTC loss library

1.  Prepare the environment.

    Install the Poplar SDK following instructions in the Getting Started guide for your IPU system. 
	Make sure to source the `enable.sh` scripts for poplar and popart.

2.  Setup a virtual environment.

```
virtualenv venv -p python3.6
source venv/bin/activate
```

3.  Build the CTC custom loss library.

```
make all
```