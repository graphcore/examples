# Approximate Bayesian Computation (ABC)
Approximate Bayesian Computation for probabilistic COVID-19 modelling, optimised for Graphcore's IPU.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| TensorFlow 2 | Simulation | ABC | CSSEGIS COVID-19 | Object detection | <p style="text-align: center;">✅ <br> Min. 1 IPU (POD4) required | <p style="text-align: center;">✅ <br> Min. 1 IPU (POD4) required | ['Accelerating Simulation-based Inference with Emerging AI Hardware'](https://ieeexplore.ieee.org/document/9325369) |


## Instructions summary

1. Install and enable the Poplar SDK (see Poplar SDK setup)

2. Install the system and Python requirements (see Environment setup)

3. Download the CSSEGIS covid-19 dataset (See Dataset setup)


## Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
 ```

If no path is provided, then follow these steps:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable.sh
```

More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the TensorFlow 2 and IPU TensorFlow add-ons wheels:
```bash
cd <poplar sdk root dir>
pip3 install tensorflow-2.X.X...<OS_arch>...x86_64.whl
pip3 install ipu_tensorflow_addons-2.X.X...any.whl
```
For the CPU architecture you are running on

4. Build the custom ops:
```bash
cd static_ops && make
```


More detailed instructions on setting up your TensorFlow 2 environment are available in the [TensorFlow 2 quick start guide](https://docs.graphcore.ai/projects/tensorflow2-quick-start).

## Dataset setup
Download the dataset from [the source](https://github.com/CSSEGISandData/COVID-19), or use the script provided:
```bash
python3 covid_data.py
```


## Run the example
To execute the application, you can run for example

```bash
python3 ABC_IPU.py --enqueue-chunk-size 10000 --tolerance 5e5 --n-samples-target 100 --n-samples-per-batch 400000 --country US --samples-filepath US_5e5_100.txt --replication-factor 4
```


## Background
We are looking at data from Italy in the Johns Hopkins University dataset
over a period of 49 days.
Data for the USA and New Zealand is also provided.
The data contains the active confirmed cases (`A`), confirmed recoveries (`R`),
and confirmed deaths(`D`). The data starts
when there are more than 100 recorded cases (2020-02-23 for Italy).

The model additionally considers the number of unconfirmed recoveries (`Ru`),
susceptible people (`S`), and undocumented infected (`I`).
The constant P is the total population record for the country.

All together, we have a state vector:
```
X = [S, I, A, R, D, Ru]
```

We initialize with `S = P - A - R - D - I`. An interesting model parameter
is `k` (kappa) which it the ratio between initial confirmed and unconfirmed cases:
`I[0] = k * A[0]`. Note the index denotes the time step,
that means `I[0]` is the number of undocumented infected people at `t=0`.
The parameter `k` is randomly sampled between 0 and 2 and
the purpose of the ABC algorithm is to find a good fit/estimate.
`Ru[0]` is set to zero.

There are several additional hyperparameters to model the COVID-19 spread
which are all analyzed by the ABC algorithm.
They are managed in a parameter
vector and randomly sampled from uniform distributions with different
boundaries:
```
    param_vector = [alpha_0, alpha, n, beta, gamma, delta, eta, kappa]
    upper_bound = [1.0, 100.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0]
```
The first three are used to model the active transmission rate function::
```
     g = alpha_0 + alpha/(1+(U)^n)
```
where `U = A + R + D`. So, `n` controls the rate of decrease,  `alpha` is a scale factor, and
`alpha_0` is the initial offset.
Different utility functions `U` could be chosen to model different
non-pharmaceutical interventions specific for policies in each country.

`gamma` is the case identification rate.
`beta` is the case recovery rate with the relative latent recovery rate `eta`.
`delta` is the case fatality rate.

For `S`, `g * S * I/P` cases are expected to transition to undocumented infected,
assuming that active confirmed cases (`A`) do not infect others
and that recovered people (`R + Ru`) cannot get reinfected.
The more unidentified infected people (`I`) the more people get infected.
Analyzing the change of `g` over time helps to understand certain policies.

With the identification rate, we get that `gamma * I` people change from
infected (`I`) to active confirmed cases (`A`).
With the recovery rate, we get that `beta * A`
identified infected people recover (`R`)
and together with the relative latent recovery rate,
we get that `beta * eta * I` people become unidentified recovered people (`Ru`).
Last, but not least, `delta * A`
is the number of active confirmed cases dying from COVID-19 (`D`).

Together, the rates are used to calculate the hazard function which
represents the expected change in cases on average.
To map the average numbers to real simulated data,
the average rates are used in a Poisson sampling.
The respective sampled data is then used to adjust the estimated numbers from
one day to the other.
In our code, we simulate the Poisson distribution with a Normal distribution
with same mean `h` and a standard deviation of `sqrt(h)`.
Since the values in `h` are sufficiently big, this is a good approximation.

See section A.1 of
[Hindsight is 2020 vision: a characterisation of the global response to the COVID-19 pandemic](https://link.springer.com/article/10.1186/s12889-020-09972-z)
for further details of the model and its parameters.

David J. Warne

- School of Mathematical Sciences, Faculty of Science, Queensland University of Technology, Brisbane, Australia
- Centre for Data Science, Queensland University of Technology, Brisbane, Australia
- ARC Centre of Excellence for Mathematical and Statistical Frontiers.
