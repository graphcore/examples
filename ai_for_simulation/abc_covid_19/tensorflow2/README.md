# Probabilistic COVID-19 modelling with Approximate Bayesian Computation (ABC)

## Overview

This is a representative implementation 
of Approximate Bayesian Computation (ABC)
for Simulation-based Inference (SBI).
A detailed explanation is provided in the background section and in
"Accelerating Simulation-based Inference with Emerging AI Hardware",
S Kulkarni, A Tsyplikhin, MM Krell, and CA Moritz, 
IEEE International Conference on Rebooting Computing (ICRC), 2020.
In its core, the algorithm determines simulation parameters that best
describe the observed data from COVID-19 infections 
to enable statistical inference.

## Dataset

The data is provided in the script `covid_data.py`.
It was extracted from the 
[JHU CSSE COVID-19 Data](https://github.com/CSSEGISandData/COVID-19).
It contains the COVID-19 numbers (detected active cases, 
recovered cases, and detected cases that lead to death) 
after the first day with 100 known infections.

See also: 
Dong E, Du H, Gardner L. An interactive web-based dashboard
to track COVID-19 in real time. Lancet Inf Dis. 20(5):533-534.
doi: 10.1016/S1473-3099(20)30120-1

## File Structure

| File                         | Description                                |
| ---------------------------- | ------------------------------------------ |
| `README.md`                  | How to run the model                       |
| `ABC_IPU.py`                 | Main algorithm script to run IPU model     |
| `argparser.py`               | Document and read command line parameters  |
| `covid_data.py`              | Provide already downloaded COVID-19 data   |
| `requirements.txt`           | Required Python 3 packages                 |
| `test_ABC.py`                | Test script. Run using `python -m pytest`  |

## Quick start guide

### 1) Download the Poplar SDK

Install the Poplar SDK following the instructions in the 
Getting Started guide for your IPU system 
which can be found here: 
https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html.
Make sure to source the `enable.sh` script for Poplar as well as the drivers.

### 2) Package installation

Make sure that the virtualenv package is installed for Python 3.

### 3) Prepare the TensorFlow environment

Activate a Python3 virtual environment with the `tensorflow`
wheel version 2.4 included in the SDK as follows:

```
python3 -m venv venv
source venv/bin/activate
pip install <path to tensorflow_2.4.whl>
pip install -r requirements.txt
```

### 4) Execution

To execute the application, you can run for example

```
python ABC_IPU.py --enqueue-chunk-size 10000 --tolerance 5e5 --n-samples-target 100 --n-samples-per-batch 400000 --country US --samples-filepath US_5e5_100.txt --replication-factor 4
```

All command line options and defaults are explained in `argparser.py`.

## Background
We are looking at data from Italy in the Johns Hopkins University dataset
over a period of 49 days. 
Data for the USA and New Zealand is also provided.
The data contains the active confirmed cases (`A`), confirmed recoveries (`R`),
and confirmed deaths(`D`). The data starts 
when there is more than 100 recorded cases (2020-02-23 for Italy).

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