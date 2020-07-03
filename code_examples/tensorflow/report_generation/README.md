# Graphcore

---
## Generate a Poplar report from a Tensorflow Model

Poplar reports are useful for profiling models running on IPUs. 
There are two types of report: compile and execution.
This simple example shows how to generate Poplar reports from TensorFlow.

### File structure

* `report_generation_example.py` Main python script.
* `README.md` This file.
* `test_report_generation.py` Script for testing this example.

### Quick start guide

#### Prepare the environment

##### 1) Download the Poplar SDK

Install the `poplar-sdk` following the README provided. 
Make sure to source the `enable.sh`
scripts for poplar, gc_drivers (if running on hardware) and popART.

##### 2) Python

Create a virtualenv and install the required packages:

```
bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
pip install <path to gc_tensorflow.whl>
```

##### 3) Run the script.

   `python report_generation_example.py`

##### 4) Retrieve the reports generated.

   With default arguments, a report.txt file should be created in this directory.

### Notes

In TensorFlow, every time `session.run(...)` is called 
to evaluate (part of) a graph that hasn't been evaluated before, 
a new Poplar graph is compiled and executed.
If variables are initialised on the IPU, 
then evaluating the variable initialisation graph i.e. 
`session.run(tf.global_variables_initializer())` 
will generate a new Poplar graph.
When generating the report, 
this variable initialisation will be profiled and included in the report, 
alongside the profiling of any other executed graphs.
In practise, the variable initialisation part is rarely useful. 
There are several ways to profile only the relevant graph, 
as shown in this example. 
Use the `-h` flag to check the options, 
and examine the code to see what each does.

### Known Issues

Some options like `--split-reports` will fail
if the `--executable_cache_path` option in the
`TF_POPLAR_FLAGS` environment variable is used. 
