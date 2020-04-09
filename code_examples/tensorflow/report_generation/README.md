# Graphcore

---
## Generate a Poplar report from a Tensorflow Model

Poplar reports are useful for profiling models running on IPUs. There are two types of report: compile and execution.
This simple example shows how to generate Poplar reports from TensorFlow.

### File structure

* `report_generation_example.py` Main python script.
* `README.md` This file.
* `test_report_generation.py` Script for testing this example.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the poplar-sdk following the README provided. Make sure to run the enable.sh scripts and activate a Python virtualenv with gc_tensorflow installed.

2) Run the script.

   `python report_generation_example.py`

3) Retrieve the reports generated.

   With default arguments, a report.txt file should be created in this directory.

### Notes

In TensorFlow, every time `session.run(...)` is called to evaluate (part of) a graph that hasn't been evaluated before, a new Poplar graph is compiled and executed.
If variables are initialised on the IPU, then evaluating the variable initialisation graph i.e. `session.run(tf.global_variables_initializer())` will generate a new Poplar graph.
When generating the report, this variable initialisation will be profiled and included in the report, alongside the profiling of any other executed graphs.
In practise, the variable initialisation part is rarely useful. There are several ways to profile only the relevant graph, as shown in this example. Use the `-h` flag to check the options, and examine the code to see what each does.
