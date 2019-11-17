
# Graphcore
---
## Generate a Poplar report from a Tensorflow Graph

### Run the example

See `report-generation-example.py` script for a simple example. In order to run it, you'll first need to install the poplar-sdk (>=0.8.0) following the README provided. Make sure to run the enable.sh scripts and activate a Python virtualenv with gc_tensorflow installed. Then:

`python report-generation-example.py`

If everything has been set up correctly, a report.txt file should have been created in the same directory.

### Notes

The generated report.txt file contains two different reports: one for the variables initializer graph and one for the actual graph (in this example, a fully connected layer). It's very likely that you might only be interested in the second report. In order to write on file just that one, uncomment the following command right after `sess.run(tf.global_variables_initializer())`:

`sess.run(report)`

If you get an out of memory error while trying to generate a report for your graph check the `OutOfMemory_guide.md` in troubleshooting folder.
