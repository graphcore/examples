# Graphcore code examples - contribution guide

The guide contains some quick notes to adding example code to this repository.

## Put the example in the correct place

The code should either go as a new folder in ``applications/[framework]`` or
``code_examples/[framework]`` depending on whether it is a full application
example or a small piece of example code to show people how to develop for the
IPU.

There is no common enforced naming scheme below the framework level. Just make
the name and sub-folder structure clear for users.

## Ensure the example has a README.md file

The README should explain what the example is and who to run it.

## Add unit tests

New code should have unit tests added by adding files called ``test_*.py`` to
code example folder. This folder then needs to be added to the list in the
top-level `pytest.ini`. See other tests in the repo to see common usage and
the use of the modules in `utils/tests`.


