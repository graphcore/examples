<!-- Copyright (c) 2022 Graphcore Ltd. All rights reserved. -->
# VSCode setup

In this page we explain how to configure your workspace in VSCode to use
IntelliSense and the visual debugger with a PopXL application.
For a more general and comprehensive guide see
[Using VS Code with the Poplar SDK and IPUs](../standard_tools/using_vscode/README.md).

We assume you have downloaded the Poplar SDK and have sourced the `enable.sh`
scripts for both PopART and Poplar as described in the [Getting Started
guide](https://docs.graphcore.ai/en/latest/getting-started.html) for your IPU
system.

We also assume you have installed popxl.addons inside a virtual environment.
popxl.addons is a requirement for all PopXL tutorials, so instructions on
installing it are given in the `README.md` for each PopXL tutorial.

```bash
python3 -m venv virtual_env
. virtual_env/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## IntelliSense

- Add the folder you are working on to your VSCode workspace. You can do this
  by opening the Command Palette with Cmd+Shift+P (or Ctrl+Shift+P on Windows),
  and then typing `Workspaces: Add Folder to Workspace...`; alternatively, you
  can directly edit your `.code-workspace` file.
- To work with popxl.addons, you need to specify the interpreter path for the
  folder using the `Python: Select Interpreter` command and selecting the
  interpreter of the virtual environment where you installed popxl.addons,
  located at `<virtual_env_path>/bin/python3`
- Create a `settings.json` file for your folder. If you type `Preferences: Open
  Workspace Settings (JSON)`, the file will be created for you inside a `.vscode`
  folder. Otherwise you can create the folder and the file manually. This file
  allows you to specify general settings of the folder. Note: you might be
  tempted to specify the `python.defaultInterpreterPath` here, but you may still
  run into problems since the default interpreter path is not used once an
  interpreter has been selected. You can find more information on the
  [VSCode Python wiki](https://github.com/microsoft/vscode-python/wiki/AB-Experiments#tldr).
- To work with PopXL you need to include PopART relevant paths in your
  `settings.json` file, adding them to `"python.autoComplete.extraPaths"` and
  `"python.analysis.extraPaths"` options.

Below is a template for `settings.json`:

```json
{
    "python.autoComplete.extraPaths": [
        "<path_to_sdk>/poplar_sdk-<platform>-<version>/popart-<platform>-<version>/python",
    ],
    "python.analysis.extraPaths":  [
        "<path_to_sdk>/poplar_sdk-<platform>-<version>/popart-<platform>-<version>/python",
    ],
}
```

## Debugging

The easiest way to use the visual debugger is by creating an attach
configuration and then running the debugger from the command line.

- Create a simple attach configuration in `.vscode/launch.json` inside your workspace folder

```json
{
    "configurations": [
    {
        "name": "Attach_debugpy",
        "type": "python",
        "request": "attach",
        "connect": {
            "host": "localhost",
            "port": 7000
        },
    }
    ]
}
```

- Install the [`debugpy`](https://pypi.org/project/debugpy/) package in your virtual environment:

```bash
pip3 install --upgrade debugpy
```

- Run the script with the debugger. The `--wait-for-client` option prevents the
  script from running until you attach to the process.

```bash
python3 -m debugpy --wait-for-client --listen 7000 mnist_template.py
```

- From the *Run and Debug* pane (click the play symbol on the left bar to open
  the *Run and Debug* pane), launch your `Attach_debugpy` configuration.
