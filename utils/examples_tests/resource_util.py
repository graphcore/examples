# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import os
import subprocess
import six
import sys
from contextlib import contextmanager


def fetch_resources(script_name, test_file, cwd):
    if not os.path.isfile(test_file):
        print("Fetching resource files")
        if not os.path.isfile(os.path.join(cwd, script_name)):
            raise Exception('Unable to find ' + script_name)
        # The script may contain relative paths, therefore we
        # must set use the cwd passed in
        subprocess.check_call([os.path.join(cwd, script_name)], cwd=cwd)


@contextmanager
def captured_output():
    """Return a context manager that temporarily replaces the sys stream stdout
       with a StringIO
    """
    new_out = six.StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield sys.stdout
    finally:
        sys.stdout = old_out
