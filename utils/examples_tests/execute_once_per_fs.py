# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import functools
import os
import time


class ExecuteOncePerFS:
    """Adds synchronization to the execution of a function so it only executes
    once per file-system."""

    def __init__(self, lockfile, file_list, exe_list=[], timeout=60, retries=10):
        self.lockfile = lockfile
        self.file_list = file_list
        self.exe_list = exe_list
        self.timeout = timeout
        self.retries = retries

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # Race to become master process
            result = None
            try:
                with open(self.lockfile, "x"):
                    # Master process executes function
                    result = fn(*args, **kwargs)
            except FileExistsError:
                pass

            # Every process waits for files to be created
            attempts = 0
            sleep_time = self.timeout / self.retries
            remaining_files = self.file_list[:]
            remaining_exes = self.exe_list[:]
            while attempts < self.retries:
                remaining_files = [path for path in remaining_files if not os.path.exists(path)]
                remaining_exes = [path for path in remaining_exes if not os.access(path, os.R_OK | os.X_OK)]
                if len(remaining_files) == 0 and len(remaining_exes) == 0:
                    return result

                time.sleep(sleep_time)
                attempts += 1

            # If we are here it means that we timed out...
            raise RuntimeError(
                f"Timed out waiting for {remaining_files} to be made and/or {remaining_exes} to become executable."
            )

        return wrapped
