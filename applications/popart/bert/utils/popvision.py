# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Utilities to work with Graphcore's PopVision Graph Analyser'''
import os
import json
import logging

logger = logging.getLogger(__name__)


def get_profile_directory():
    engine_options = json.loads(os.environ.get("POPLAR_ENGINE_OPTIONS", "{}"))
    return engine_options.get("autoReport.directory", None)


def get_profile_logging_handler(append=False):
    '''Gets a python logging handler for saving log output to <profile-dir>/app_log.txt'''
    directory = get_profile_directory()
    if directory is None:
        raise RuntimeError("Poplar Engine Option 'autoReport.directory' was not set. Please specify using '--profile-dir' or as an environment variable.")
    os.makedirs(directory, exist_ok=True)
    mode = 'w'
    if append:
        mode = 'a'
    profile_handler = logging.FileHandler(os.path.join(directory, "app_log.txt"), mode)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    profile_handler.setFormatter(formatter)
    return profile_handler


def set_profiling_vars(path: str = None, instrument: bool = True):
    '''Set environment variables to profile a program. autoReport.directory will be set to path'''
    popvision_options = {
        "autoReport.all": "true",
        "autoReport.directory": path,
        # The serialized graph will always be quite large for Bert. Better to not generate it.
        "autoReport.outputSerializedGraph": "false",
        "debug.allowOutOfMemory": "true",
        "debug.outputAllSymbols": "true",
        "debug.instrument": str(instrument).lower(),
        # The reports are typically very large. This compresses them where possible.
        "profiler.format": "experimental"
    }
    existing = json.loads(os.environ.get("POPLAR_ENGINE_OPTIONS", "{}"))
    popvision_options.update(**existing)

    if popvision_options["autoReport.directory"] is None:
        raise RuntimeError("Poplar Engine Option 'autoReport.directory' was not set. Please specify using '--profile-dir' or as an environment variable.")

    os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(popvision_options)


def set_logging_vars():
    '''Set environment variable to capture stdout, stderr, poplar, popart logging into autoReport.directory'''
    directory = get_profile_directory()
    if directory is None:
        logger.warning("Poplar Engine Option 'autoReport.directory' is not set. Skipping log capture.")
        return

    loggers = {
        "poplar": "TRACE",
        "poplibs": "TRACE",
        "popart": "TRACE"
    }

    for lib, level in loggers.items():
        level_str = lib.upper() + "_LOG_LEVEL"
        os.environ[level_str] = os.environ.get(level_str, level)

        dest_str = lib.upper() + "_LOG_DEST"
        dest_path = os.path.join(directory, lib + "_log.txt")
        os.environ[dest_str] = os.environ.get(dest_str, dest_path)


def save_app_info(args):
    '''Takes a dictionary and saves it to <autoReport.directory>/app.json.
        If this file already exists, it will be updated.'''
    directory = get_profile_directory()
    if directory is None:
        logger.warning("Poplar Engine Option 'autoReport.directory' is not set. Skipping app.json save.")
        return

    os.makedirs(directory, exist_ok=True)

    app_file = os.path.join(directory, "app.json")

    app_dict = {}
    if os.path.exists(app_file):
        with open(app_file, "r") as f:
            app_dict = json.load(f)

    app_dict.update(**args)

    with open(app_file, "w") as f:
        json.dump(app_dict, f)
