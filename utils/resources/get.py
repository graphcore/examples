#!/usr/bin/env python
# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

"""
This script will download a resource file to go with a demo.
"""
import sys
import argparse
import shutil
try:
    from urllib.request import urlopen  # Python 3
except ImportError:
    from urllib2 import urlopen  # Python 2

URL = "https://s3-us-west-1.amazonaws.com/gc-demo-resources/"
CHUNK = 16*1024


def download(name, dst=None):
    """Function to perform a download"""
    if dst is None:
        dst = name
    response = urlopen(URL + name)  # Will throw a HTTPError if response code is not 200
    with open(name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file, CHUNK)
        data = response.read()
        out_file.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download a resource')
    parser.add_argument('resource_name', metavar='resource_name', type=str,
                        help='The name of the resource file to download')
    args = parser.parse_args()
    name = args.resource_name
    print("Fetching {}".format(name))
    download(name)
