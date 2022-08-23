# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import json
import multiprocessing
from tqdm import tqdm
from typing import List

openwebtext_dir = sys.argv[1]
extract_dir = sys.argv[2]


def process_single_file(filepath: str, writer):
    with open(filepath, "r") as fr:
        fr = fr.read()
    url = os.path.basename(filepath).split(".")[0]
    json_line = json.dumps({"text": fr, "url": url}, ensure_ascii=False)
    writer.write(json_line + "\n")


def extract(filepaths: List[str]):
    for xz_name in tqdm(filepaths):
        xz_path = os.path.join(openwebtext_dir, xz_name)
        extract_path = os.path.join(extract_dir, xz_name.split(".")[0])
        os.makedirs(extract_path)
        os.system("tar -xf {} -C {}".format(xz_path, extract_path))


if __name__ == "__main__":
    os.makedirs(extract_dir)
    print("extracting files...")
    xz_files = os.listdir(openwebtext_dir)
    p = multiprocessing.Pool(8)
    for i in range(0, len(xz_files), 1000):
        p.apply_async(func=extract, args=(xz_files[i:i+1000],))
    p.close()
    p.join()
    print("merging files...")
    fw = open("openwebtext_raw.json", "w", encoding="utf-8")
    for dirname in tqdm(os.listdir(extract_dir)):
        dirpath = os.path.join(extract_dir, dirname)
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            process_single_file(filepath, fw)
    fw.close()
