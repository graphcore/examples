# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import subprocess
import re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# regex that matches a link
LINK_REGEX = r"(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?!&//=]*))"
# command to collect possible link candidates in readme and python files
GREP_CMD = "grep -REinH '(http|https)://' --include *.py --include=*.md"
# Large files, slow response or not fitting into the pattern.
EXCLUDED_LINKS = {"https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip",
                  "https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block",
                  "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-litex.tar.gz",
                  "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-bx.tar.gz",
                  "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/",
                  "https://github.com/graphcore/examples/tree/master/nlp/bert/pytorch/",
                  "http://www.cmake.org"}
EXCLUDED_PATTERNS = {
    # popxl-addons is not publically available yet.
    r"https:\/\/github\.com\/graphcore\/popxl-addons.*"
}


def collect_links():
    location = Path(__file__).absolute().parent.parent.parent.parent
    grep_result = str(subprocess.check_output(GREP_CMD, cwd=location, shell=True, stderr=subprocess.STDOUT))
    extracted_links = re.findall(LINK_REGEX, grep_result)
    cleaned_links = [link_elements[0] for link_elements in extracted_links]
    for idx, visitable_link in enumerate(cleaned_links):
        cleaned_links[idx] = visitable_link.rstrip(").:")  # possible endings of the links, which should be removed
    return set(cleaned_links)  # remove duplicates


def check_single_url(url):
    try:
        result = requests.head(url, timeout=10, allow_redirects=True)
        if result.status_code != 404:
            return True
        else:
            result = requests.get(url, timeout=10)  # Try with get
            if result.status_code != 404:
                return True
            else:
                return False
    except Exception:
        return False


def check_links(link_list):
    wrong_links = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for available, link in zip(list(executor.map(check_single_url, link_list)), link_list):
            if not available:
                wrong_links.append(link)
    return wrong_links


def test_links():
    link_list = collect_links() - EXCLUDED_LINKS
    for pattern in EXCLUDED_PATTERNS:
        m = re.compile(pattern)
        link_list = {link for link in link_list if not m.match(link)}
    wrong_links = check_links(link_list)
    assert len(wrong_links) == 0, f"The following links {wrong_links} are not working."
