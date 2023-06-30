# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

"""Test script to check hyperlinks work; external (http), internal (relative
files) and page (references to section headings)."""

from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from typing import Container, Iterable, List, Optional, Tuple
import pytest
import re

import requests

from link_parsers import get_parser

# URLs which don't exist yet (e.g documentation for a future release) can be
# added to the list of exceptions below.
#
# Make sure to add a TODO(TXXXX) comment to remove the exception once the link
# is fixed.
EXCEPTIONS: List[str] = [
    "https://github.com/graphcore/popxl-addons/blob/master/popxl_addons/ops/group_quantize_decompress",  # not public until SDK 3.3. TODO: remove on SDK 3.3 release
    "https://github.com/graphcore/paperspace-automation",  # private repo
    "box.com",  # dropbox cannot be checked
    "marketplace.visualstudio.com",  # visual studio marketplace cannot be checked
    r"\.git@",  # git repo for pip not a navigable URL
    r"\${.*}?",  # contains bash substitution
    "…",  # parser can't handle ellipsis at end of URL
]

UNPINNED_PATTERNS = ["tree/master", "en/latest"]
# URLs which are un-pinned and don't correspond to an SDK release
PINNING_EXCEPTIONS = [
    "https://docs.graphcore.ai/en/latest/getting-started.html",
    "https://docs.graphcore.ai/en/latest/hardware.html",
    "https://docs.graphcore.ai/en/latest/software.html",
    "https://docs.graphcore.ai/projects/graphcore-glossary/en/latest/index.html",
    "https://docs.pytest.org/en/latest/cache.html",
]

root_path = Path(__file__).parents[3]
suffixes = {".md", ".rst", ".ipynb"}
file_list = [str(p.relative_to(root_path)) for p in root_path.rglob("*") if p.suffix in suffixes]
checked_urls = {}


def check_url_works(url: str) -> Optional[Tuple[str, Path, str, int]]:
    """Check given `url` is responsive. If an error occurs return the error
    response string and code. Return `None` if good.
    """
    global checked_urls
    try:
        response = checked_urls.get(url, requests.head(url, allow_redirects=True) or requests.get(url))
        print(f"\t{url} -> {response.reason} ({response.status_code})")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Allow the test to succeed with intermittent issues.
        # (TooManyRedirects is not caught as could be a broken url.)
        print(f"\t{url} -> ConnectionError/Timeout")
        return None

    if response.status_code in (403, 405, 406, 418):
        """Details from https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
        403 - the server understands the request but refuses to authorize it
        405 - the server knows the request method, but the target resource doesn't support this method
        406 - the server cannot produce a response matching the list of acceptable values defined in the request's
            proactive content negotiation headers
        418 - the server refuses the attempt to brew coffee with a teapot.
        """
        return None

    # Report only client error responses (400-499) because server error
    # responses (500-599) are likely temporary and unnecessarily cause CI to fail
    if 400 <= response.status_code and response.status_code < 500:
        return response

    return None


def check_page_links(anchors: Container[str], links: Iterable[str]) -> List[str]:
    """Check links with same file. Return list of failed links."""

    # Links appear to be case insensitive, so `lower`
    return [f"\t{link} -> NON-EXISTENT" for link in links if link.lower() not in anchors]


def check_file_links(file_path: Path, links: Iterable[str]) -> List[str]:
    """Checks given list of file links are all valid relative to given filename.
    Returns list of failed links.
    """
    failed_paths = []

    for link in links:
        if "mailto:support@graphcore.ai" in link:
            print(f"SKIPPING EMAIL: {link}")
            continue

        link_target = file_path.parent / link
        if link_target.exists():
            print(f"\t{link_target} -> EXISTS")
        else:
            print(f"\t{link_target} -> NON-EXISTENT")
            failed_paths.append(f"\t{link_target} -> NON-EXISTENT")

    return failed_paths


@pytest.mark.parametrize("file_path", file_list)
def test_links_are_pinned(file_path: str) -> None:
    """pytest to test links are pinned at a version"""
    file_path = Path(file_path)
    parser = get_parser(file_path)
    with open(file_path) as f:
        lines = f.readlines()
    links = parser.get_all_links(lines)
    external_links = parser.select_external_links(links)

    failed_urls: List[str] = []
    for external_link in external_links:
        if any(external_link.startswith(exception) for exception in PINNING_EXCEPTIONS):
            continue
        if any(pattern in external_link for pattern in UNPINNED_PATTERNS):
            failed_urls.append(f"\t{external_link}")

    all_failed = "\n".join(failed_urls)
    assert len(failed_urls) == 0, f"In {file_path}, the following links should be pinned to a version:\n{all_failed}"


@pytest.mark.parametrize("file_path", file_list)
def test_links_reachable(file_path: str, test_urls: bool = True) -> None:
    """pytest to test links from markdown, RST and Notebooks are valid."""
    file_path = Path(file_path)
    parser = get_parser(file_path)
    with open(file_path) as f:
        lines = f.readlines()
    links = parser.get_all_links(lines)
    links = set(link for link in links if not re.search("|".join(EXCEPTIONS), link))

    failed_urls: List[str] = []

    # List of internet/external links
    external_links = parser.select_external_links(links)

    # Links within same file/page
    anchors = parser.generate_page_anchors(lines)
    page_links = parser.select_page_links(links)
    failed_urls += check_page_links(anchors, page_links)

    # File to local file links
    file_links = links.difference(external_links).difference(page_links)
    failed_urls += check_file_links(file_path, file_links)

    # Test URLs (multi-threaded as its slow)
    if test_urls:
        with ThreadPool(processes=max([1, len(external_links)])) as thread_pool:
            responses = thread_pool.map(check_url_works, external_links)
        for response in responses:
            if response is not None:
                failed_urls.append(f"\t{response.url} -> {response.reason} ({response.status_code})")

    all_failed = "\n".join(failed_urls)
    assert len(failed_urls) == 0, f"In {file_path}, the following links were unreachable:\n{all_failed}"
