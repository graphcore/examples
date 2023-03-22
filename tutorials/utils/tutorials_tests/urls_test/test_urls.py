# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

"""Test script to check hyperlinks work; external (http), internal (relative
files) and page (references to section headings)."""

from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from typing import Container, Iterable, List, Optional, Tuple

import requests

from ..testing_util import get_file_list
from .link_parsers import get_parser

# URLs which don't exist yet (e.g documentation for a future release) can be
# added to the list of exceptions below.
#
# Make sure to add a TODO(TXXXX) comment to remove the exception once the link
# is fixed.
EXCEPTIONS: List[str] = []

# URLs which are un-pegged but the repo doesn't currently have versions
EXCEPTIONS_PEGGING = [
    "https://docs.graphcore.ai/en/latest/getting-started.html",
    "https://docs.graphcore.ai/en/latest/hardware.html",
    "https://docs.graphcore.ai/en/latest/software.html",
    "https://docs.graphcore.ai/projects/graphcore-glossary/en/latest/index.html",
    "https://docs.pytest.org/en/latest/cache.html",
]


def check_url_works(url: str, file_path: Path) -> Optional[Tuple[str, Path, str, int]]:
    """Check given `url` is responsive. If an error occurs return the error
    response string and code. Return `None` if good.
    """
    try:
        response = requests.head(url)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Allow the test to succeed with intermittent issues.
        # (TooManyRedirects is not caught as could be a broken url.)
        print(f"\t{url} -> ConnectionError/Timeout")
        return None

    code = response.status_code
    message = requests.status_codes._codes[code][0]  # pylint: disable=protected-access

    print(f"\t{url} -> {message} ({code})")

    if response.status_code == 302:
        check_url_works(response.headers["Location"], file_path)
    else:
        # Allow any non 4xx status code, as other failures could be temporary
        # and break the CI tests.
        if response.status_code >= 400 and response.status_code < 500:
            return url, file_path, message, code

    return None


def check_page_links(file_path: Path, anchors: Container[str], links: Iterable[str]) -> List[str]:
    """Check links with same file. Return list of failed links."""

    # Links appear to be case insensitive, so `lower`
    return [f"{file_path}: {link} NON-EXISTENT" for link in links if link.lower() not in anchors]


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
            failed_paths.append(f"{file_path}: {link_target} NON-EXISTENT")

    return failed_paths


def test_links_are_pegged() -> None:
    """pytest to test links are pegged at a version"""

    UNVERSIONED = ["tree/master", "en/latest"]
    root_path = Path(__file__).parents[3]
    text_types = (".md", ".rst", ".ipynb")
    file_list = get_file_list(root_path, text_types)

    failed_urls: List[str] = []

    for file_path in file_list:
        lines = file_path.read_text(encoding="utf-8").splitlines()

        parser = get_parser(file_path)
        links = parser.get_all_links(lines)
        external_links = parser.select_external_links(links)

        for link in external_links:
            if any(item in link for item in UNVERSIONED) and not any(
                link.startswith(item) for item in EXCEPTIONS_PEGGING
            ):
                failed_urls.append(f"{file_path}: {link}")

    no_failures = not failed_urls
    assert no_failures, "The following links should be pegged to a version:\n" + "\n".join(failed_urls)


def test_all_links(test_urls: bool = True, force_full_build: bool = False) -> None:
    """pytest to test links from markdown, RST and Notebooks are valid."""
    root_path = Path(__file__).parents[3]
    text_types = (".md", ".rst", ".ipynb")
    file_list = get_file_list(root_path, text_types, force_full_build)

    # Get list of URLs to test (serially as its fast)
    jobs: List[Tuple[str, Path]] = []
    failed_urls: List[str] = []
    for file_path in file_list:
        print("Processing ", file_path)
        lines = file_path.read_text(encoding="utf-8").splitlines()

        parser = get_parser(file_path)
        links = parser.get_all_links(lines)
        anchors = parser.generate_page_anchors(lines)

        # List of internet/external links
        external_links = parser.select_external_links(links)
        jobs.extend((url, file_path) for url in external_links)

        # Links within same file/page
        page_links = parser.select_page_links(links)
        failed_urls += check_page_links(file_path, anchors, page_links)

        # File to local file links
        file_links = links.difference(external_links).difference(page_links)
        failed_urls += check_file_links(file_path, file_links)

    # Test URLs (multi-threaded as its slow)
    if test_urls:
        with ThreadPool(processes=max([1, len(jobs)])) as thread_pool:
            url_results = thread_pool.starmap(check_url_works, jobs)
        for url_result in url_results:
            if url_result is not None:
                url, filename, message, code = url_result
                if url in EXCEPTIONS:
                    print(f"{url} found in exceptions: ignoring {message} ({code})")
                else:
                    failed_urls.append(f"{filename}: {url} {message} ({code})")

    no_failures = not failed_urls
    assert no_failures, "\n".join(failed_urls)


def test_all_internal_links() -> None:
    """Always needed as internal references can come from anywhere in the repo
    not just the edited paths.
    """
    test_all_links(test_urls=False, force_full_build=True)
