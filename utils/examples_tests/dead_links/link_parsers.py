# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

"""Collection of documentation file parsers for extracting information about
hyper-links."""

import re
import string
import warnings
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterable, List, Set


class Parser(metaclass=ABCMeta):
    """Base class interface to parsers for each file type we wish to test."""

    disallowed_char = r"\s\]\"\\>`')}:,"
    url_pattern = re.compile(rf"(https?:[^{disallowed_char}]*[^{disallowed_char}\.])")

    @staticmethod
    @abstractmethod
    def file_suffix() -> str:
        """Return file extension that the parser is used for."""

    def __init__(self, file_path: Path) -> None:
        self._filename = Path(file_path)

    @abstractmethod
    def get_all_links(self, file_contents_lines: List[str], warn_on_raw_links=False) -> Set[str]:
        """Return all hyper-links found in the file."""

    @abstractmethod
    def generate_page_anchors(self, file_contents_lines: List[str]) -> List[str]:
        """Calculate and return list of page anchors from heading text."""

    def select_page_links(self, links: Iterable[str]) -> Set[str]:
        """Return internal hyper-links which target same page/file."""
        return {link for link in links if self.page_link_pattern.match(link)}

    def select_external_links(self, links: Iterable[str]) -> Set[str]:
        """Return just the external http links found in input links."""
        return {link for link in links if self.url_pattern.search(link)}


class MdParser(Parser):
    """Class for parsing Markdown files"""

    link_pattern = re.compile(r"\[(.+|\!\[])\]\(<?([^)>\s\*]+)[)>\s]")
    automatic_link_pattern = re.compile(r"<([^)>\s\*]+)>")
    page_link_pattern = re.compile(r"^#")

    @staticmethod
    def file_suffix() -> str:
        return ".md"

    def get_all_links(self, file_contents_lines: List[str], warn_on_raw_links=False) -> Set[str]:

        file_contents = " ".join([line.strip() for line in file_contents_lines])

        links = set(groups[-1] for groups in self.link_pattern.findall(file_contents))
        raw_links = set(self.url_pattern.findall(file_contents))
        automatic_links = set(self.automatic_link_pattern.findall(file_contents))

        additional_raw = raw_links.difference(links).difference(automatic_links)
        if additional_raw and warn_on_raw_links:
            warnings.warn(f"Raw link(s) found in: {self._filename} {additional_raw}", stacklevel=2)

        return links | raw_links

    def generate_page_anchors(self, file_contents_lines: List[str]) -> List[str]:

        anchors = []

        for line in file_contents_lines:
            heading = self._extract_heading(line)
            if heading:
                anchors.append(self._calc_markdown_anchor(heading))

        return anchors

    def _extract_heading(self, line: str) -> str:
        if line.startswith("#"):
            while line.startswith("#"):
                line = line[1:]
            return line.strip()

        return ""

    def _calc_markdown_anchor(self, heading: str) -> str:
        """Convert section title string to link anchor using markdown rules."""

        # Based on:
        # https://docs.gitlab.com/ee/user/markdown.html#header-ids-and-links
        # https://github.com/gjtorikian/html-pipeline/blob/main/lib/html/pipeline/toc_filter.rb

        # Downcase the string.
        heading = heading.lower()

        # Remove all punctuation ...
        translation = str.maketrans("", "", string.punctuation)

        # ... except hyphens ...
        translation[ord("-")] = ord("-")

        # ... and convert spaces to hyphens.
        translation[ord(" ")] = ord("-")

        heading = heading.translate(translation)

        # TODO If that is not unique, add "-1", "-2", "-3",... to make it unique

        return "#" + heading


class IpynbParser(MdParser):
    """Class for parsing Notebook files"""

    @staticmethod
    def file_suffix() -> str:
        return ".ipynb"

    def _extract_heading(self, line: str) -> str:
        line = line.strip()
        if line.startswith('"#') and (line.endswith('"') or line.endswith('",')):

            if line.endswith(","):
                line = line[:-1]

            # Remove quotes
            line = line[1:]
            line = line[:-1]

            while line.startswith("#"):
                line = line[1:]

            while line.endswith("\\n"):
                line = line[:-2]

            return line.strip()

        return ""


class RstParser(Parser):
    """Class for parsing reStructuredText files"""

    link_pattern = re.compile(r"[^`]`[^`]*<([^>]+)>`_")
    page_link_pattern = re.compile(r"\s`([^<>]+)`_")

    @staticmethod
    def file_suffix() -> str:
        return ".rst"

    def get_all_links(self, file_contents_lines: List[str], warn_on_raw_links=False) -> Set[str]:

        file_contents = "".join([line.strip() for line in file_contents_lines])

        links = set(self.link_pattern.findall(file_contents))
        links_page = set(self.page_link_pattern.findall(file_contents))
        raw_links = set(self.url_pattern.findall(file_contents))

        additional_raw = raw_links.difference(links)
        if additional_raw and warn_on_raw_links:
            warnings.warn(f"Raw link(s) found in: {self._filename} {additional_raw}", stacklevel=2)

        return links | links_page | raw_links

    def generate_page_anchors(self, file_contents_lines: List[str]) -> List[str]:

        anchors = []

        title_is_next = False
        for line in reversed(file_contents_lines):
            if title_is_next:
                anchors.append(line.lower())

            title_is_next = self._is_section_char(line)

        # To help testing
        anchors.reverse()

        return anchors

    def select_page_links(self, links: Iterable[str]) -> Set[str]:
        return {link for link in links if not link.startswith("http") and not (self._filename.parent / link).exists()}

    def _is_section_char(self, line: str) -> bool:
        is_section_underline = all(c == line[0] for c in line)
        allowed_underline_char = len(line) > 0 and line[0] in "=-`:'\"~^_*+#<>."
        return is_section_underline and allowed_underline_char


def get_parser(file_path: Path) -> Parser:
    """Parser factory method"""

    PARSER_FACTORY = {
        MdParser.file_suffix(): MdParser,
        IpynbParser.file_suffix(): IpynbParser,
        RstParser.file_suffix(): RstParser,
    }

    return PARSER_FACTORY[file_path.suffix.lower()](file_path)
