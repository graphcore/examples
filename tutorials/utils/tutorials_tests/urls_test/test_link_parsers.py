# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest

from .link_parsers import (
    LINK_PATTERN_MD,
    LINK_PATTERN_RAW,
    IpynbParser,
    MdParser,
    RstParser,
)


def test_md_containing_cpp_lambda() -> None:
    result = LINK_PATTERN_MD.search("blah `[](Class &class)` blah")
    assert result == None, "Regex matched a C++ lambda function"


def test_md_link_with_alt_text() -> None:
    result = LINK_PATTERN_MD.search('blah [Alt Text](figures/ExampleScreen.png "TensorBoard Example") blah')
    assert result, "Regex failed to find a match"
    assert result.group(2) == "figures/ExampleScreen.png"


def test_md_link_with_extra_angle_brackets() -> None:
    result = LINK_PATTERN_MD.search("blah [Alt Text](<https://docs.graphcore.ai/>) blah")
    assert result, "Regex failed to find a match"
    assert result.group(2) == "https://docs.graphcore.ai/"


def test_md_should_not_match_python_list_callable() -> None:
    result = LINK_PATTERN_MD.search('args = layer["func"](*args)')
    print(result)
    assert not result, "Regex should ignore (*...)"


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("blah https://docs.graphcore.ai/ blah", "https://docs.graphcore.ai/"),
        ("blah `https://docs.graphcore.ai/` blah", "https://docs.graphcore.ai/"),
        ("blah 'https://docs.graphcore.ai/' blah", "https://docs.graphcore.ai/"),
        ("blah <https://docs.graphcore.ai/> blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/. blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/, blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/: blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/} blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/] blah", "https://docs.graphcore.ai/"),
        (r"blah https://docs.graphcore.ai/\n blah", "https://docs.graphcore.ai/"),
        ("blah https://docs.graphcore.ai/\n blah", "https://docs.graphcore.ai/"),
        ('blah https://docs.graphcore.ai/" blah', "https://docs.graphcore.ai/"),
    ],
)
def test_raw_http_single_link(test_input: str, expected: str) -> None:

    result = LINK_PATTERN_RAW.findall(test_input)
    assert len(result) == 1, f"Regex failed to match exactly one item. Found {len(result)}"
    assert result[0] == expected


def test_md_parser_get_all_links_external():
    file = [
        "    - Ensure the Poplar SDK is installed (follow the instructions in the Getting",
        "Started guide for your IPU system: https://docs.graphcore.ai/en/latest/getting-started.html).",
        "- Install the requirements for the Python program with:",
    ]
    parser = MdParser("test.md")
    all_links = parser.get_all_links(file)
    assert len(all_links) == 1
    assert next(iter(all_links)) == "https://docs.graphcore.ai/en/latest/getting-started.html"


def test_md_parser_get_all_links_internal():
    file = [
        "Build the custom operator in the [PopART Leaky ReLU",
        "example](../../popart/custom_operators/leaky_relu_example) (after making sure",
    ]
    parser = MdParser("test.md")
    all_links = parser.get_all_links(file)
    assert len(all_links) == 1
    assert next(iter(all_links)) == "../../popart/custom_operators/leaky_relu_example"


def test_md_parser_get_all_links_page():
    file = [
        "and we can even [link](#head1234) to it so",
    ]
    parser = MdParser("test.md")
    all_links = parser.get_all_links(file)
    assert len(all_links) == 1
    assert next(iter(all_links)) == "#head1234"


def test_md_parser_generate_page_anchors():
    file = [
        "# Intro",
        "Blah blah",
        "## Subsection Blah",
        "Blah blah",
    ]
    parser = MdParser("test.md")
    anchors = parser.generate_page_anchors(file)
    assert len(anchors) == 2
    assert anchors[0] == "#intro"
    assert anchors[1] == "#subsection-blah"


def test_md_parser_select_page_links():
    links = [
        "./",
        "#summary",
        "https://docs.graphcore.ai/projects/model.html",
    ]
    parser = MdParser("test.md")
    page_links = parser.select_page_links(links)
    assert len(page_links) == 1
    assert "#summary" in page_links


def test_ipynb_parser_generate_page_anchors():
    file = [
        "{",
        '    "# Intro",',
        '    "Blah blah",',
        '    "## Subsection Blah",',
        '    "Blah blah",',
        "}",
    ]
    parser = IpynbParser("test.md")
    anchors = parser.generate_page_anchors(file)
    assert len(anchors) == 2
    assert anchors[0] == "#intro"
    assert anchors[1] == "#subsection-blah"


def test_rst_parser_get_all_links():
    file = [
        "the IPU's architecture by reading the `IPU Programmer's Guide",
        "<https://docs.graphcore.ai/projects/model.html>`_. You can ",
        "A brief `summary`_ and a list of additional resources are included at the end this tutorial.",
        "Graphcore also provides tutorials using Python deep learning frameworks `PyTorch <../../pytorch/>`_",
        " and `TensorFlow 2 <../../tensorflow2/>`_.",
    ]
    parser = RstParser("test.rst")
    links = parser.get_all_links(file)

    assert len(links) == 4
    assert "summary" in links
    assert "../../pytorch/" in links
    assert "../../tensorflow2/" in links
    assert "https://docs.graphcore.ai/projects/model.html" in links


def test_rst_parser_select_page_links():
    links = [
        "summary",
        "./",
        "https://docs.graphcore.ai/projects/model.html",
    ]
    parser = RstParser("test.rst")
    page_links = parser.select_page_links(links)
    assert len(page_links) == 1
    assert "summary" in page_links


def test_rst_parser_generate_page_anchors():
    file = [
        "Intro",
        "=====",
        "Blah blah",
        "Subsection Blah",
        "---------------",
        "Blah blah",
    ]
    parser = RstParser("test.rst")
    anchors = parser.generate_page_anchors(file)
    assert len(anchors) == 2
    assert anchors[0] == "intro"
    assert anchors[1] == "subsection blah"
