"""Test cases for search utils."""

from server import search_utils


def test_get_prompt() -> None:
    """It gets the correct number of contexts."""
    prompt, _ = search_utils.get_prompt(
        "This is my prompt content",
        "Question?",
        ["Answer...1", "Answer...2", "Answer...3", "Answer...4", "Answer...5"],
        100,
    )
    assert 90 <= len(prompt) <= 100
    assert (
        prompt
        == "This is my prompt content\n\nContexts:\n\n1. Answer...1\n---\n2. Answer...2\n\n"
        + "Question: Question?\n\nAnswer:"
    )


def test_remove_markdown() -> None:
    """It tests removing markdown."""
    text = """## Header
* List item 1
* List item 2
Inline # character
1. Numbered list
2. Another number
This is **bold** and *italic*.
This is a [link](https://example.com)
This is `inline code`
===
This is an image ![alt text](image.jpg)
---
> This is a block quote

This is a paragraph."""
    clean = search_utils.remove_markdown(text)
    assert (
        clean
        == """
Header
List item 1
List item 2
Inline # character
Numbered list
Another number
This is bold and italic.
This is a link
This is inline code
This is an image alt text
This is a block quote
This is a paragraph."""
    )
