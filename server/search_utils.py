"""Search utility functions."""

import logging
import re
from typing import Any
from typing import Tuple

import botocore  # type: ignore


logger = logging.getLogger(__name__)


def remove_markdown(text: str) -> str:
    """Remove markdown markup from the given text."""
    # Patterns to remove (basic Markdown syntax)
    patterns = [
        (r"\!\[(.*?)\]\(.*?\)", "\\1"),  # Images
        (r"\[(.*?)\]\(.*?\)", "\\1"),  # Links
        (r"\*{1,2}(.*?)\*{1,2}", "\\1"),  # Bold and Italic
        (r"\~{2}(.*?)\~{2}", "\\1"),  # Strikethrough
        (r"\`{1,3}(.*?)\`{1,3}", "\\1"),  # Inline code and code blocks
        (r"(?:^|\n) *\> *(.*)", "\n\\1"),  # Blockquotes
        (r"(?:^|\n) *\* *(.*)", "\n\\1"),  # Lists
        (r"(?:^|\n) *\#{1,6} *", "\n\n"),  # Headers
        (r"(?:^|\n)\-{3,}\s", "\n"),  # Horizontal rules
        (r"\n{3,}", "\n\n"),  # Extra newlines
        (r"\ {2,}", " "),  # Extra spaces
    ]
    # Remove each pattern from the text
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


def get_prompt(
    prompt: str,
    query: str,
    contexts: list[str],
    prompt_limit: int,
) -> Tuple[str, int]:
    """Get prompt for query and contexts."""

    def _get_prompt_for_contexts(ctxs: list[str]) -> str:
        return (
            prompt
            + "\n\nContexts:\n\n"
            + "\n---\n".join([f"{ix+1}. {ctx}" for ix, ctx in enumerate(ctxs)])
            + f"\n\nQuestion: {query}\n\nAnswer:"
        )

    contexts = [remove_markdown(context) for context in contexts]
    n_contexts = 0
    while (
        n_contexts < len(contexts)
        and len(_get_prompt_for_contexts(contexts[0 : n_contexts + 1])) < prompt_limit
    ):
        n_contexts += 1
    return _get_prompt_for_contexts(contexts[0:n_contexts]), n_contexts


def log_metrics(
    cloudwatch: Any,
    metric_namespace: str,
    metric_name: str,
    embed_secs: float,
    index_secs: float,
    answer_secs: float,
    prompt_len: int,
    n_contexts: int,
    answer_len: int,
) -> None:
    """Log metrics to CloudWatch."""
    try:
        cloudwatch.put_metric_data(
            Namespace=metric_namespace,
            MetricData=[
                {
                    "MetricName": f"{metric_name}_embed_seconds",
                    "Value": embed_secs,
                    "Unit": "Seconds",
                },
                {
                    "MetricName": f"{metric_name}_index_seconds",
                    "Value": index_secs,
                    "Unit": "Seconds",
                },
                {
                    "MetricName": f"{metric_name}_answer_seconds",
                    "Value": answer_secs,
                    "Unit": "Seconds",
                },
                {
                    "MetricName": f"{metric_name}_prompt_length",
                    "Value": prompt_len,
                    "Unit": "Count",
                },
                {
                    "MetricName": f"{metric_name}_prompt_contexts",
                    "Value": n_contexts,
                    "Unit": "Count",
                },
                {
                    "MetricName": f"{metric_name}_answer_length",
                    "Value": answer_len,
                    "Unit": "Count",
                },
                {
                    "MetricName": f"{metric_name}_hits",
                    "Value": 1,
                    "Unit": "Count",
                },
            ],
        )
    except (
        botocore.exceptions.ClientError,
        botocore.exceptions.ParamValidationError,
    ) as error:
        logger.error("cloudwatch", extra={"error": error})
