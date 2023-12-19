"""Search utility functions."""

import logging
import re
from typing import Any
from typing import List
from typing import Tuple

import botocore  # type: ignore
import numpy as np
from langchain_community.utils.math import cosine_similarity


logger = logging.getLogger(__name__)


def remove_markdown(text: str) -> str:
    """Remove markdown markup from the given text."""
    # copied to models repo: models/split_utils.py
    # Patterns to remove (basic Markdown syntax)
    patterns = [
        (r"\!\[(.*?)\]\(.*?\)", "\\1"),  # Images
        (r"\[(.*?)\]\(.*?\)", "\\1"),  # Links
        (r"\*{1,2}(.*?)\*{1,2}", "\\1"),  # Bold and Italic
        (r"\~{2}(.*?)\~{2}", "\\1"),  # Strikethrough
        (r"\`{1,3}(.*?)\`{1,3}", "\\1"),  # Inline code and code blocks
        (r"(?:^|\n) *\> *(.*)", "\n\\1"),  # Blockquotes
        (r"(?:^|\n) *\* *(.*)", "\n\\1"),  # Lists
        (r"(?:^|\n) *\d+\.? *(.*)", "\n\\1"),  # Lists
        (r"(?:^|\n) *\#{1,6} *", "\n\n"),  # Headers
        (r"(?:^|\n)\-{3,}\s", "\n"),  # Horizontal rules
        (r"(?:^|\n)\={3,}\s", "\n"),  # Horizontal rules
        (r"(\n *){2,}", "\n"),  # Extra newlines
        (r"\ {2,}", " "),  # Extra spaces
    ]
    # Remove each pattern from the text
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


def get_url(metadata: dict[str, Any]) -> str:
    """Get URL from metadata."""
    url = ""
    if "url" in metadata:
        url += metadata["url"]
    if "anchor" in metadata:
        url += "#" + metadata["anchor"]
    return url


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


def get_norag_prompt(prompt: str, query: str) -> str:
    """Get prompt for query."""
    return prompt + f"\n\nQuestion: {query}\n\nAnswer:"


def fix_citations(answer: str) -> str:
    """Fix citation references in the answer."""
    # replace [\d{1,2}] with [^\d{1,2}]
    answer = re.sub(r"\[(\d{1,2})\]", r"[^\1]", answer)
    # replace [^context \d{1,2}] with [^\d{1,2}]
    answer = re.sub(r"\[\^context (\d{1,2})\]", r"[^\1]", answer)
    return answer


def log_metrics(
    cloudwatch: Any,
    metric_namespace: str,
    metric_name: str,
    query_type: str,
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
                    "MetricName": f"{metric_name}_{query_type}_embed_seconds",
                    "Value": embed_secs,
                    "Unit": "Seconds",
                },
                {
                    "MetricName": f"{metric_name}_{query_type}_index_seconds",
                    "Value": index_secs,
                    "Unit": "Seconds",
                },
                {
                    "MetricName": f"{metric_name}_{query_type}_answer_seconds",
                    "Value": answer_secs,
                    "Unit": "Seconds",
                },
                {
                    "MetricName": f"{metric_name}_{query_type}_prompt_length",
                    "Value": prompt_len,
                    "Unit": "Count",
                },
                {
                    "MetricName": f"{metric_name}_{query_type}_prompt_contexts",
                    "Value": n_contexts,
                    "Unit": "Count",
                },
                {
                    "MetricName": f"{metric_name}_{query_type}_answer_length",
                    "Value": answer_len,
                    "Unit": "Count",
                },
                {
                    "MetricName": f"{metric_name}_{query_type}_hits",
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


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs
