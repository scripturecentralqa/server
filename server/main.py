"""Server."""
import json
import logging.config
import os
import random
import string
import time
import traceback
from enum import IntEnum
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Optional

import boto3  # type: ignore
import numpy as np
import openai
import pinecone  # type: ignore
import requests

# import voyageai  # type: ignore
from dotenv import load_dotenv
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi.responses import StreamingResponse
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

from server.search_utils import fix_citations
from server.search_utils import get_norag_prompt
from server.search_utils import get_prompt
from server.search_utils import get_url
from server.search_utils import log_metrics


# init environment
load_dotenv()
pinecone_key = os.environ["PINECONE_API_KEY"]
pinecone_env = os.environ["PINECONE_ENV"]
openai_key = os.environ["OPENAI_API_KEY"]
# voyageai_key = os.environ["VOYAGE_API_KEY"]
upstash_redis_url = os.environ["UPSTASH_REDIS_REST_URL"]
upstash_redis_token = os.environ["UPSTASH_REDIS_REST_TOKEN"]
# redis = Redis.from_env()
# init logging
logging.config.fileConfig("server/logging.ini")
logger = logging.getLogger(__name__)

# init fastapi
debug = os.environ.get("DEBUG", "false").lower() == "true"
app = FastAPI(debug=debug)

# init openai
openai.api_key = openai_key
embedding_model = "text-embedding-ada-002"
prompt_limit = 10000
similarity_threshold = 0.86

# init voyageai
# voyageai.api_key = voyageai_key

# init pinecone
index_name = "scqa"
# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=pinecone_key,
    environment=pinecone_env,
)
index = pinecone.Index(index_name)

# init metrics
metric_namespace = os.environ.get("METRIC_NAMESPACE", "")
cloudwatch = boto3.client("cloudwatch") if metric_namespace else None

# other constants
search_limit = 20
max_answer_tokens = 512
rag_role_content = "You are an apostle of Jesus Christ."
# role_content = "You are an apostle of the Church of Jesus Christ of Latter-day Saints"
rag_prompt_content = (
    "Answer the question as truthfully as possible using the numbered contexts below and if the"
    + ' answer is not contained within the text below, say "Sorry, I don\'t know". Please give a'
    + " detailed answer. For each sentence in your answer, provide a link to the contexts the"
    + " sentence came from using the format [^context number]."
    # "Answer the question as truthfully as possible using the provided context, "
    # + 'and if answer is not contained within the text below, say "Sorry, I don\'t know".'
)
norag_role_content = (
    "You are a helpful assistant who understands the doctrine of the"
    + "Church of Jesus Christ of Latter-day Saints."
)
norag_prompt_content = (
    "Please answer the following question using the teachings of the"
    + "Church of Jesus Christ of Latter-day Saints."
)


# data models
class SearchResult(BaseModel):
    """Search result."""

    id: str
    index: int
    title: str
    url: Optional[str] = ""
    score: float
    text: str


class SearchResponse(BaseModel):
    """Search response."""

    q: str
    session: int
    answer: str
    results: list[SearchResult]


class SearchResultsResponse(BaseModel):
    """Search results response."""

    key: str
    results: list[SearchResult]


@app.middleware("http")
async def log_exceptions_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log exceptions."""
    try:
        return await call_next(request)
    except Exception:
        body = await request.body()
        logger.error(
            traceback.format_exc(),
            extra={
                "url": request.url,
                "method": request.method,
                # "headers": request.headers,
                "body": body,
            },
        )
        return Response(status_code=500, content="Internal Server Error")


def get_search_results(q: str) -> tuple[list[dict[str, Any]], float, float, float]:
    """Get the search results."""
    # get new query using HyDE approach
    start = time.perf_counter()
    query_prompt = f"Please write a paragraph to answer the question \nQuestion: {q}"
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": rag_role_content,
            },
            {"role": "user", "content": query_prompt},
        ],
        temperature=0.0,
        max_tokens=max_answer_tokens,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )  # type: ignore
    hyde_query = gpt_response["choices"][0]["message"]["content"].strip()
    hyde_secs = time.perf_counter() - start

    # get embedding for HyDE query
    start = time.perf_counter()
    embed_response = openai.Embedding.create(
        input=[hyde_query], engine=embedding_model
    )  # type: ignore
    embedding = embed_response["data"][0]["embedding"]
    # embedding = get_voyageai_embeddings(
    #     [hyde_query], model="voyage-01", input_type="query"
    # )[0]
    embed_secs = time.perf_counter() - start

    # query index
    start = time.perf_counter()
    query_response = index.query(embedding, top_k=search_limit, include_metadata=True)
    index_secs = time.perf_counter() - start
    search_results = query_response["matches"]

    # only keep results with similarity above threshold
    search_results = [
        res for res in search_results if res["score"] >= similarity_threshold
    ]

    # re-order search results for maximal marginal relevance
    all_texts = [res["metadata"]["text"] for res in search_results] + [
        hyde_query
    ]  # all result texts plus the query
    embed_response = openai.Embedding.create(input=all_texts, engine="text-embedding-ada-002")  # type: ignore
    all_embeddings = [data["embedding"] for data in embed_response["data"]]
    doc_embeddings = np.array(
        all_embeddings[:-1]
    )  # search result embeddings as a numpy array
    query_embedding = np.array(all_embeddings[-1])  # query embedding as a numpy array
    mmr_result = maximal_marginal_relevance(
        query_embedding, doc_embeddings.tolist(), 0.7, len(search_results)
    )
    new_results = []
    for ix in mmr_result:
        new_results.append(search_results[ix])

    # return re-ordered search results and various times
    return new_results, hyde_secs, embed_secs, index_secs


@app.get("/search")
async def search(
    q: str = Query(max_length=100),
    query_type: str = Query(default="rag", max_length=10),
    background_tasks: BackgroundTasks = BackgroundTasks(),  # noqa: B008
) -> SearchResponse:
    """Search."""
    embed_secs = 0.0
    index_secs = 0.0
    while True:
        if query_type == "rag" or query_type == "ragonly":
            # get search results
            search_results, hyde_secs, embed_secs, index_secs = get_search_results(q)
            print("search_results", search_results)

            # get prompt
            texts = [res["metadata"]["text"] for res in search_results]
            prompt_content = rag_prompt_content
            prompt, n_contexts = get_prompt(rag_prompt_content, q, texts, prompt_limit)
            role_content = rag_role_content
        else:  # norag
            prompt_content = norag_prompt_content
            prompt = get_norag_prompt(norag_prompt_content, q)
            role_content = norag_role_content
            search_results = []
            n_contexts = 0

        print("PROMPT", prompt)
        start = time.perf_counter()
        try:
            answer_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": role_content,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_answer_tokens,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )  # type: ignore

            answer = answer_response["choices"][0]["message"]["content"].strip()
        except Exception:
            answer = (
                "Sorry, we can't provide an answer right now. Please try again later."
            )
        answer_secs = time.perf_counter() - start
        if query_type == "rag" and answer == "Sorry, I don't know.":
            query_type = "norag"
        else:
            answer = fix_citations(answer)
            break

    # create response
    response = SearchResponse(
        q=q,
        session=random.getrandbits(32),
        answer=answer,
        results=[
            SearchResult(
                id=res["id"],
                index=ix + 1,
                score=res["score"],
                title=res["metadata"]["title"],
                text=res["metadata"]["text"],
                url=get_url(res["metadata"]),
            )
            for ix, res in enumerate(search_results)
        ],
    )

    logger.info(
        "search",
        extra={
            "role": role_content,
            "prefix": prompt_content,
            "response": response.dict(),
        },
    )
    if cloudwatch:
        background_tasks.add_task(
            log_metrics,
            cloudwatch,
            metric_namespace,
            "search",
            query_type,
            hyde_secs,
            embed_secs,
            index_secs,
            answer_secs,
            len(prompt),
            n_contexts,
            len(answer),
        )
    return response


# generate a random string
def generate_random_string(length: int = 10) -> str:
    """Generate a random (non-secure) key."""
    return "".join(
        random.choices(string.ascii_letters + string.digits, k=length)  # noqa: S311
    )


@app.get("/search_results")
async def search_results(
    q: str = Query(max_length=100),
    background_tasks: BackgroundTasks = BackgroundTasks(),  # noqa: B008
) -> SearchResultsResponse:
    """New endpoint for fetching search results."""
    # Get search results
    results, hyde_secs, embed_secs, index_secs = get_search_results(q)
    texts = [res["metadata"]["text"] for res in results]
    # prompt_content = rag_prompt_content
    prompt, n_contexts = get_prompt(rag_prompt_content, q, texts, prompt_limit)
    # Generate a random key
    key = generate_random_string()
    # Save results to Upstash Redis
    upstash_redis_set_url = f"{upstash_redis_url}/SET/{key}"
    headers = {"Authorization": f"Bearer {upstash_redis_token}"}
    requests.post(upstash_redis_set_url, data=prompt, headers=headers, timeout=5)
    # Return the key to the client
    response = SearchResultsResponse(
        key=key,
        results=[
            SearchResult(
                id=res["id"],
                index=ix + 1,
                score=res["score"],
                title=res["metadata"]["title"],
                text=res["metadata"]["text"],
                url=get_url(res["metadata"]),
            )
            for ix, res in enumerate(results)
        ],
    )
    if cloudwatch:
        background_tasks.add_task(
            log_metrics,
            cloudwatch,
            metric_namespace,
            "search_results",
            "",
            hyde_secs,
            embed_secs,
            index_secs,
            0,
            len(prompt),
            n_contexts,
            0,
        )
    return response


# New endpoint to stream search results based on a key
@app.get("/stream_answer")
async def stream_answer(key: str = Query(max_length=100)) -> StreamingResponse:
    """New endpoint for streaming answer."""
    # Get results from Upstash Redis
    upstash_redis_get_url = f"{upstash_redis_url}/GET/{key}"
    headers = {"Authorization": f"Bearer {upstash_redis_token}"}
    response = requests.get(upstash_redis_get_url, headers=headers, timeout=5)

    # Check if the key exists in Upstash Redis
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Key not found")
    # get prompt
    prompt = json.loads(response.content)["result"]
    role_content = rag_role_content

    # Stream the GPT-3.5 response to the client
    def stream_openai_response() -> Any:
        """Call openai."""
        try:
            answer_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": role_content,
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=0.0,
                max_tokens=max_answer_tokens,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )  # type: ignore

        except Exception as e:
            raise HTTPException(
                503,
                detail=f"Sorry, we can't provide an answer right now. Please try again later.{e}",
            ) from e

        try:
            for chunk in answer_response:
                current_content = chunk["choices"][0]["delta"].get("content", "")
                yield current_content
        except Exception as e:
            print("OpenAI Response (Streaming) Error: " + str(e))
            raise HTTPException(
                503,
                detail=f"Sorry, we can't provide an answer right now. Please try again later.{e}",
            ) from e

    return StreamingResponse(stream_openai_response(), media_type="text/event-stream")


class RatingScore(IntEnum):
    """Valid rating score."""

    UP = 1
    UP2 = 2
    DOWN = -1
    DOWN2 = -2


class Rating(BaseModel):
    """Rating."""

    session: int
    user: int
    result: int
    score: RatingScore


@app.post("/rate")
async def rate(rating: Rating) -> Response:
    """Rate."""
    logger.info(
        "rate",
        extra={
            "session": rating.session,
            "user": rating.user,
            "result": rating.result,
            "score": rating.score,
        },
    )
    return Response(status_code=201)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    # logger.info("health")
    return Response(status_code=200, content="OK")
