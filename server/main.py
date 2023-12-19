"""Server."""
import logging.config
import os
import random
import time
import traceback
from enum import IntEnum
from typing import Awaitable
from typing import Callable
from typing import Optional

import boto3  # type: ignore
import numpy as np
import openai
import pinecone  # type: ignore
import voyageai  # type: ignore
from dotenv import load_dotenv
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import Query
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response
from voyageai import get_embeddings as get_voyageai_embeddings

from server.search_utils import fix_citations
from server.search_utils import get_norag_prompt
from server.search_utils import get_prompt
from server.search_utils import get_url
from server.search_utils import log_metrics
from server.search_utils import maximal_marginal_relevance


# init environment
load_dotenv()
pinecone_key = os.environ["PINECONE_KEY"]
pinecone_env = os.environ["PINECONE_ENV"]
openai_key = os.environ["OPENAI_KEY"]
voyageai_key = os.environ["VOYAGE_API_KEY"]

# init logging
logging.config.fileConfig("server/logging.ini")
logger = logging.getLogger(__name__)

# init fastapi
debug = os.environ.get("DEBUG", "false").lower() == "true"
app = FastAPI(debug=debug)

# init openai
openai.api_key = openai_key
# embedding_model = "text-embedding-ada-002"
prompt_limit = 10000

# init voyageai
voyageai.api_key = voyageai_key

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


@app.get("/search")
async def search(
    q: str = Query(max_length=100),
    query_type: str = Query(default="rag", max_length=10),
    background_tasks: BackgroundTasks = BackgroundTasks(),  # noqa: B008
) -> SearchResponse:
    """Search."""
    embed_secs = 0.0
    index_secs = 0.0
    # get answer

    while True:
        if query_type == "rag" or query_type == "ragonly":

            # get query embedding
            start = time.perf_counter()
            # embed_response = openai.Embedding.create(
            #     input=[q], engine=embedding_model
            # )  # type: ignore
            # embedding = embed_response["data"][0]["embedding"]
            embedding = get_voyageai_embeddings(
                [q], model="voyage-01", input_type="query"
            )[0]
            embed_secs = time.perf_counter() - start

            # query index
            start = time.perf_counter()
            query_response = index.query(
                embedding, top_k=search_limit, include_metadata=True
            )
            index_secs = time.perf_counter() - start
            search_results = query_response["matches"]
            print("search_results", search_results)
            all_texts = [res["metadata"]["text"] for res in search_results] + [
                q
            ]  # all search result texts plus the query
            embed_response = openai.Embedding.create(input=all_texts, engine="text-embedding-ada-002")  # type: ignore
            all_embeddings = [data["embedding"] for data in embed_response["data"]]
            doc_embeddings = np.array(
                all_embeddings[:-1]
            )  # search result embeddings as a numpy array
            doc_embeddings = doc_embeddings
            query_embedding = np.array(
                all_embeddings[-1]
            )  # query embedding as a numpy array

            mmr_result = maximal_marginal_relevance(
                query_embedding, doc_embeddings.tolist(), 0.7, len(search_results)
            )

            new_results = []
            for ix in mmr_result:
                new_results.append(search_results[ix])
            search_results = new_results

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
            embed_secs,
            index_secs,
            answer_secs,
            len(prompt),
            n_contexts,
            len(answer),
        )
    return response


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
