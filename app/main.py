from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated

import milvus_model
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import markdown

from server.context import Context
from server.data_processing import ask_action
from server.db import Database
from server.models import AskRequest
from milvus_model.base import BaseEmbeddingFunction


class TextEventStreamResponse(StreamingResponse):
    media_type = 'text/event-stream'


io_pool: ThreadPoolExecutor
cpu_pool: ProcessPoolExecutor
model: BaseEmbeddingFunction


@asynccontextmanager
async def lifespan(_: FastAPI):
    global io_pool, cpu_pool, model

    model = milvus_model.dense.SentenceTransformerEmbeddingFunction(
        model_name='all-MiniLM-L6-v2',
        device='cpu',
        normalize_embeddings=True,
    )
    with ThreadPoolExecutor(max_workers=8) as thread_pool, ProcessPoolExecutor(max_workers=1) as process_pool:
        io_pool = thread_pool
        cpu_pool = process_pool
        yield


def get_session() -> Context:
    assert io_pool is not None
    assert cpu_pool is not None

    return Context(
        db=Database(),
        io_pool=io_pool,
        cpu_pool=cpu_pool,
        embedding_model=model,
    )


ContextDep = Annotated[Context, Depends(get_session)]

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_root_content():
    with open('README.md', 'r') as f:
        return markdown.markdown(f.read())


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(get_root_content())


@app.post("/ask", response_class=TextEventStreamResponse)
async def ask(query: AskRequest, context: ContextDep) -> StreamingResponse:
    return StreamingResponse(ask_action(query.questions, context), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
