import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.context import Context
from app.data_processing import ask_action
from app.db import Database
from app.models import AskRequest
from common import setup_logging


class TextEventStreamResponse(StreamingResponse):
    media_type = 'text/event-stream'


io_pool: ThreadPoolExecutor
logger: logging.Logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    global io_pool, logger

    logger = setup_logging('app')
    with ThreadPoolExecutor(max_workers=8) as thread_pool:
        io_pool = thread_pool
        yield


def get_session() -> Context:
    assert io_pool is not None
    return Context(
        db=Database(),
        io_pool=io_pool,
        logger=logger,
    )


ContextDep = Annotated[Context, Depends(get_session)]

app = FastAPI(lifespan=lifespan)

instrumentator = Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse('<h1>QA System with AI grounding</h1>')


@app.post("/ask", response_class=TextEventStreamResponse)
async def ask(query: AskRequest, context: ContextDep) -> StreamingResponse:
    return StreamingResponse(await ask_action(query.questions, context), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
