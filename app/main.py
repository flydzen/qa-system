from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated

import markdown
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse

from app.context import Context
from app.data_processing import ask_action
from app.db import Database
from app.models import AskRequest


class TextEventStreamResponse(StreamingResponse):
    media_type = 'text/event-stream'


io_pool: ThreadPoolExecutor


@asynccontextmanager
async def lifespan(_: FastAPI):
    global io_pool

    with ThreadPoolExecutor(max_workers=1) as thread_pool:
        io_pool = thread_pool
        yield


def get_session() -> Context:
    assert io_pool is not None
    return Context(
        db=Database(),
        io_pool=io_pool,
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
    return StreamingResponse(await ask_action(query.questions, context), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
