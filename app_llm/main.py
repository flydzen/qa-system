import logging
from contextlib import asynccontextmanager
from typing import Annotated

import ujson
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from sentence_transformers import SentenceTransformer

from app_llm.entities import LLMRequest, Context
from app_llm.llm_model import LLMModel
from common import setup_logging


class TextEventStreamResponse(StreamingResponse):
    media_type = 'text/event-stream'


model: SentenceTransformer
llm_model: LLMModel
logger: logging.Logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, llm_model, logger

    model = SentenceTransformer(
        model_name_or_path='all-MiniLM-L6-v2',
        device='cpu',
    )
    llm_model = LLMModel(seed=14)
    logger = setup_logging('app_llm')
    yield


def get_session() -> Context:
    return Context(
        embedding_model=model,
        llm_model=llm_model,
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


@app.post("/encode")
def encode(data: LLMRequest, context: ContextDep) -> list[list[float]]:
    context.logger.info('request /encode: %u items', len(data.items))
    return context.embedding_model.encode(data.items).tolist()


@app.post("/llm_ask", response_class=TextEventStreamResponse)
def ask(query: LLMRequest, context: ContextDep) -> StreamingResponse:
    def gen_response():
        for i, response in enumerate(context.llm_model.ask(query.items)):
            context.logger.info('/ask: sent part №%u', i)
            yield (f'event: qasystem\n'
                   f'id: {i}\n'
                   f'data: {ujson.dumps(response)}\n\n')

    context.logger.info('/ask: %u items', len(query.items))
    return StreamingResponse(gen_response(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
