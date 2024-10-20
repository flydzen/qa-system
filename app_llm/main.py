from contextlib import asynccontextmanager
from typing import Annotated

import ujson
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer

from app_llm.entities import LLMRequest, Context
from app_llm.llm_model import LLMModel


class TextEventStreamResponse(StreamingResponse):
    media_type = 'text/event-stream'


model: SentenceTransformer
llm_model: LLMModel


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, llm_model

    model = SentenceTransformer(
        model_name_or_path='all-MiniLM-L6-v2',
        device='cpu',
    )
    llm_model = LLMModel(seed=14)
    yield


def get_session() -> Context:
    return Context(
        embedding_model=model,
        llm_model=llm_model,
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


@app.post("/encode")
def root(data: LLMRequest, context: ContextDep) -> list[list[float]]:
    return context.embedding_model.encode(data.items).tolist()


@app.post("/llm_ask", response_class=TextEventStreamResponse)
def ask(query: LLMRequest, context: ContextDep) -> StreamingResponse:
    def gen_response():
        for i, response in enumerate(context.llm_model.ask(query.items)):
            yield (f'event: qasystem\n'
                   f'id: {i}\n'
                   f'data: {ujson.dumps(response)}\n\n')

    return StreamingResponse(gen_response(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
