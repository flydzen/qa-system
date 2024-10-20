from dataclasses import dataclass

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app_llm.llm_model import LLMModel


@dataclass
class Context:
    embedding_model: SentenceTransformer
    llm_model: LLMModel


class LLMRequest(BaseModel):
    items: list[str]
