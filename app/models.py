import enum
from typing import Iterable

from pydantic import BaseModel


class Topic(enum.Enum):
    BUSINESS = 'business'
    SPORT = 'sports'


# API models


class Question(BaseModel):
    question: str
    topic: Topic


class AskRequest(BaseModel):
    questions: list[Question]


# DB models

class DBSearchRequest(BaseModel):
    topic: Topic
    embedding: Iterable


class DBResponseEntity(BaseModel):
    topic: Topic
    text: str


class DBSearchResponseItem(BaseModel):
    id: int
    distance: float
    entity: DBResponseEntity


class DBSearchResponse(BaseModel):
    items: list[DBSearchResponseItem]
