import enum
from typing import Iterable

from pydantic import BaseModel


class Topic(enum.Enum):
    BUSINESS = 'business'
    SPORT = 'sports'


class Question(BaseModel):
    question: str
    topic: Topic


class AskRequest(BaseModel):
    questions: list[Question]


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
