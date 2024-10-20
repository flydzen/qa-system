import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Iterator, AsyncIterator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport, Response

import app.main
from app.main import app as fastapi_app

client = TestClient(fastapi_app)


@pytest.fixture(scope='session', autouse=True)
def setup_app():
    # lifespan doesn't work with TestClient
    with ThreadPoolExecutor(max_workers=1) as thread_pool:
        app.main.io_pool = thread_pool
        yield


@pytest.mark.asyncio
async def test_empty_request():
    async with AsyncClient(transport=ASGITransport(app=fastapi_app), base_url='http://test') as ac:
        response = await ac.post('/ask', json={'questions': []})

    assert response.status_code == 200, response.text
    assert response.text == 'event: qasystem\nid: 0\ndata: []\n\n'


@pytest.mark.asyncio
async def test_single_request():
    async with AsyncClient(transport=ASGITransport(app=fastapi_app), base_url='http://test') as ac:
        response = await ac.post(
            '/ask', json={'questions': [{'question': 'Who is the best hockey player?', 'topic': 'sports'}]}
        )

    assert response.status_code == 200, response.text
    events = list(parse_events(response.text))
    assert 1 < len(events) < 5
    for i, event in enumerate(events):
        assert event.event == 'qasystem'
        assert event.id == i
        assert len(event.data) == 1
        assert len(event.data[0]) > 0


@pytest.mark.asyncio
async def test_batch_request():
    q1 = {'question': 'Who is the best hockey player?', 'topic': 'sports'}
    q2 = {'question': 'What happened in Tiananmen Square in 1989? Nothing?', 'topic': 'business'}
    async with AsyncClient(transport=ASGITransport(app=fastapi_app), base_url='http://test') as ac:
        response = await ac.post(
            '/ask', json={'questions': [q1, q2]}
        )

    assert response.status_code == 200, response.text
    events = list(parse_events(response.text))
    assert 1 < len(events) < 5
    for i, event in enumerate(events):
        assert event.event == 'qasystem'
        assert event.id == i
        assert len(event.data) == 2
        assert len(event.data[0]) > 0
        assert len(event.data[1]) > 0


@dataclass
class Event:
    event: str
    id: int
    data: list


def parse_events(source: str) -> Iterator[Event]:
    events = source.strip().split('\n\n')
    for event in events:
        lines = event.splitlines()
        yield Event(
            event=lines[0].split(':', maxsplit=1)[1].strip(),
            id=int(lines[1].split(':', maxsplit=1)[1].strip()),
            data=json.loads(lines[2].split(':', maxsplit=1)[1].strip())
        )
