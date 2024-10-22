import logging

import pytest
from fastapi.testclient import TestClient
from sentence_transformers import SentenceTransformer

import app_llm.main
from app_llm.llm_model import LLMModel
from app_llm.main import app

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def setup_app():
    # lifespan doesn't work with TestClient
    app_llm.main.model = SentenceTransformer(
        model_name_or_path='all-MiniLM-L6-v2',
        device='cpu',
    )
    app_llm.main.llm_model = LLMModel(seed=14)
    app_llm.main.logger = logging.getLogger('test')


class TestEncode:
    URL = '/encode'

    def test_empty_request(self):
        response = client.post(self.URL, json={'items': []})
        assert response.status_code == 200, response.text
        assert response.json() == []

    def test_invalid_request(self):
        response = client.post(self.URL, json={})
        assert response.status_code == 422, response.text
        assert response.json() == {
            "detail": [{"type": "missing", "loc": ["body", "items"], "msg": "Field required", "input": {}}]
        }

    def test_single_item(self):
        response = client.post(self.URL, json={'items': ['hello']})
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result) == 1
        assert all(len(r) == 384 for r in result)
        assert all(isinstance(i, float) for i in result[0])

    def test_batch(self):
        response = client.post(self.URL, json={'items': ['hello', 'hello', 'world']})
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result) == 3
        assert all(len(r) == 384 for r in result)
        assert result[0] == result[1]
        assert result[0] != result[2]


class TestLLM:
    URL = '/llm_ask'

    def test_empty_request(self):
        response = client.post(self.URL, json={'items': []})
        assert response.status_code == 200, response.text
        assert response.text == 'event: qasystem\nid: 0\ndata: []\n\n'

    def test_invalid_request(self):
        response = client.post(self.URL, json={})
        assert response.status_code == 422, response.text
        assert response.json() == {
            "detail": [{"type": "missing", "loc": ["body", "items"], "msg": "Field required", "input": {}}]
        }

    def test_sse(self):
        # number of events equal to number of tags in request
        request = '''<input>input</input>
        <article>a</article>
        <article>b</article>'''
        response = client.post(self.URL, json={'items': [request]})
        assert response.status_code == 200, response.text
        expected = (
            'event: qasystem\n'
            'id: 0\n'
            'data: ["Answer for question \\"input\\" is:"]\n\n'

            'event: qasystem\n'
            'id: 1\n'
            'data: ["a"]\n\n'

            'event: qasystem\n'
            'id: 2\n'
            'data: ["b"]\n\n'
        )
        assert response.text == expected

    def test_batch(self):
        response = client.post(self.URL, json={'items': ['hello', 'world', '!']})
        assert response.status_code == 200, response.text
        expected = (
            'event: qasystem\n'
            'id: 0\n'
            'data: ["I don\'t know what to say..","I don\'t know what to say..","I don\'t know what to say.."]\n\n'
        )
        assert response.text == expected
