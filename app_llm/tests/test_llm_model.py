import pytest

from app_llm.llm_model import LLMModel


@pytest.mark.parametrize('num_queries', [1, 2, 8])
def test_with_valid_prompt(num_queries):
    model = LLMModel(14)
    response = model.ask(['<input>?</input><article>Article from topic</article>' for _ in range(num_queries)])
    response = list(response)
    assert len(response) > 1  # test response was split on chunks
    for chunk in response:
        assert len(chunk) == num_queries
