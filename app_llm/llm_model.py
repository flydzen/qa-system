import random
import re
from typing import Iterator


class LLMModel:
    def __init__(self, seed):
        random.seed(seed)

    def _cpu_load(self, n) -> None:
        t = 1
        for i in range(n):
            t += t / (t + 1)

    def ask(self, prompts: list[str]) -> Iterator[list[str]]:
        # yes, we are parsing the message we just made, but this is done intentionally
        response = []
        for prompt in prompts:
            match = re.compile('<input>(.+)</input>').search(prompt)
            if match is None:
                self._cpu_load(10000000)
                response.append("I don't know what to say..")
            else:
                response.append(f'Answer for question "{match.group(1)}" is:')
        yield response
        iters = [re.compile('<article>(.+)</article>').finditer(prompt) for prompt in prompts]
        while True:
            response = []
            for iterator in iters:
                if (match := next(iterator, None)) is None:
                    response.append([])
                    continue
                article = match.group(1)
                start = random.randint(0, len(article))
                stop = random.randint(start, start + 128)
                response.append(article[start: stop])
                # self._cpu_load(stop * 10000)
                self._cpu_load(100_000)
            if not any(response):
                break
            yield response
