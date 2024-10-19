import asyncio
from asyncio import Future
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TypeVar, Callable, Any

from server.db import Database
from milvus_model.base import BaseEmbeddingFunction


T = TypeVar('T')


@dataclass
class Context:
    db: Database
    io_pool: Executor
    cpu_pool: Executor
    embedding_model: BaseEmbeddingFunction

    async def run_io(self, task: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self.io_pool, task, *args)

    async def run_cpu(self, task: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self.cpu_pool, task, *args)
