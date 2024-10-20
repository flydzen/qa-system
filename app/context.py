import asyncio
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TypeVar, Callable, Any

from app.db import Database

T = TypeVar('T')


@dataclass
class Context:
    db: Database
    io_pool: Executor

    async def run_io(self, task: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.io_pool, task, *args)
