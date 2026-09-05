from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import GPUSnapshot


class BaseProfiler(ABC):
    @abstractmethod
    async def snapshot(self) -> list[GPUSnapshot]: ...

    @abstractmethod
    async def start_continuous(self, interval_ms: float = 500) -> None: ...

    @abstractmethod
    async def stop_continuous(self) -> list[GPUSnapshot]: ...
