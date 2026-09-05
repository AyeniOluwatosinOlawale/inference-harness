from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import RunResult, ScenarioDef


class BaseBackend(ABC):
    backend_name: str = "base"

    @abstractmethod
    async def measure_single(
        self,
        scenario: ScenarioDef,
        *,
        run_index: int = 0,
    ) -> RunResult: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    @abstractmethod
    async def list_models(self) -> list[str]: ...

    async def close(self) -> None:
        pass
