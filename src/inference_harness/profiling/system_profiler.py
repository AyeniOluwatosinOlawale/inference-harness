from __future__ import annotations

import asyncio
import subprocess
import time
import xml.etree.ElementTree as ET

from ..types import GPUSnapshot
from .base import BaseProfiler


class SystemProfiler(BaseProfiler):
    """
    GPU metrics via pynvml (preferred) with nvidia-smi XML fallback.
    Supports both one-shot snapshots and continuous background polling.
    """

    def __init__(self) -> None:
        self._continuous_task: asyncio.Task | None = None
        self._continuous_results: list[GPUSnapshot] = []
        self._stop_event = asyncio.Event()

    async def snapshot(self) -> list[GPUSnapshot]:
        snaps = await _snapshot_pynvml()
        if not snaps:
            snaps = await _snapshot_smi()
        return snaps

    async def start_continuous(self, interval_ms: float = 500) -> None:
        self._continuous_results = []
        self._stop_event.clear()
        self._continuous_task = asyncio.create_task(
            self._poll_loop(interval_ms / 1000)
        )

    async def stop_continuous(self) -> list[GPUSnapshot]:
        self._stop_event.set()
        if self._continuous_task:
            await self._continuous_task
        return list(self._continuous_results)

    async def _poll_loop(self, interval_s: float) -> None:
        while not self._stop_event.is_set():
            snaps = await self.snapshot()
            self._continuous_results.extend(snaps)
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=interval_s,
                )
            except asyncio.TimeoutError:
                pass


async def _snapshot_pynvml() -> list[GPUSnapshot]:
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        snaps: list[GPUSnapshot] = []
        ts = time.perf_counter() * 1000
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except pynvml.NVMLError:
                power = 0.0
            try:
                sm_clock = float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            except pynvml.NVMLError:
                sm_clock = 0.0
            try:
                mem_clock = float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
            except pynvml.NVMLError:
                mem_clock = 0.0
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except pynvml.NVMLError:
                temp = 0.0
            snaps.append(GPUSnapshot(
                gpu_index=i,
                memory_used_mb=mem.used / 1024 / 1024,
                memory_total_mb=mem.total / 1024 / 1024,
                utilization_pct=float(util.gpu),
                power_watts=power,
                sm_clock_mhz=sm_clock,
                memory_clock_mhz=mem_clock,
                temperature_c=temp,
                timestamp_ms=ts,
            ))
        return snaps
    except Exception:
        return []


async def _snapshot_smi() -> list[GPUSnapshot]:
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["nvidia-smi", "-q", "-x"],
            capture_output=True, timeout=5, check=True,
        )
        root = ET.fromstring(result.stdout)
        snaps: list[GPUSnapshot] = []
        ts = time.perf_counter() * 1000
        for i, gpu in enumerate(root.findall("gpu")):
            def _t(path: str, default: str = "0") -> str:
                el = gpu.find(path)
                return el.text.strip() if el is not None and el.text else default

            def _f(raw: str) -> float:
                try:
                    return float(raw.split()[0])
                except Exception:
                    return 0.0

            snaps.append(GPUSnapshot(
                gpu_index=i,
                memory_used_mb=_f(_t("fb_memory_usage/used")),
                memory_total_mb=_f(_t("fb_memory_usage/total")),
                utilization_pct=_f(_t("utilization/gpu_util")),
                power_watts=_f(_t("power_readings/power_draw", "0 W")),
                sm_clock_mhz=0.0,
                memory_clock_mhz=0.0,
                temperature_c=_f(_t("temperature/gpu_temp", "0 C")),
                timestamp_ms=ts,
            ))
        return snaps
    except Exception:
        return []
