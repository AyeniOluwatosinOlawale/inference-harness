from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NsightReport:
    report_path: str
    cuda_prefill_kernel_ms: float = 0.0
    cuda_decode_kernel_ms: float = 0.0
    hbm_read_bandwidth_gbps: float = 0.0
    hbm_write_bandwidth_gbps: float = 0.0
    memory_transfer_ms: float = 0.0
    raw_stats: dict = field(default_factory=dict)


class CUDAProfiler:
    """
    Nsight Systems subprocess wrapper + NVTX range annotations.

    Usage:
        profiler = CUDAProfiler(output_dir="./profiles")
        with profiler.nvtx_range("prefill"):
            # ... inference call ...
        report = await profiler.export_report("run1")
    """

    def __init__(self, output_dir: str = "./profiles") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._nsys_available = shutil.which("nsys") is not None
        self._nvtx_available = False
        try:
            import nvtx  # noqa: F401
            self._nvtx_available = True
        except ImportError:
            pass

    @contextlib.contextmanager
    def nvtx_range(self, name: str):
        """Context manager for NVTX range annotation (prefill / decode phases)."""
        if self._nvtx_available:
            import nvtx
            with nvtx.annotate(name, color="blue"):
                yield
        else:
            yield

    def nvtx_push(self, name: str) -> None:
        if self._nvtx_available:
            import nvtx
            nvtx.push_range(name)

    def nvtx_pop(self) -> None:
        if self._nvtx_available:
            import nvtx
            nvtx.pop_range()

    async def profile_subprocess(
        self,
        cmd: list[str],
        output_name: str = "profile",
    ) -> NsightReport:
        """
        Launch `cmd` under nsys profile and collect the report.
        Falls back to running cmd directly if nsys not available.
        """
        report_path = str(self.output_dir / output_name)

        if not self._nsys_available:
            await asyncio.to_thread(subprocess.run, cmd, check=False)
            return NsightReport(report_path=report_path)

        nsys_cmd = [
            "nsys", "profile",
            "--output", report_path,
            "--trace", "cuda,nvtx,osrt",
            "--force-overwrite", "true",
            "--export", "json",
        ] + cmd

        await asyncio.to_thread(subprocess.run, nsys_cmd, check=False)
        return await self._parse_report(report_path)

    async def export_report(self, output_name: str) -> NsightReport:
        """Parse an existing .nsys-rep file into NsightReport."""
        report_path = str(self.output_dir / output_name)
        return await self._parse_report(report_path)

    async def _parse_report(self, report_path: str) -> NsightReport:
        report = NsightReport(report_path=report_path)

        # nsys stats extracts summary CSV / JSON
        stats_file = report_path + "_stats.json"
        if self._nsys_available:
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["nsys", "stats", "--report", "cuda_gpu_kern_sum",
                     "--format", "json", "--output", stats_file,
                     report_path + ".nsys-rep"],
                    capture_output=True,
                    check=False,
                    timeout=60,
                )
                if os.path.exists(stats_file):
                    with open(stats_file) as f:
                        data = json.load(f)
                    report.raw_stats = data
                    report = _extract_metrics(report, data)
            except Exception:
                pass

        return report


def _extract_metrics(report: NsightReport, data: dict) -> NsightReport:
    """
    Extract prefill/decode kernel times and HBM bandwidth from nsys stats JSON.
    Kernel names containing "flash_attn" or "attention" in the prefill NVTX range
    are counted as prefill; "paged_attention" or "decode" in decode range as decode.
    """
    try:
        kernels = data.get("NvtxKernelSum", data.get("CudaGpuKernSum", []))
        prefill_ms = 0.0
        decode_ms = 0.0
        for k in kernels:
            name = (k.get("Name") or k.get("name") or "").lower()
            duration_ns = float(k.get("Total Time (ns)") or k.get("totalTimeNs") or 0)
            duration_ms = duration_ns / 1_000_000
            if any(t in name for t in ["flash_attn", "fmha_forward", "context_attn"]):
                prefill_ms += duration_ms
            elif any(t in name for t in ["paged_attn", "decode", "generation"]):
                decode_ms += duration_ms

        report.cuda_prefill_kernel_ms = prefill_ms
        report.cuda_decode_kernel_ms = decode_ms
    except Exception:
        pass
    return report
