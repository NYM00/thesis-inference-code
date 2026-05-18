from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .io_utils import write_csv_rows
from .runtime import now_iso

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class SystemMonitor:
    """Background sampler for lightweight host-level observations.

    The current implementation samples machine-level CPU, RAM, disk I/O rate,
    and optional GPU utilization/memory via ``nvidia-smi``. It does not attempt
    per-process attribution.
    """

    interval_sec: float = 0.5
    gpu_index: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_perf: float | None = None
        self._prev_disk_bytes: tuple[int, int] | None = None
        self._prev_disk_time: float | None = None
        self._nvidia_smi = shutil.which("nvidia-smi")

    def start(self) -> None:
        self.samples = []
        self._stop_event.clear()
        self._start_perf = time.perf_counter()
        self._prev_disk_bytes = None
        self._prev_disk_time = None
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, name="system-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_sec * 4))
        return list(self.samples)

    def save_csv(self, path: str) -> None:
        fieldnames = [
            "timestamp",
            "elapsed_sec",
            "cpu_util_pct",
            "ram_util_pct",
            "ram_used_mb",
            "ram_total_mb",
            "disk_read_mb_per_sec",
            "disk_write_mb_per_sec",
            "gpu_util_pct",
            "gpu_mem_mb",
            "gpu_mem_total_mb",
            "gpu_mem_util_pct",
        ]
        write_csv_rows(path, fieldnames, self.samples)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.perf_counter()
            sample = self._sample_once()
            self.samples.append(sample)
            elapsed = time.perf_counter() - started
            remaining = max(0.0, self.interval_sec - elapsed)
            self._stop_event.wait(remaining)

    def _sample_once(self) -> dict[str, Any]:
        elapsed_sec = None
        if self._start_perf is not None:
            elapsed_sec = round(time.perf_counter() - self._start_perf, 6)

        # These are host-level observations, not process-level usage metrics.
        cpu_util_pct = None
        ram_util_pct = None
        ram_used_mb = None
        ram_total_mb = None
        disk_read_mb_per_sec = None
        disk_write_mb_per_sec = None

        if psutil is not None:
            cpu_util_pct = round(psutil.cpu_percent(interval=None), 4)
            virtual_memory = psutil.virtual_memory()
            ram_util_pct = round(float(virtual_memory.percent), 4)
            ram_used_mb = round(virtual_memory.used / (1024 * 1024), 2)
            ram_total_mb = round(virtual_memory.total / (1024 * 1024), 2)
            disk_counters = psutil.disk_io_counters()
            if disk_counters is not None:
                now_perf = time.perf_counter()
                current_bytes = (int(disk_counters.read_bytes), int(disk_counters.write_bytes))
                if self._prev_disk_bytes is not None and self._prev_disk_time is not None:
                    delta_t = max(now_perf - self._prev_disk_time, 1e-6)
                    disk_read_mb_per_sec = round((current_bytes[0] - self._prev_disk_bytes[0]) / (1024 * 1024) / delta_t, 4)
                    disk_write_mb_per_sec = round((current_bytes[1] - self._prev_disk_bytes[1]) / (1024 * 1024) / delta_t, 4)
                self._prev_disk_bytes = current_bytes
                self._prev_disk_time = now_perf

        gpu_util_pct, gpu_mem_mb, gpu_mem_total_mb, gpu_mem_util_pct = self._sample_gpu()
        return {
            "timestamp": now_iso(),
            "elapsed_sec": elapsed_sec,
            "cpu_util_pct": cpu_util_pct,
            "ram_util_pct": ram_util_pct,
            "ram_used_mb": ram_used_mb,
            "ram_total_mb": ram_total_mb,
            "disk_read_mb_per_sec": disk_read_mb_per_sec,
            "disk_write_mb_per_sec": disk_write_mb_per_sec,
            "gpu_util_pct": gpu_util_pct,
            "gpu_mem_mb": gpu_mem_mb,
            "gpu_mem_total_mb": gpu_mem_total_mb,
            "gpu_mem_util_pct": gpu_mem_util_pct,
        }

    def _sample_gpu(self) -> tuple[float | None, float | None, float | None, float | None]:
        # GPU values come from host-observed ``nvidia-smi`` output when available.
        if not self._nvidia_smi:
            return None, None, None, None

        command = [
            self._nvidia_smi,
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            f"--id={self.gpu_index}",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None, None, None, None

        line = completed.stdout.strip().splitlines()
        if not line:
            return None, None, None, None
        try:
            util_text, mem_text, mem_total_text = [item.strip() for item in line[0].split(",", maxsplit=2)]
            gpu_util_pct = round(float(util_text), 4)
            gpu_mem_mb = round(float(mem_text), 2)
            gpu_mem_total_mb = round(float(mem_total_text), 2)
            gpu_mem_util_pct = None
            if gpu_mem_total_mb > 0:
                gpu_mem_util_pct = round((gpu_mem_mb / gpu_mem_total_mb) * 100.0, 4)
            return gpu_util_pct, gpu_mem_mb, gpu_mem_total_mb, gpu_mem_util_pct
        except (ValueError, IndexError):
            return None, None, None, None
