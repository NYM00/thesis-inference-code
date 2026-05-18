from __future__ import annotations

import platform
import shutil
import sys
from typing import Any

import torch


def _parse_cuda_index(resolved_device: str | None) -> int | None:
    if not resolved_device:
        return None
    lowered = resolved_device.lower()
    if not lowered.startswith("cuda"):
        return None
    try:
        if ":" in lowered:
            return int(lowered.split(":", maxsplit=1)[1])
        return 0
    except (ValueError, IndexError):
        return None


def _safe_gpu_name(gpu_index: int | None) -> str | None:
    if gpu_index is None or not torch.cuda.is_available():
        return None
    try:
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            return None
        return torch.cuda.get_device_name(gpu_index)
    except Exception:
        return None


def _safe_gpu_total_memory_mb(gpu_index: int | None) -> float | None:
    if gpu_index is None or not torch.cuda.is_available():
        return None
    try:
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            return None
        properties = torch.cuda.get_device_properties(gpu_index)
        return round(float(properties.total_memory) / (1024 * 1024), 2)
    except Exception:
        return None


def collect_environment_metadata(
    *,
    resolved_device: str,
    include_ultralytics: bool = False,
) -> dict[str, Any]:
    ultralytics_version = None
    if include_ultralytics:
        try:
            import ultralytics  # type: ignore

            ultralytics_version = getattr(ultralytics, "__version__", None)
        except Exception:
            ultralytics_version = None

    cuda_available = bool(torch.cuda.is_available())
    gpu_index = _parse_cuda_index(resolved_device)
    gpu_name = _safe_gpu_name(gpu_index)
    gpu_total_memory_mb = _safe_gpu_total_memory_mb(gpu_index)

    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None),
        "ultralytics_version": ultralytics_version,
        "cuda_available": cuda_available,
        "resolved_device": resolved_device,
        "gpu_index": gpu_index,
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_memory_mb,
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
    }
