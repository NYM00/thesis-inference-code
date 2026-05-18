from __future__ import annotations

import math
import statistics
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from .io_utils import ensure_dir


PROVENANCE_MODEL_REPORTED = "model_reported"
PROVENANCE_WRAPPER_MEASURED = "wrapper_measured"
PROVENANCE_HOST_OBSERVED = "host_observed"
PROVENANCE_UNAVAILABLE = "unavailable"
PROVENANCE_DERIVED = "derived_from_existing_timings"

RUNTIME_STAGE_STATUS_MEASURED = "measured"
RUNTIME_STAGE_STATUS_PARTIAL = "partial_stage_coverage"
RUNTIME_STAGE_STATUS_EXTERNAL = "external_to_runner"

RUNTIME_STAGE_SUMMARY_FIELDS = [
    "stage_upload_download_sec",
    "stage_upload_download_status",
    "stage_frame_extraction_preprocessing_sec",
    "stage_frame_extraction_preprocessing_status",
    "stage_model_inference_sec",
    "stage_model_inference_status",
    "stage_postprocess_output_sec",
    "stage_postprocess_output_status",
]


RUN_SUMMARY_FIELDS = [
    "schema_version",
    "run_id",
    "model_name",
    "dataset_name",
    "input_path",
    "num_images",
    "num_videos",
    "num_frames",
    "batch_size",
    "image_size",
    "confidence_threshold",
    "device",
    "start_time",
    "end_time",
    "total_runtime_sec",
    "input_collection_sec",
    "model_load_sec",
    "input_loading_preparation_sec",
    "predict_wall_time_sec",
    "model_reported_total_sec",
    "predict_overhead_sec",
    "decode_time_sec",
    "preprocess_time_sec",
    "inference_time_sec",
    "postprocess_time_sec",
    "predictions_json_write_sec",
    "predictions_csv_write_sec",
    "write_time_sec",
    "quality_eval_time_sec",
    "end_to_end_runtime_sec",
    *RUNTIME_STAGE_SUMMARY_FIELDS,
    "throughput_fps",
    "throughput_clips_per_sec",
    "gpu_util_avg",
    "gpu_mem_mb_avg",
    "gpu_mem_util_pct_avg",
    "gpu_mem_total_mb",
    "cpu_util_avg",
    "ram_mb_avg",
    "ram_util_pct_avg",
    "ram_total_mb",
    "disk_read_mb_per_sec_avg",
    "disk_write_mb_per_sec_avg",
    "precision",
    "recall",
    "f1",
    "map50",
    "map50_95",
    "stride",
    "clip_length",
    "fps_target",
    "quality_metric_source",
    "gpu_stats_available",
    "notes",
]


def build_measurement_provenance_map(entries: dict[str, str]) -> dict[str, str]:
    return dict(entries)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_run_id(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    slug = model_name.lower().replace(" ", "_")
    return f"{timestamp}_{slug}_{uuid.uuid4().hex[:8]}"


def round_or_none(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
    return value


def build_runtime_stage_entry(
    *,
    seconds: float | None,
    status: str,
    components: list[str],
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "seconds": round_or_none(seconds),
        "status": status,
        "provenance": PROVENANCE_DERIVED if seconds is not None else PROVENANCE_UNAVAILABLE,
        "components": list(components),
        "note": note,
    }


def flatten_runtime_stage_breakdown(stage_breakdown: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage_upload_download_sec": stage_breakdown["upload_download"]["seconds"],
        "stage_upload_download_status": stage_breakdown["upload_download"]["status"],
        "stage_frame_extraction_preprocessing_sec": stage_breakdown["frame_extraction_preprocessing"]["seconds"],
        "stage_frame_extraction_preprocessing_status": stage_breakdown["frame_extraction_preprocessing"]["status"],
        "stage_model_inference_sec": stage_breakdown["model_inference"]["seconds"],
        "stage_model_inference_status": stage_breakdown["model_inference"]["status"],
        "stage_postprocess_output_sec": stage_breakdown["postprocess_output"]["seconds"],
        "stage_postprocess_output_status": stage_breakdown["postprocess_output"]["status"],
    }


def average_or_none(values: list[float | int | None], digits: int = 6) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(statistics.fmean(clean), digits)


def max_or_none(values: list[float | int | None], digits: int = 6) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(max(clean), digits)


def summarize_monitor_samples(samples: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        "cpu_util_avg": average_or_none([sample.get("cpu_util_pct") for sample in samples], digits=4),
        "ram_util_pct_avg": average_or_none([sample.get("ram_util_pct") for sample in samples], digits=4),
        "ram_mb_avg": average_or_none([sample.get("ram_used_mb") for sample in samples], digits=2),
        "ram_total_mb": max_or_none([sample.get("ram_total_mb") for sample in samples], digits=2),
        "gpu_util_avg": average_or_none([sample.get("gpu_util_pct") for sample in samples], digits=4),
        "gpu_mem_mb_avg": average_or_none([sample.get("gpu_mem_mb") for sample in samples], digits=2),
        "gpu_mem_util_pct_avg": average_or_none([sample.get("gpu_mem_util_pct") for sample in samples], digits=4),
        "gpu_mem_total_mb": max_or_none([sample.get("gpu_mem_total_mb") for sample in samples], digits=2),
        "disk_read_mb_per_sec_avg": average_or_none([sample.get("disk_read_mb_per_sec") for sample in samples], digits=4),
        "disk_write_mb_per_sec_avg": average_or_none([sample.get("disk_write_mb_per_sec") for sample in samples], digits=4),
    }


def prepare_run_directories(outputs_root: str | Path, run_id: str) -> dict[str, Path]:
    root = ensure_dir(Path(outputs_root) / "runs" / run_id)
    directories = {
        "run_root": root,
        "predictions": ensure_dir(root / "predictions"),
        "metrics": ensure_dir(root / "metrics"),
        "system": ensure_dir(root / "system"),
    }
    return directories


@dataclass
class StageTimings:
    values: dict[str, float] = field(default_factory=dict)

    def add(self, key: str, seconds: float) -> None:
        self.values[key] = self.values.get(key, 0.0) + max(seconds, 0.0)

    def get(self, key: str) -> float:
        return round(self.values.get(key, 0.0), 6)

    def as_dict(self) -> dict[str, float]:
        return {key: round(value, 6) for key, value in self.values.items()}


@contextmanager
def timed_stage(timings: StageTimings, key: str) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        timings.add(key, perf_counter() - started)
