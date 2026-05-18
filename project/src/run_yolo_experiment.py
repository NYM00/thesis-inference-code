from __future__ import annotations

import argparse
import gc
import hashlib
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from ultralytics.data.dataset import DATASET_CACHE_VERSION
from ultralytics.data.utils import get_hash, save_dataset_cache_file, verify_image_label
from ultralytics import YOLO

from common.config import apply_config_defaults, namespace_to_dict
from common.environment import collect_environment_metadata
from common.io_utils import write_csv_rows, write_json
from common.runtime import (
    PROVENANCE_DERIVED,
    PROVENANCE_HOST_OBSERVED,
    PROVENANCE_MODEL_REPORTED,
    PROVENANCE_UNAVAILABLE,
    PROVENANCE_WRAPPER_MEASURED,
    RUNTIME_STAGE_STATUS_EXTERNAL,
    RUNTIME_STAGE_STATUS_MEASURED,
    RUNTIME_STAGE_STATUS_PARTIAL,
    RUN_SUMMARY_FIELDS,
    StageTimings,
    build_measurement_provenance_map,
    build_runtime_stage_entry,
    flatten_runtime_stage_breakdown,
    make_run_id,
    now_iso,
    prepare_run_directories,
    summarize_monitor_samples,
)
from common.system_monitor import SystemMonitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = PROJECT_ROOT.parent / "assets"
SCHEMA_VERSION = "v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    source_path: Path
    relative_path: Path

    @property
    def relative_key(self) -> str:
        return self.relative_path.as_posix()

    @property
    def file_name(self) -> str:
        return self.relative_path.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a thesis-oriented YOLO inference experiment.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--input-path", type=str, required=True, help="Image file or directory.")
    parser.add_argument("--labels-path", type=str, default=None, help="YOLO label directory for evaluation.")
    parser.add_argument("--weights", type=str, default=str(ASSETS_ROOT / "models" / "yolo" / "detection.pt"), help="Path to YOLO weights.")
    parser.add_argument("--dataset-name", type=str, default="soccersum", help="Dataset name stored in outputs.")
    parser.add_argument("--dataset-name-prefix", type=str, default=None, help="Optional base used to construct dataset_name as <base>_<device_slug>_b<batch_size>_rep<repeat_index>.")
    parser.add_argument("--repeat-index", type=int, default=None, help="Optional repeat index used with --dataset-name-prefix.")
    parser.add_argument("--device", type=str, default=None, help="Ultralytics device string, for example cuda:0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=None,
        help="Optional number of images per YOLO predict call. Default keeps the existing single-call behavior.",
    )
    parser.add_argument(
        "--predict-source-mode",
        type=str,
        default="list",
        choices=("list", "manifest"),
        help=(
            "How to hand prediction inputs to Ultralytics. "
            "'list' preserves the original behavior, while 'manifest' writes a temporary .txt file of image paths "
            "and lets Ultralytics stream them via its file loader."
        ),
    )
    parser.add_argument("--recursive-input", action="store_true", help="Recursively collect images from nested subfolders under the input root.")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated images.")
    parser.add_argument(
        "--skip-quality-eval",
        action="store_true",
        help="Skip Ultralytics validation. Intended for exploratory profiling only; default thesis runs should leave this off.",
    )
    parser.add_argument("--disable-system-monitor", action="store_true", help="Disable background system monitoring.")
    parser.add_argument("--monitor-interval-sec", type=float, default=0.5, help="System monitoring interval in seconds.")
    parser.add_argument("--outputs-root", type=str, default=str(PROJECT_ROOT / "outputs"), help="Experiment output root.")
    parser.add_argument("--notes", type=str, default=None, help="Optional notes stored with the run.")
    return parser


def collect_images(input_path: str | Path, recursive_input: bool) -> list[ImageRecord]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image file: {path}")
        return [ImageRecord(source_path=path, relative_path=Path(path.name))]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    if recursive_input:
        image_paths = sorted(
            (item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda item: item.relative_to(path).as_posix(),
        )
    else:
        image_paths = sorted(
            (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda item: item.name,
        )

    if not image_paths:
        raise FileNotFoundError(f"No supported images found under: {path}")
    return [ImageRecord(source_path=item, relative_path=item.relative_to(path)) for item in image_paths]


def chunk_image_records(image_records: list[ImageRecord], chunk_size: int | None) -> list[list[ImageRecord]]:
    if chunk_size is None:
        return [image_records]
    if chunk_size <= 0:
        raise ValueError("--predict-chunk-size must be a positive integer when provided.")
    return [image_records[index : index + chunk_size] for index in range(0, len(image_records), chunk_size)]


def records_for_manifest_mode(image_records: list[ImageRecord]) -> list[ImageRecord]:
    return sorted(image_records, key=lambda record: str(record.source_path.absolute()))


def create_predict_manifest(run_dir: Path, image_records: list[ImageRecord]) -> Path:
    manifest_path = run_dir / "metrics" / "predict_sources.txt"
    manifest_path.write_text(
        "\n".join(str(record.source_path.resolve()) for record in image_records) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def resolved_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def dataset_name_was_explicit(cli_args: list[str], loaded_config: dict[str, Any]) -> bool:
    return any(arg == "--dataset-name" or arg.startswith("--dataset-name=") for arg in cli_args) or "dataset_name" in loaded_config


def dataset_device_slug(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        return "gpu"
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "device"


def resolve_dataset_name(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    loaded_config: dict[str, Any],
    cli_args: list[str],
) -> str:
    if dataset_name_was_explicit(cli_args, loaded_config):
        return args.dataset_name

    helper_requested = args.dataset_name_prefix is not None or args.repeat_index is not None
    if not helper_requested:
        return args.dataset_name

    if args.dataset_name_prefix is None or args.repeat_index is None:
        parser.error("--dataset-name-prefix and --repeat-index must be provided together when using the dataset naming helper.")

    return f"{args.dataset_name_prefix}_{dataset_device_slug(args.device)}_b{args.batch_size}_rep{args.repeat_index}"


def extract_predictions(
    results: list[Any],
    image_records: list[ImageRecord],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, tuple[int, int]], float]:
    if len(results) != len(image_records):
        raise RuntimeError(
            "YOLO prediction result count does not match the collected input image count. "
            "The wrapper relies on stable result ordering for recursive runs."
        )

    predictions_by_image: dict[str, list[dict[str, Any]]] = {}
    flat_predictions: list[dict[str, Any]] = []
    image_shapes: dict[str, tuple[int, int]] = {}
    postprocess_time_sec = 0.0

    for record, result in zip(image_records, results):
        image_key = record.relative_key
        image_shapes[image_key] = tuple(result.orig_shape)
        postprocess_time_sec += float(result.speed.get("postprocess", 0.0)) / 1000.0
        rows: list[dict[str, Any]] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            row = {
                "image_name": record.file_name,
                "relative_path": image_key,
                "class_id": class_id,
                "class_name": result.names.get(class_id, str(class_id)),
                "confidence": confidence,
                "xyxy": xyxy,
            }
            rows.append(row)
            flat_predictions.append(
                {
                    "image_name": record.file_name,
                    "relative_path": image_key,
                    "source_path": str(record.source_path),
                    "class_id": class_id,
                    "class_name": row["class_name"],
                    "confidence": confidence,
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                }
            )
        predictions_by_image[image_key] = rows
    return predictions_by_image, flat_predictions, image_shapes, postprocess_time_sec


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def temporary_dataset_filename(relative_path: Path) -> str:
    relative_key = relative_path.as_posix()
    digest = hashlib.sha1(relative_key.encode("utf-8")).hexdigest()[:10]
    stem = "__".join(relative_path.with_suffix("").parts)
    return f"{stem}__{digest}{relative_path.suffix.lower()}"


def create_ultralytics_dataset_view(
    run_dir: Path,
    image_records: list[ImageRecord],
    labels_dir: Path,
    names: dict[int, str],
) -> Path:
    dataset_root = run_dir / "metrics" / "ultralytics_val_dataset"
    images_dir = dataset_root / "images" / "val"
    labels_target_dir = dataset_root / "labels" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_target_dir.mkdir(parents=True, exist_ok=True)

    for record in image_records:
        temp_name = temporary_dataset_filename(record.relative_path)
        target_image = images_dir / temp_name
        link_or_copy(record.source_path, target_image)
        source_label = labels_dir / record.relative_path.with_suffix(".txt")
        target_label = labels_target_dir / Path(temp_name).with_suffix(".txt")
        if source_label.is_file():
            link_or_copy(source_label, target_label)
        else:
            target_label.write_text("", encoding="utf-8")

    names_lines = "\n".join(f"  {class_id}: {name}" for class_id, name in sorted(names.items()))
    yaml_path = dataset_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {dataset_root.as_posix()}\ntrain: images/val\nval: images/val\nnames:\n{names_lines}\n",
        encoding="utf-8",
    )
    return yaml_path


def create_ultralytics_label_cache(
    dataset_root: Path,
    image_records: list[ImageRecord],
    names: dict[int, str],
) -> Path:
    labels_dir = dataset_root / "labels" / "val"
    cache_path = labels_dir.with_suffix(".cache")
    x: dict[str, Any] = {"labels": []}
    nm = nf = ne = nc = 0
    msgs: list[str] = []
    im_files = [str(dataset_root / "images" / "val" / temporary_dataset_filename(record.relative_path)) for record in image_records]
    label_files = [
        str(labels_dir / Path(temporary_dataset_filename(record.relative_path)).with_suffix(".txt"))
        for record in image_records
    ]

    for result in map(
        verify_image_label,
        zip(
            im_files,
            label_files,
            [""] * len(im_files),
            [False] * len(im_files),
            [len(names)] * len(im_files),
            [0] * len(im_files),
            [0] * len(im_files),
            [False] * len(im_files),
        ),
    ):
        im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg = result
        nm += nm_f
        nf += nf_f
        ne += ne_f
        nc += nc_f
        if im_file:
            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": shape,
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": segments,
                    "keypoints": keypoint,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if msg:
            msgs.append(msg)

    x["hash"] = get_hash(label_files + im_files)
    x["results"] = (nf, nm, ne, nc, len(im_files))
    x["msgs"] = msgs
    save_dataset_cache_file("", cache_path, x, DATASET_CACHE_VERSION)
    return cache_path


def compute_quality_metrics(
    model: YOLO,
    image_records: list[ImageRecord],
    labels_path: str | Path | None,
    args: argparse.Namespace,
    run_dirs: dict[str, Path],
) -> tuple[dict[str, Any] | None, str | None, float]:
    if not labels_path:
        return None, None, 0.0

    labels_dir = Path(labels_path)
    if not labels_dir.is_dir():
        return None, None, 0.0

    quality_started = perf_counter()
    names = {int(key): str(value) for key, value in model.names.items()}
    mpl_config_dir = run_dirs["metrics"] / "mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    original_mpl = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    try:
        dataset_yaml = create_ultralytics_dataset_view(
            run_dir=run_dirs["run_root"],
            image_records=image_records,
            labels_dir=labels_dir,
            names=names,
        )
        create_ultralytics_label_cache(dataset_yaml.parent, image_records, names)
        metrics = model.val(
            data=str(dataset_yaml),
            split="val",
            imgsz=args.imgsz,
            conf=args.conf,
            batch=args.batch_size,
            device=args.device,
            workers=0,
            verbose=False,
            plots=False,
        )
        quality_payload = {
            "source": "ultralytics_val",
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "f1": None,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
        elapsed = perf_counter() - quality_started
        return quality_payload, "ultralytics_val", elapsed
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics validation failed in the main YOLO thesis runner. "
            "This runner is locked to ultralytics_val only. "
            "Use the separate debug evaluation path if you need wrapper-side diagnostics."
        ) from exc
    finally:
        if original_mpl is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = original_mpl


def main() -> None:
    parser = build_parser()
    cli_args = sys.argv[1:]
    args = parser.parse_args()
    args, loaded_config = apply_config_defaults(args, parser)
    args.device = resolved_device(args.device)
    args.dataset_name = resolve_dataset_name(args, parser, loaded_config, cli_args)
    if args.predict_source_mode == "manifest" and args.predict_chunk_size is not None:
        parser.error("--predict-chunk-size cannot be combined with --predict-source-mode manifest.")

    input_collection_started = perf_counter()
    image_records = collect_images(args.input_path, recursive_input=args.recursive_input)
    input_collection_sec = perf_counter() - input_collection_started
    run_id = make_run_id("yolo")
    run_dirs = prepare_run_directories(args.outputs_root, run_id)
    config_snapshot = namespace_to_dict(args, exclude={"config"})
    environment = collect_environment_metadata(resolved_device=args.device, include_ultralytics=True)
    measurement_provenance = build_measurement_provenance_map(
        {
            "total_runtime_sec": PROVENANCE_WRAPPER_MEASURED,
            "input_collection_sec": PROVENANCE_WRAPPER_MEASURED,
            "model_load_sec": PROVENANCE_WRAPPER_MEASURED,
            "input_loading_preparation_sec": PROVENANCE_WRAPPER_MEASURED,
            "predict_wall_time_sec": PROVENANCE_WRAPPER_MEASURED,
            "model_reported_total_sec": PROVENANCE_DERIVED,
            "predict_overhead_sec": PROVENANCE_DERIVED,
            "decode_time_sec": PROVENANCE_UNAVAILABLE,
            "preprocess_time_sec": PROVENANCE_MODEL_REPORTED,
            "inference_time_sec": PROVENANCE_MODEL_REPORTED,
            "postprocess_time_sec": PROVENANCE_MODEL_REPORTED,
            "predictions_json_write_sec": PROVENANCE_WRAPPER_MEASURED,
            "predictions_csv_write_sec": PROVENANCE_WRAPPER_MEASURED,
            "write_time_sec": PROVENANCE_WRAPPER_MEASURED,
            "quality_eval_time_sec": PROVENANCE_WRAPPER_MEASURED,
            "end_to_end_runtime_sec": PROVENANCE_DERIVED,
            "stage_upload_download_sec": PROVENANCE_UNAVAILABLE,
            "stage_frame_extraction_preprocessing_sec": PROVENANCE_DERIVED,
            "stage_model_inference_sec": PROVENANCE_DERIVED,
            "stage_postprocess_output_sec": PROVENANCE_DERIVED,
            "throughput_fps": PROVENANCE_WRAPPER_MEASURED,
            "throughput_clips_per_sec": PROVENANCE_UNAVAILABLE,
            "gpu_util_avg": PROVENANCE_HOST_OBSERVED,
            "gpu_mem_mb_avg": PROVENANCE_HOST_OBSERVED,
            "gpu_mem_util_pct_avg": PROVENANCE_HOST_OBSERVED,
            "gpu_mem_total_mb": PROVENANCE_HOST_OBSERVED,
            "cpu_util_avg": PROVENANCE_HOST_OBSERVED,
            "ram_mb_avg": PROVENANCE_HOST_OBSERVED,
            "ram_util_pct_avg": PROVENANCE_HOST_OBSERVED,
            "ram_total_mb": PROVENANCE_HOST_OBSERVED,
            "disk_read_mb_per_sec_avg": PROVENANCE_HOST_OBSERVED,
            "disk_write_mb_per_sec_avg": PROVENANCE_HOST_OBSERVED,
        }
    )

    model_load_started = perf_counter()
    model = YOLO(args.weights)
    model_load_sec = perf_counter() - model_load_started
    input_loading_preparation_sec = input_collection_sec + model_load_sec
    monitor = None if args.disable_system_monitor else SystemMonitor(interval_sec=args.monitor_interval_sec)
    if monitor is not None:
        monitor.start()

    stage_timings = StageTimings()
    chunk_metrics: list[dict[str, Any]] = []
    start_time = now_iso()
    total_started = perf_counter()
    results_list: list[Any] = []
    records_for_predict = image_records
    if args.predict_source_mode == "manifest":
        records_for_predict = records_for_manifest_mode(image_records)
        predict_manifest_path = create_predict_manifest(run_dirs["run_root"], records_for_predict)
        predict_results = model.predict(
            source=str(predict_manifest_path),
            conf=args.conf,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            save=False,
            verbose=False,
            stream=False,
        )
        results_list = list(predict_results)
    else:
        for chunk_index, image_chunk in enumerate(chunk_image_records(image_records, args.predict_chunk_size), start=1):
            chunk_started = perf_counter()
            chunk_results = model.predict(
                source=[str(record.source_path) for record in image_chunk],
                conf=args.conf,
                imgsz=args.imgsz,
                batch=args.batch_size,
                device=args.device,
                save=False,
                verbose=False,
                stream=False,
            )
            chunk_elapsed = perf_counter() - chunk_started
            chunk_results_list = list(chunk_results)
            chunk_preprocess_time_sec = sum(float(item.speed.get("preprocess", 0.0)) for item in chunk_results_list) / 1000.0
            chunk_inference_time_sec = sum(float(item.speed.get("inference", 0.0)) for item in chunk_results_list) / 1000.0
            chunk_postprocess_time_sec = sum(float(item.speed.get("postprocess", 0.0)) for item in chunk_results_list) / 1000.0
            chunk_model_reported_total_sec = (
                chunk_preprocess_time_sec + chunk_inference_time_sec + chunk_postprocess_time_sec
            )
            chunk_predict_overhead_sec = max(chunk_elapsed - chunk_model_reported_total_sec, 0.0)
            chunk_metrics.append(
                {
                    "chunk_index": chunk_index,
                    "num_images": len(image_chunk),
                    "predict_wall_time_sec": round(chunk_elapsed, 6),
                    "preprocess_time_sec": round(chunk_preprocess_time_sec, 6),
                    "inference_time_sec": round(chunk_inference_time_sec, 6),
                    "postprocess_time_sec": round(chunk_postprocess_time_sec, 6),
                    "model_reported_total_sec": round(chunk_model_reported_total_sec, 6),
                    "predict_overhead_sec": round(chunk_predict_overhead_sec, 6),
                }
            )
            results_list.extend(chunk_results_list)
    predict_elapsed = perf_counter() - total_started

    # Ultralytics reports per-image preprocess and inference times in milliseconds.
    # We aggregate those model-reported values across the batch for method transparency.
    preprocess_time_sec = sum(float(item.speed.get("preprocess", 0.0)) for item in results_list) / 1000.0
    inference_time_sec = sum(float(item.speed.get("inference", 0.0)) for item in results_list) / 1000.0
    predictions_by_image, flat_predictions, image_shapes, postprocess_time_sec = extract_predictions(results_list, records_for_predict)
    stage_timings.add("preprocess_time_sec", preprocess_time_sec)
    stage_timings.add("inference_time_sec", inference_time_sec)
    stage_timings.add("postprocess_time_sec", postprocess_time_sec)
    model_reported_total_sec = preprocess_time_sec + inference_time_sec + postprocess_time_sec
    predict_overhead_sec = max(predict_elapsed - model_reported_total_sec, 0.0)

    write_started = perf_counter()
    predictions_json_path = run_dirs["predictions"] / "predictions.json"
    predictions_csv_path = run_dirs["predictions"] / "predictions.csv"
    json_write_started = perf_counter()
    write_json(predictions_json_path, predictions_by_image)
    predictions_json_write_sec = perf_counter() - json_write_started
    csv_write_started = perf_counter()
    write_csv_rows(
        predictions_csv_path,
        ["image_name", "relative_path", "source_path", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
        flat_predictions,
    )
    predictions_csv_write_sec = perf_counter() - csv_write_started
    if args.save_annotated:
        import cv2

        annotated_dir = run_dirs["predictions"] / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        for record, result in zip(image_records, results_list):
            output_path = annotated_dir / record.relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result.plot())
    stage_timings.add("write_time_sec", perf_counter() - write_started)

    total_runtime_sec = predict_elapsed + stage_timings.get("write_time_sec")
    end_time = now_iso()
    samples = monitor.stop() if monitor is not None else []
    if monitor is not None:
        monitor.save_csv(str(run_dirs["system"] / "system_samples.csv"))
    system_summary = summarize_monitor_samples(samples)
    gpu_stats_available = any(sample.get("gpu_util_pct") is not None for sample in samples)

    chunk_metrics_path = None
    if args.predict_chunk_size is not None:
        chunk_metrics_path = run_dirs["metrics"] / "predict_chunks.csv"
        write_csv_rows(
            chunk_metrics_path,
            [
                "chunk_index",
                "num_images",
                "predict_wall_time_sec",
                "preprocess_time_sec",
                "inference_time_sec",
                "postprocess_time_sec",
                "model_reported_total_sec",
                "predict_overhead_sec",
            ],
            chunk_metrics,
        )

    del results_list
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.skip_quality_eval:
        quality_metrics, quality_source, quality_eval_time_sec = None, None, 0.0
    else:
        quality_metrics, quality_source, quality_eval_time_sec = compute_quality_metrics(
            model=model,
            image_records=image_records,
            labels_path=args.labels_path,
            args=args,
            run_dirs=run_dirs,
        )
    end_to_end_runtime_sec = input_loading_preparation_sec + total_runtime_sec + quality_eval_time_sec

    precision = quality_metrics.get("precision") if quality_metrics else None
    recall = quality_metrics.get("recall") if quality_metrics else None
    f1 = quality_metrics.get("f1") if quality_metrics else None
    map50 = quality_metrics.get("map50") if quality_metrics else None
    map50_95 = quality_metrics.get("map50_95") if quality_metrics else None
    runtime_stage_breakdown = {
        "upload_download": build_runtime_stage_entry(
            seconds=None,
            status=RUNTIME_STAGE_STATUS_EXTERNAL,
            components=[],
            note="Upload and download are not executed inside the YOLO thesis runner.",
        ),
        "frame_extraction_preprocessing": build_runtime_stage_entry(
            seconds=stage_timings.get("preprocess_time_sec"),
            status=RUNTIME_STAGE_STATUS_PARTIAL,
            components=["preprocess_time_sec"],
            note=(
                "The YOLO runner consumes already extracted image frames. "
                "This stage therefore includes preprocessing only; frame extraction is external to the runner."
            ),
        ),
        "model_inference": build_runtime_stage_entry(
            seconds=stage_timings.get("inference_time_sec"),
            status=RUNTIME_STAGE_STATUS_MEASURED,
            components=["inference_time_sec"],
            note="Derived from model-reported per-image inference timings.",
        ),
        "postprocess_output": build_runtime_stage_entry(
            seconds=stage_timings.get("postprocess_time_sec") + stage_timings.get("write_time_sec"),
            status=RUNTIME_STAGE_STATUS_MEASURED,
            components=["postprocess_time_sec", "write_time_sec"],
            note="Combines model-reported postprocess time with wrapper-measured output writing.",
        ),
    }
    runtime_stage_summary = flatten_runtime_stage_breakdown(runtime_stage_breakdown)

    summary_row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_name": "yolo",
        "dataset_name": args.dataset_name,
        "input_path": str(Path(args.input_path).resolve()),
        "num_images": len(image_records),
        "num_videos": None,
        "num_frames": None,
        "batch_size": args.batch_size,
        "image_size": args.imgsz,
        "confidence_threshold": args.conf,
        "device": args.device,
        "start_time": start_time,
        "end_time": end_time,
        "total_runtime_sec": round(total_runtime_sec, 6),
        "input_collection_sec": round(input_collection_sec, 6),
        "model_load_sec": round(model_load_sec, 6),
        "input_loading_preparation_sec": round(input_loading_preparation_sec, 6),
        "predict_wall_time_sec": round(predict_elapsed, 6),
        "model_reported_total_sec": round(model_reported_total_sec, 6),
        "predict_overhead_sec": round(predict_overhead_sec, 6),
        "decode_time_sec": None,
        "preprocess_time_sec": round(stage_timings.get("preprocess_time_sec"), 6),
        "inference_time_sec": round(stage_timings.get("inference_time_sec"), 6),
        "postprocess_time_sec": round(stage_timings.get("postprocess_time_sec"), 6),
        "predictions_json_write_sec": round(predictions_json_write_sec, 6),
        "predictions_csv_write_sec": round(predictions_csv_write_sec, 6),
        "write_time_sec": round(stage_timings.get("write_time_sec"), 6),
        "quality_eval_time_sec": round(quality_eval_time_sec, 6),
        "end_to_end_runtime_sec": round(end_to_end_runtime_sec, 6),
        **runtime_stage_summary,
        "throughput_fps": round(len(image_records) / total_runtime_sec, 6) if total_runtime_sec > 0 else None,
        "throughput_clips_per_sec": None,
        "gpu_util_avg": system_summary["gpu_util_avg"],
        "gpu_mem_mb_avg": system_summary["gpu_mem_mb_avg"],
        "gpu_mem_util_pct_avg": system_summary["gpu_mem_util_pct_avg"],
        "gpu_mem_total_mb": system_summary["gpu_mem_total_mb"],
        "cpu_util_avg": system_summary["cpu_util_avg"],
        "ram_mb_avg": system_summary["ram_mb_avg"],
        "ram_util_pct_avg": system_summary["ram_util_pct_avg"],
        "ram_total_mb": system_summary["ram_total_mb"],
        "disk_read_mb_per_sec_avg": system_summary["disk_read_mb_per_sec_avg"],
        "disk_write_mb_per_sec_avg": system_summary["disk_write_mb_per_sec_avg"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "stride": None,
        "clip_length": None,
        "fps_target": None,
        "quality_metric_source": quality_source,
        "gpu_stats_available": gpu_stats_available,
        "notes": args.notes,
    }

    write_json(run_dirs["metrics"] / "config_snapshot.json", {"loaded_config": loaded_config, "resolved_args": config_snapshot})
    write_json(
        run_dirs["metrics"] / "run_metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "model_name": "yolo",
            "dataset_name": args.dataset_name,
            "input_path": str(Path(args.input_path).resolve()),
            "labels_path": str(Path(args.labels_path).resolve()) if args.labels_path else None,
            "weights": str(Path(args.weights).resolve()),
            "environment": environment,
            "measurement_provenance": measurement_provenance,
            "system_monitoring": {
                "scope": "host_level",
                "process_level_stats": False,
                "sampling_interval_sec": args.monitor_interval_sec,
                "nvidia_smi_available": environment["nvidia_smi_available"],
            },
            "model_interaction": {
                "checkpoint_read_only": True,
                "vendor_code_modified": False,
                "checkpoint_modified": False,
                "quality_metric_policy": "skipped_via_flag" if args.skip_quality_eval else "ultralytics_val_only_in_main_runner",
                "predict_source_mode": args.predict_source_mode,
            },
            "files": {
                "predictions_json": str(predictions_json_path),
                "predictions_csv": str(predictions_csv_path),
                "system_samples_csv": str(run_dirs["system"] / "system_samples.csv") if samples else None,
                "predict_chunks_csv": str(chunk_metrics_path) if chunk_metrics_path else None,
                "predict_sources_txt": str(run_dirs["metrics"] / "predict_sources.txt") if args.predict_source_mode == "manifest" else None,
            },
        },
    )
    write_json(
        run_dirs["metrics"] / "run_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "summary": summary_row,
            "measurement_provenance": measurement_provenance,
            "timings": {
                "total_runtime_sec": round(total_runtime_sec, 6),
                "input_collection_sec": round(input_collection_sec, 6),
                "model_load_sec": round(model_load_sec, 6),
                "input_loading_preparation_sec": round(input_loading_preparation_sec, 6),
                "predict_wall_time_sec": round(predict_elapsed, 6),
                "model_reported_total_sec": round(model_reported_total_sec, 6),
                "predict_overhead_sec": round(predict_overhead_sec, 6),
                "preprocess_time_sec": round(stage_timings.get("preprocess_time_sec"), 6),
                "inference_time_sec": round(stage_timings.get("inference_time_sec"), 6),
                "postprocess_time_sec": round(stage_timings.get("postprocess_time_sec"), 6),
                "predictions_json_write_sec": round(predictions_json_write_sec, 6),
                "predictions_csv_write_sec": round(predictions_csv_write_sec, 6),
                "write_time_sec": round(stage_timings.get("write_time_sec"), 6),
                "quality_eval_time_sec": round(quality_eval_time_sec, 6),
                "end_to_end_runtime_sec": round(end_to_end_runtime_sec, 6),
            },
            "runtime_stage_breakdown": runtime_stage_breakdown,
            "quality": quality_metrics,
            "system": {
                "scope": "host_level",
                "process_level_stats": False,
                "sampling_interval_sec": args.monitor_interval_sec,
                "gpu_stats_available": gpu_stats_available,
                "sample_count": len(samples),
                "summary": system_summary,
            },
        },
    )
    write_csv_rows(run_dirs["metrics"] / "run_summary.csv", RUN_SUMMARY_FIELDS, [summary_row])

    print(
        f"run_id={run_id} model=yolo images={len(image_records)} total_runtime_sec={summary_row['total_runtime_sec']} "
        f"precision={precision} recall={recall} map50={map50} quality_source={quality_source}"
    )


if __name__ == "__main__":
    main()
