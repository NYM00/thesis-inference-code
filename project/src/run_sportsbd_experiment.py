from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from common.config import apply_config_defaults, namespace_to_dict
from common.environment import collect_environment_metadata
from common.io_utils import write_csv_rows, write_json
from common.runtime import (
    PROVENANCE_DERIVED,
    PROVENANCE_HOST_OBSERVED,
    PROVENANCE_WRAPPER_MEASURED,
    PROVENANCE_UNAVAILABLE,
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
from sportsbd.evaluation import evaluate_transitions, load_ground_truth_transitions
from sportsbd.inference import iter_video_records, run_video_inference
from sportsbd.modeling import load_checkpoint_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = PROJECT_ROOT.parent / "assets"
SCHEMA_VERSION = "v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a thesis-oriented SportSBD inference experiment.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--input-path", type=str, required=True, help="Video file or dataset directory.")
    parser.add_argument("--weights", type=str, default=str(ASSETS_ROOT / "models" / "sportsbd" / "best.pt"), help="Path to SportSBD checkpoint.")
    parser.add_argument("--dataset-name", type=str, default="sportsbd", help="Dataset name stored in outputs.")
    parser.add_argument("--dataset-name-prefix", type=str, default=None, help="Optional base prefix for auto-generated dataset_name values.")
    parser.add_argument("--repeat-index", type=int, default=None, help="Optional repeat index used for auto-generated dataset_name values.")
    parser.add_argument("--device", type=str, default=None, help="Execution device: cpu, cuda:0, or neuron.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold.")
    parser.add_argument("--stride", type=int, default=4, help="Sliding-window stride in frames.")
    parser.add_argument("--clip-length", type=int, default=16, help="Frames per clip.")
    parser.add_argument("--fps", type=float, default=25.0, help="Target FPS for decoding and frame indexing.")
    parser.add_argument("--match-tolerance-frames", type=int, default=None, help="Transition matching tolerance. Defaults to clip_length // 2.")
    parser.add_argument("--limit-videos", type=int, default=None, help="Optional cap on number of videos processed.")
    parser.add_argument("--disable-system-monitor", action="store_true", help="Disable background system monitoring.")
    parser.add_argument("--monitor-interval-sec", type=float, default=0.5, help="System monitoring interval in seconds.")
    parser.add_argument("--outputs-root", type=str, default=str(PROJECT_ROOT / "outputs"), help="Experiment output root.")
    parser.add_argument(
        "--neuron-cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "neuron_cache"),
        help="Directory where compiled Neuron artifacts are stored.",
    )
    parser.add_argument(
        "--force-neuron-recompile",
        action="store_true",
        help="Force recompilation of the saved Neuron artifact.",
    )
    parser.add_argument("--notes", type=str, default=None, help="Optional notes stored with the run.")
    return parser


def resolved_device(device_arg: str | None) -> str:
    if device_arg:
        normalized = device_arg.strip().lower()
        if normalized == "neuron":
            return "neuron"
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _arg_was_provided(flag: str) -> bool:
    return any(argument == flag or argument.startswith(f"{flag}=") for argument in sys.argv[1:])


def _slugify_decimal(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _device_slug(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        return "gpu"
    if normalized == "neuron":
        return "neuron"
    return normalized.replace(":", "_")


def resolve_dataset_name(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    loaded_config: dict[str, Any],
) -> str:
    dataset_name_explicit = _arg_was_provided("--dataset-name") or "dataset_name" in loaded_config
    if dataset_name_explicit:
        return args.dataset_name

    helper_prefix = args.dataset_name_prefix
    helper_repeat = args.repeat_index
    if helper_prefix is None and helper_repeat is None:
        return args.dataset_name
    if helper_prefix is None or helper_repeat is None:
        parser.error("--dataset-name-prefix and --repeat-index must be provided together when using helper naming.")

    limit_slug = str(args.limit_videos) if args.limit_videos is not None else "all"
    return (
        f"{helper_prefix}_"
        f"{_device_slug(args.device)}_"
        f"t{_slugify_decimal(args.threshold)}_"
        f"s{args.stride}_"
        f"l{args.clip_length}_"
        f"fps{_slugify_decimal(args.fps)}_"
        f"n{limit_slug}_"
        f"rep{helper_repeat}"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args, loaded_config = apply_config_defaults(args, parser)
    args.device = resolved_device(args.device)
    args.match_tolerance_frames = args.match_tolerance_frames if args.match_tolerance_frames is not None else args.clip_length // 2
    args.dataset_name = resolve_dataset_name(parser, args, loaded_config)

    input_collection_started = perf_counter()
    video_records = iter_video_records(args.input_path)
    if args.limit_videos is not None:
        video_records = video_records[: args.limit_videos]
    if not video_records:
        raise RuntimeError("No SportSBD videos selected for inference.")
    input_collection_sec = perf_counter() - input_collection_started

    run_id = make_run_id("sportsbd")
    run_dirs = prepare_run_directories(args.outputs_root, run_id)
    config_snapshot = namespace_to_dict(args, exclude={"config", "dataset_name_prefix", "repeat_index"})
    environment = collect_environment_metadata(resolved_device=args.device, include_ultralytics=False)
    measurement_provenance = build_measurement_provenance_map(
        {
            "input_collection_sec": PROVENANCE_WRAPPER_MEASURED,
            "model_load_sec": PROVENANCE_WRAPPER_MEASURED,
            "input_loading_preparation_sec": PROVENANCE_WRAPPER_MEASURED,
            "predict_wall_time_sec": PROVENANCE_WRAPPER_MEASURED,
            "model_reported_total_sec": PROVENANCE_DERIVED,
            "predict_overhead_sec": PROVENANCE_DERIVED,
            "total_runtime_sec": PROVENANCE_WRAPPER_MEASURED,
            "decode_time_sec": PROVENANCE_WRAPPER_MEASURED,
            "preprocess_time_sec": PROVENANCE_WRAPPER_MEASURED,
            "inference_time_sec": PROVENANCE_WRAPPER_MEASURED,
            "postprocess_time_sec": PROVENANCE_UNAVAILABLE,
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
            "throughput_clips_per_sec": PROVENANCE_WRAPPER_MEASURED,
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
    model_bundle = load_checkpoint_bundle(
        args.weights,
        device=args.device,
        clip_length=args.clip_length,
        compile_artifacts_dir=args.neuron_cache_dir,
        force_recompile=args.force_neuron_recompile,
    )
    model_load_sec = perf_counter() - model_load_started
    input_loading_preparation_sec = input_collection_sec + model_load_sec

    monitor = None if args.disable_system_monitor else SystemMonitor(interval_sec=args.monitor_interval_sec)
    if monitor is not None:
        monitor.start()

    stage_timings = StageTimings()
    start_time = now_iso()
    total_started = perf_counter()
    all_predictions: list[dict[str, Any]] = []
    per_video_results: list[dict[str, Any]] = []
    total_sampled_frames = 0
    total_windows = 0

    for record in video_records:
        inference_result = run_video_inference(
            video_path=record["video_path"],
            model_bundle=model_bundle,
            threshold=args.threshold,
            stride=args.stride,
            clip_length=args.clip_length,
            target_fps=args.fps,
        )
        stage_timings.add("decode_time_sec", inference_result["timings"].get("decode_time_sec", 0.0))
        stage_timings.add("preprocess_time_sec", inference_result["timings"].get("preprocess_time_sec", 0.0))
        stage_timings.add("inference_time_sec", inference_result["timings"].get("inference_time_sec", 0.0))
        total_sampled_frames += int(inference_result["sampled_frame_count"])
        total_windows += int(inference_result["window_count"])

        video_predictions = []
        for item in inference_result["detections"]:
            enriched = dict(item)
            enriched["video_id"] = record["video_id"]
            enriched["video_path"] = str(record["video_path"])
            video_predictions.append(enriched)
            all_predictions.append(enriched)

        per_video_results.append(
            {
                "video_id": record["video_id"],
                "video_path": str(record["video_path"]),
                "annotation_path": str(record["annotation_path"]) if record["annotation_path"] else None,
                "sampled_frame_count": inference_result["sampled_frame_count"],
                "window_count": inference_result["window_count"],
                "prediction_count": len(video_predictions),
                "source_fps": inference_result["source_fps"],
                "sample_fps": inference_result["sample_fps"],
            }
        )

    predict_wall_time_sec = perf_counter() - total_started
    model_reported_total_sec = (
        stage_timings.get("decode_time_sec")
        + stage_timings.get("preprocess_time_sec")
        + stage_timings.get("inference_time_sec")
    )
    predict_overhead_sec = max(predict_wall_time_sec - model_reported_total_sec, 0.0)

    write_started = perf_counter()
    predictions_json_path = run_dirs["predictions"] / "predictions.json"
    predictions_csv_path = run_dirs["predictions"] / "predictions.csv"
    json_write_started = perf_counter()
    write_json(predictions_json_path, all_predictions)
    predictions_json_write_sec = perf_counter() - json_write_started
    csv_write_started = perf_counter()
    write_csv_rows(
        predictions_csv_path,
        [
            "video_id",
            "video_path",
            "frame_idx",
            "source_frame_idx",
            "timestamp_ms",
            "predicted_class",
            "predicted_index",
            "confidence",
            "class_probs",
        ],
        all_predictions,
    )
    predictions_csv_write_sec = perf_counter() - csv_write_started
    stage_timings.add("write_time_sec", perf_counter() - write_started)

    total_runtime_sec = perf_counter() - total_started
    end_time = now_iso()
    samples = monitor.stop() if monitor is not None else []
    if monitor is not None:
        monitor.save_csv(str(run_dirs["system"] / "system_samples.csv"))
    system_summary = summarize_monitor_samples(samples)
    gpu_stats_available = any(sample.get("gpu_util_pct") is not None for sample in samples)

    ground_truth_items: list[dict[str, Any]] = []
    for record in video_records:
        annotation_path = record["annotation_path"]
        if not annotation_path:
            continue
        for item in load_ground_truth_transitions(annotation_path):
            enriched_gt = dict(item)
            enriched_gt["video_id"] = record["video_id"]
            ground_truth_items.append(enriched_gt)

    quality_metrics = None
    quality_source = None
    quality_eval_time_sec = 0.0
    if ground_truth_items:
        quality_eval_started = perf_counter()
        quality_metrics = evaluate_transitions(
            predictions=all_predictions,
            ground_truth=ground_truth_items,
            tolerance_frames=args.match_tolerance_frames,
        )
        quality_eval_time_sec = perf_counter() - quality_eval_started
        quality_source = "interval_match_v1"
    end_to_end_runtime_sec = input_loading_preparation_sec + total_runtime_sec + quality_eval_time_sec
    runtime_stage_breakdown = {
        "upload_download": build_runtime_stage_entry(
            seconds=None,
            status=RUNTIME_STAGE_STATUS_EXTERNAL,
            components=[],
            note="Upload and download are not executed inside the SportSBD thesis runner.",
        ),
        "frame_extraction_preprocessing": build_runtime_stage_entry(
            seconds=stage_timings.get("decode_time_sec") + stage_timings.get("preprocess_time_sec"),
            status=RUNTIME_STAGE_STATUS_MEASURED,
            components=["decode_time_sec", "preprocess_time_sec"],
            note="Combines wrapper-measured video decode and preprocessing time.",
        ),
        "model_inference": build_runtime_stage_entry(
            seconds=stage_timings.get("inference_time_sec"),
            status=RUNTIME_STAGE_STATUS_MEASURED,
            components=["inference_time_sec"],
            note="Measured inside the current SportSBD inference loop.",
        ),
        "postprocess_output": build_runtime_stage_entry(
            seconds=stage_timings.get("write_time_sec"),
            status=RUNTIME_STAGE_STATUS_PARTIAL,
            components=["write_time_sec"],
            note=(
                "This runner measures saving outputs directly. "
                "No separate post-processing timer exists outside the inference path."
            ),
        ),
    }
    runtime_stage_summary = flatten_runtime_stage_breakdown(runtime_stage_breakdown)

    summary_row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_name": "sportsbd",
        "dataset_name": args.dataset_name,
        "input_path": str(Path(args.input_path).resolve()),
        "num_images": None,
        "num_videos": len(video_records),
        "num_frames": total_sampled_frames,
        "batch_size": None,
        "image_size": int(model_bundle["image_size"]),
        "confidence_threshold": args.threshold,
        "device": args.device,
        "start_time": start_time,
        "end_time": end_time,
        "total_runtime_sec": round(total_runtime_sec, 6),
        "input_collection_sec": round(input_collection_sec, 6),
        "model_load_sec": round(model_load_sec, 6),
        "input_loading_preparation_sec": round(input_loading_preparation_sec, 6),
        "predict_wall_time_sec": round(predict_wall_time_sec, 6),
        "model_reported_total_sec": round(model_reported_total_sec, 6),
        "predict_overhead_sec": round(predict_overhead_sec, 6),
        "decode_time_sec": round(stage_timings.get("decode_time_sec"), 6),
        "preprocess_time_sec": round(stage_timings.get("preprocess_time_sec"), 6),
        "inference_time_sec": round(stage_timings.get("inference_time_sec"), 6),
        "postprocess_time_sec": None,
        "predictions_json_write_sec": round(predictions_json_write_sec, 6),
        "predictions_csv_write_sec": round(predictions_csv_write_sec, 6),
        "write_time_sec": round(stage_timings.get("write_time_sec"), 6),
        "quality_eval_time_sec": round(quality_eval_time_sec, 6),
        "end_to_end_runtime_sec": round(end_to_end_runtime_sec, 6),
        **runtime_stage_summary,
        "throughput_fps": round(total_sampled_frames / total_runtime_sec, 6) if total_runtime_sec > 0 else None,
        "throughput_clips_per_sec": round(total_windows / total_runtime_sec, 6) if total_runtime_sec > 0 else None,
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
        "precision": quality_metrics.get("precision") if quality_metrics else None,
        "recall": quality_metrics.get("recall") if quality_metrics else None,
        "f1": quality_metrics.get("f1") if quality_metrics else None,
        "map50": None,
        "map50_95": None,
        "stride": args.stride,
        "clip_length": args.clip_length,
        "fps_target": args.fps,
        "quality_metric_source": quality_source,
        "gpu_stats_available": gpu_stats_available,
        "notes": args.notes,
    }

    write_json(
        run_dirs["metrics"] / "config_snapshot.json",
        {
            "loaded_config": loaded_config,
            "resolved_args": config_snapshot,
            "checkpoint_config": model_bundle["config"],
            "class_names": model_bundle["class_names"],
            "decoder_strategy": "opencv_direct",
        },
    )
    write_json(
        run_dirs["metrics"] / "run_metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "model_name": "sportsbd",
            "dataset_name": args.dataset_name,
            "input_path": str(Path(args.input_path).resolve()),
            "weights": str(Path(args.weights).resolve()),
            "environment": environment,
            "measurement_provenance": measurement_provenance,
            "system_monitoring": {
                "scope": "host_level",
                "process_level_stats": False,
                "sampling_interval_sec": args.monitor_interval_sec,
                "nvidia_smi_available": environment["nvidia_smi_available"],
            },
            "execution_path": {
                "checkpoint_read_only": True,
                "checkpoint_is_original": True,
                "checkpoint_modified": False,
                "vendor_code_modified": False,
                "decoding_strategy": "wrapper_controlled_opencv",
                "vendor_ffmpeg_launcher_used": False,
                "notes": (
                    "Inference uses a wrapper-controlled OpenCV decode path. "
                    "The vendor ffmpeg launcher path is not used, and no vendor code or checkpoint files are modified."
                ),
            },
            "files": {
                "predictions_json": str(predictions_json_path),
                "predictions_csv": str(predictions_csv_path),
                "system_samples_csv": str(run_dirs["system"] / "system_samples.csv") if samples else None,
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
                "input_collection_sec": round(input_collection_sec, 6),
                "model_load_sec": round(model_load_sec, 6),
                "input_loading_preparation_sec": round(input_loading_preparation_sec, 6),
                "predict_wall_time_sec": round(predict_wall_time_sec, 6),
                "model_reported_total_sec": round(model_reported_total_sec, 6),
                "predict_overhead_sec": round(predict_overhead_sec, 6),
                "total_runtime_sec": round(total_runtime_sec, 6),
                "decode_time_sec": round(stage_timings.get("decode_time_sec"), 6),
                "preprocess_time_sec": round(stage_timings.get("preprocess_time_sec"), 6),
                "inference_time_sec": round(stage_timings.get("inference_time_sec"), 6),
                "postprocess_time_sec": None,
                "predictions_json_write_sec": round(predictions_json_write_sec, 6),
                "predictions_csv_write_sec": round(predictions_csv_write_sec, 6),
                "write_time_sec": round(stage_timings.get("write_time_sec"), 6),
                "quality_eval_time_sec": round(quality_eval_time_sec, 6),
                "end_to_end_runtime_sec": round(end_to_end_runtime_sec, 6),
            },
            "runtime_stage_breakdown": runtime_stage_breakdown,
            "quality": {
                "source": quality_source,
                "matching_rule": {
                    "type": "interval_contains_prediction",
                    "tolerance_frames": args.match_tolerance_frames,
                    "category_scoped": True,
                    "selection": "highest_confidence_predictions_first_then_nearest_interval_center",
                },
                "metrics": quality_metrics,
            }
            if quality_metrics
            else None,
            "system": {
                "scope": "host_level",
                "process_level_stats": False,
                "sampling_interval_sec": args.monitor_interval_sec,
                "gpu_stats_available": gpu_stats_available,
                "sample_count": len(samples),
                "summary": system_summary,
            },
            "per_video": per_video_results,
        },
    )
    write_csv_rows(run_dirs["metrics"] / "run_summary.csv", RUN_SUMMARY_FIELDS, [summary_row])

    print(
        f"run_id={run_id} model=sportsbd videos={len(video_records)} total_runtime_sec={summary_row['total_runtime_sec']} "
        f"precision={summary_row['precision']} recall={summary_row['recall']} f1={summary_row['f1']}"
    )


if __name__ == "__main__":
    main()
