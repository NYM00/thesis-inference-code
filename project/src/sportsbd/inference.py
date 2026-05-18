from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import torch

from common.runtime import StageTimings

from .modeling import normalize_label_name, predict_clip_tensor, preprocess_bgr_frame


def iter_video_records(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    if path.is_file():
        annotation_path = path.parent.parent / "transitions.db"
        info_path = path.parent.parent / "video_info.json"
        return [
            {
                "video_path": path,
                "annotation_path": annotation_path if annotation_path.is_file() else None,
                "video_info_path": info_path if info_path.is_file() else None,
                "video_id": path.stem,
            }
        ]

    if not path.is_dir():
        raise FileNotFoundError(f"Video input path not found: {path}")

    records = []
    for video_file in sorted(path.glob("**/video/video.mp4")):
        parent = video_file.parent.parent
        annotation_path = parent / "transitions.db"
        info_path = parent / "video_info.json"
        records.append(
            {
                "video_path": video_file,
                "annotation_path": annotation_path if annotation_path.is_file() else None,
                "video_info_path": info_path if info_path.is_file() else None,
                "video_id": parent.name,
            }
        )
    if not records:
        raise FileNotFoundError(f"No SportSBD videos found under: {path}")
    return records


def _merge_nearby_detections(detections: list[dict[str, Any]], threshold_frames: int) -> list[dict[str, Any]]:
    if not detections:
        return detections
    sorted_items = sorted(detections, key=lambda item: item["frame_idx"])
    merged: list[dict[str, Any]] = []
    current = sorted_items[0]
    for candidate in sorted_items[1:]:
        if candidate["predicted_class"] == current["predicted_class"] and candidate["frame_idx"] - current["frame_idx"] <= threshold_frames:
            if candidate["confidence"] > current["confidence"]:
                current = candidate
            continue
        merged.append(current)
        current = candidate
    merged.append(current)
    return merged


def run_video_inference(
    video_path: str | Path,
    model_bundle: dict[str, Any],
    threshold: float,
    stride: int,
    clip_length: int,
    target_fps: float | None,
) -> dict[str, Any]:
    video_file = Path(video_path)
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or (target_fps or 25.0)
    sample_fps = float(target_fps or source_fps)
    target_interval = 1.0 / sample_fps if sample_fps > 0 else None
    next_sample_time = 0.0
    source_frame_index = -1
    sampled_frame_index = -1
    sampled_frame_count = 0
    window_count = 0
    detections: list[dict[str, Any]] = []
    stage_timings = StageTimings()

    frame_window: deque[torch.Tensor] = deque(maxlen=clip_length)
    sampled_index_window: deque[int] = deque(maxlen=clip_length)
    source_index_window: deque[int] = deque(maxlen=clip_length)

    while True:
        decode_started = perf_counter()
        ok, frame_bgr = capture.read()
        stage_timings.add("decode_time_sec", perf_counter() - decode_started)
        if not ok:
            break

        source_frame_index += 1
        current_time = source_frame_index / source_fps if source_fps > 0 else 0.0
        if target_interval is not None and current_time + 1e-9 < next_sample_time:
            continue
        if target_interval is not None:
            next_sample_time = current_time + target_interval

        preprocess_started = perf_counter()
        frame_tensor = preprocess_bgr_frame(
            frame_bgr=frame_bgr,
            image_size=int(model_bundle["image_size"]),
            mean=tuple(model_bundle["mean"]),
            std=tuple(model_bundle["std"]),
        )
        stage_timings.add("preprocess_time_sec", perf_counter() - preprocess_started)

        sampled_frame_index += 1
        sampled_frame_count += 1
        frame_window.append(frame_tensor)
        sampled_index_window.append(sampled_frame_index)
        source_index_window.append(source_frame_index)

        if len(frame_window) < clip_length:
            continue

        window_start_index = sampled_frame_index - clip_length + 1
        if window_start_index % stride != 0:
            continue

        inference_started = perf_counter()
        prediction = predict_clip_tensor(
            clip_frames=list(frame_window),
            model=model_bundle["model"],
            class_names=model_bundle["class_names"],
            device=model_bundle["device"],
            backend=str(model_bundle.get("backend", "eager")),
        )
        stage_timings.add("inference_time_sec", perf_counter() - inference_started)
        window_count += 1

        predicted_class = normalize_label_name(prediction["predicted_class"])
        if predicted_class == "negative":
            continue
        if prediction["confidence"] < threshold:
            continue

        center_offset = clip_length // 2
        center_frame_idx = sampled_index_window[center_offset]
        center_source_frame_idx = source_index_window[center_offset]
        detections.append(
            {
                "frame_idx": int(center_frame_idx),
                "source_frame_idx": int(center_source_frame_idx),
                "timestamp_ms": int(math.floor(center_frame_idx / sample_fps * 1000.0)),
                "confidence": float(prediction["confidence"]),
                "predicted_class": predicted_class,
                "predicted_index": int(prediction["predicted_index"]),
                "class_probs": prediction["class_probs"],
            }
        )

    capture.release()

    merged_detections = _merge_nearby_detections(
        detections=detections,
        threshold_frames=max(int(round(sample_fps)), 1),
    )
    return {
        "detections": merged_detections,
        "timings": stage_timings.as_dict(),
        "source_fps": source_fps,
        "sample_fps": sample_fps,
        "sampled_frame_count": sampled_frame_count,
        "window_count": window_count,
    }
