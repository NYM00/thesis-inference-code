from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from .modeling import normalize_label_name


def load_ground_truth_transitions(db_path: str | Path) -> list[dict[str, Any]]:
    database_path = Path(db_path)
    if not database_path.is_file():
        raise FileNotFoundError(f"Transition database not found: {database_path}")

    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.cursor()
        rows = cursor.execute(
            "SELECT category, start_frame, end_frame, probability FROM transitions ORDER BY start_frame"
        ).fetchall()
    finally:
        connection.close()

    transitions = []
    for category, start_frame, end_frame, probability in rows:
        transitions.append(
            {
                "category": normalize_label_name(str(category)),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "probability": float(probability) if probability is not None else None,
            }
        )
    return transitions


def evaluate_transitions(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    tolerance_frames: int = 0,
) -> dict[str, Any]:
    gt_by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pred_by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for item in ground_truth:
        scope = (str(item.get("video_id", "")), item["category"])
        gt_by_scope[scope].append(item)
    for item in predictions:
        scope = (str(item.get("video_id", "")), normalize_label_name(item["predicted_class"]) or "unknown")
        pred_by_scope[scope].append(item)

    scopes = sorted(set(gt_by_scope.keys()) | set(pred_by_scope.keys()))
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_type: dict[str, Any] = {}

    for video_id, category in scopes:
        gt_items = gt_by_scope.get((video_id, category), [])
        pred_items = sorted(pred_by_scope.get((video_id, category), []), key=lambda item: item["confidence"], reverse=True)
        matched_indices: set[int] = set()
        tp = 0
        fp = 0

        for prediction in pred_items:
            best_match = None
            best_distance = None
            frame_idx = int(prediction["frame_idx"])
            for gt_index, gt_item in enumerate(gt_items):
                if gt_index in matched_indices:
                    continue
                lower = gt_item["start_frame"] - tolerance_frames
                upper = gt_item["end_frame"] + tolerance_frames
                if lower <= frame_idx <= upper:
                    center = (gt_item["start_frame"] + gt_item["end_frame"]) / 2.0
                    distance = abs(frame_idx - center)
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_match = gt_index

            if best_match is not None:
                matched_indices.add(best_match)
                tp += 1
            else:
                fp += 1

        fn = max(len(gt_items) - tp, 0)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        category_bucket = per_type.setdefault(
            category,
            {
                "support": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
            },
        )
        category_bucket["support"] += len(gt_items)
        category_bucket["tp"] += tp
        category_bucket["fp"] += fp
        category_bucket["fn"] += fn

    for category, bucket in per_type.items():
        tp = bucket["tp"]
        fp = bucket["fp"]
        fn = bucket["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
        bucket["precision"] = precision
        bucket["recall"] = recall
        bucket["f1"] = f1

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tolerance_frames": tolerance_frames,
        "per_type": per_type,
        "ground_truth_count": len(ground_truth),
    }
