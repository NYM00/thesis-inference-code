from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def xywhn_to_xyxy(box: list[float], image_width: int, image_height: int) -> list[float]:
    _, x_center, y_center, width, height = box
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0
    return [x1, y1, x2, y2]


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def load_ground_truth(
    image_shapes: dict[str, tuple[int, int]],
    labels_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    labels_root = Path(labels_path)
    if not labels_root.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")

    ground_truth: dict[str, list[dict[str, Any]]] = {}
    for image_name, (image_height, image_width) in image_shapes.items():
        label_file = labels_root / f"{Path(image_name).stem}.txt"
        items: list[dict[str, Any]] = []
        if label_file.is_file():
            with label_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    if len(parts) != 5:
                        raise ValueError(f"Expected 5 columns in YOLO label file: {label_file}")
                    raw = [float(part) for part in parts]
                    class_id = int(raw[0])
                    items.append(
                        {
                            "class_id": class_id,
                            "xyxy": xywhn_to_xyxy(raw, image_width=image_width, image_height=image_height),
                        }
                    )
        ground_truth[image_name] = items
    return ground_truth


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recall_points = np.linspace(0.0, 1.0, 101)
    envelope = []
    for point in recall_points:
        candidates = precisions[recalls >= point]
        envelope.append(float(np.max(candidates)) if candidates.size else 0.0)
    return float(np.mean(envelope))


def _evaluate_single_class(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, list[dict[str, Any]]],
    iou_threshold: float,
) -> tuple[float, int, int, int]:
    gt_by_image: dict[str, list[list[float]]] = {}
    for image_name, items in ground_truth.items():
        gt_by_image[image_name] = [item["xyxy"] for item in items]

    positives = sum(len(items) for items in gt_by_image.values())
    if positives == 0:
        return 0.0, 0, 0, 0

    matches: dict[str, set[int]] = defaultdict(set)
    sorted_predictions = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
    true_positive_flags = []
    false_positive_flags = []

    for prediction in sorted_predictions:
        image_name = prediction["image_name"]
        candidate_boxes = gt_by_image.get(image_name, [])
        best_index = -1
        best_iou = 0.0
        for gt_index, gt_box in enumerate(candidate_boxes):
            if gt_index in matches[image_name]:
                continue
            iou = box_iou(prediction["xyxy"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index

        if best_index >= 0 and best_iou >= iou_threshold:
            matches[image_name].add(best_index)
            true_positive_flags.append(1)
            false_positive_flags.append(0)
        else:
            true_positive_flags.append(0)
            false_positive_flags.append(1)

    if not true_positive_flags:
        return 0.0, 0, len(sorted_predictions), positives

    tp_cumsum = np.cumsum(true_positive_flags)
    fp_cumsum = np.cumsum(false_positive_flags)
    recalls = tp_cumsum / max(float(positives), 1.0)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
    ap = _compute_ap(recalls.astype(float), precisions.astype(float))
    tp_total = int(tp_cumsum[-1])
    fp_total = int(fp_cumsum[-1])
    fn_total = max(positives - tp_total, 0)
    return ap, tp_total, fp_total, fn_total


def evaluate_detection_predictions(
    predictions_by_image: dict[str, list[dict[str, Any]]],
    ground_truth_by_image: dict[str, list[dict[str, Any]]],
    class_names: dict[int, str],
) -> dict[str, Any]:
    predictions_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ground_truth_by_class: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for image_name, predictions in predictions_by_image.items():
        for prediction in predictions:
            predictions_by_class[int(prediction["class_id"])].append(
                {
                    "image_name": image_name,
                    "confidence": float(prediction["confidence"]),
                    "xyxy": prediction["xyxy"],
                }
            )

    for image_name, annotations in ground_truth_by_image.items():
        for annotation in annotations:
            class_id = int(annotation["class_id"])
            ground_truth_by_class[class_id][image_name].append(annotation)

    iou_thresholds = [round(0.5 + 0.05 * index, 2) for index in range(10)]
    map_50_values: list[float] = []
    map_50_95_values: list[float] = []
    per_class: dict[str, Any] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    candidate_class_ids = sorted(set(ground_truth_by_class.keys()) | set(predictions_by_class.keys()))
    for class_id in candidate_class_ids:
        class_name = class_names.get(class_id, str(class_id))
        gt_items = ground_truth_by_class.get(class_id, {})
        pred_items = predictions_by_class.get(class_id, [])
        ap_values = []
        tp_50 = fp_50 = fn_50 = 0

        for iou_threshold in iou_thresholds:
            ap, tp_count, fp_count, fn_count = _evaluate_single_class(pred_items, gt_items, iou_threshold=iou_threshold)
            ap_values.append(ap)
            if abs(iou_threshold - 0.5) < 1e-9:
                tp_50, fp_50, fn_50 = tp_count, fp_count, fn_count

        gt_support = sum(len(items) for items in gt_items.values())
        if gt_support > 0:
            map_50_values.append(ap_values[0])
            map_50_95_values.append(float(np.mean(ap_values)))
            total_tp += tp_50
            total_fp += fp_50
            total_fn += fn_50

        per_class[class_name] = {
            "class_id": class_id,
            "support": gt_support,
            "ap50": ap_values[0],
            "ap50_95": float(np.mean(ap_values)),
            "tp": tp_50,
            "fp": fp_50,
            "fn": fn_50,
        }

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 0.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(np.mean(map_50_values)) if map_50_values else None,
        "map50_95": float(np.mean(map_50_95_values)) if map_50_95_values else None,
        "per_class": per_class,
        "supports_labels": bool(map_50_values),
        "label_count": sum(len(items) for items in ground_truth_by_image.values()),
    }
