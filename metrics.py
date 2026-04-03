"""Lightweight detection metrics for custom YOLO training loops.

This module provides practical implementations of common detection metrics:
- Precision / Recall / F1
- AP@IoU (e.g. AP50)
- mAP@0.5 and mAP@0.5:0.95
- Simple speed (latency/FPS) tracking

Prediction format per image:
    ndarray of shape (N, 6): [x1, y1, x2, y2, conf, cls]

Target format per image supports either:
    ndarray of shape (M, 5): [cls, x1, y1, x2, y2]
or  ndarray of shape (M, 5): [x1, y1, x2, y2, cls]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


EPS = 1e-9


def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes in xyxy format."""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1 = boxes1.astype(np.float32)
    b2 = boxes2.astype(np.float32)

    x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])

    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area1 = np.clip((b1[:, 2] - b1[:, 0]), 0.0, None) * np.clip((b1[:, 3] - b1[:, 1]), 0.0, None)
    area2 = np.clip((b2[:, 2] - b2[:, 0]), 0.0, None) * np.clip((b2[:, 3] - b2[:, 1]), 0.0, None)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, EPS, None)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return precision, recall and F1 from counts."""
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    return float(precision), float(recall), float(f1)


def _normalize_targets(targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize targets to (boxes_xyxy, cls).

    Supports [cls, x1, y1, x2, y2] and [x1, y1, x2, y2, cls].
    """
    if targets.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    t = np.asarray(targets)
    if t.shape[1] != 5:
        raise ValueError(f"targets must have shape (M, 5), got {t.shape}")

    first_col_int_like = np.allclose(t[:, 0], np.round(t[:, 0]))
    last_col_int_like = np.allclose(t[:, -1], np.round(t[:, -1]))

    if first_col_int_like and not last_col_int_like:
        cls = t[:, 0].astype(np.int64)
        boxes = t[:, 1:5].astype(np.float32)
        return boxes, cls

    if last_col_int_like and not first_col_int_like:
        cls = t[:, 4].astype(np.int64)
        boxes = t[:, 0:4].astype(np.float32)
        return boxes, cls

    cls = t[:, 0].astype(np.int64)
    boxes = t[:, 1:5].astype(np.float32)
    return boxes, cls


def _compute_ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP as area under precision-recall curve (VOC-style envelope integration)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def ap_per_class(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
    num_classes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class AP, Precision, Recall and F1 at one IoU threshold.

    Returns arrays shaped (num_classes,).
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")

    if num_classes is None:
        max_cls = -1
        for p in predictions:
            if p.size:
                max_cls = max(max_cls, int(np.max(p[:, 5])))
        for t in targets:
            if t.size:
                t_boxes, t_cls = _normalize_targets(t)
                del t_boxes
                max_cls = max(max_cls, int(np.max(t_cls)))
        num_classes = max_cls + 1 if max_cls >= 0 else 0

    aps = np.zeros((num_classes,), dtype=np.float32)
    precisions = np.zeros((num_classes,), dtype=np.float32)
    recalls = np.zeros((num_classes,), dtype=np.float32)
    f1s = np.zeros((num_classes,), dtype=np.float32)

    for c in range(num_classes):
        cls_pred_conf: List[float] = []
        cls_pred_tp: List[int] = []
        n_gt = 0

        for pred_img, tgt_img in zip(predictions, targets):
            pred_img = np.asarray(pred_img)
            tgt_img = np.asarray(tgt_img)

            tgt_boxes, tgt_cls = _normalize_targets(tgt_img)
            gt_mask = tgt_cls == c
            gt_boxes_c = tgt_boxes[gt_mask]
            n_gt += len(gt_boxes_c)

            if pred_img.size == 0:
                continue

            pred_cls = pred_img[:, 5].astype(np.int64)
            pred_mask = pred_cls == c
            pred_c = pred_img[pred_mask]
            if pred_c.size == 0:
                continue

            order = np.argsort(-pred_c[:, 4])
            pred_c = pred_c[order]

            if len(gt_boxes_c) == 0:
                cls_pred_conf.extend(pred_c[:, 4].tolist())
                cls_pred_tp.extend([0] * len(pred_c))
                continue

            ious = box_iou_matrix(pred_c[:, :4], gt_boxes_c)
            matched_gt = np.zeros((len(gt_boxes_c),), dtype=bool)

            for i, p in enumerate(pred_c):
                cls_pred_conf.append(float(p[4]))
                best_gt = int(np.argmax(ious[i]))
                best_iou = float(ious[i, best_gt])

                if best_iou >= iou_threshold and not matched_gt[best_gt]:
                    cls_pred_tp.append(1)
                    matched_gt[best_gt] = True
                else:
                    cls_pred_tp.append(0)

        if n_gt == 0:
            continue

        if len(cls_pred_conf) == 0:
            continue

        conf = np.asarray(cls_pred_conf)
        tp = np.asarray(cls_pred_tp, dtype=np.int32)
        order = np.argsort(-conf)
        tp = tp[order]
        fp = 1 - tp

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall_curve = tp_cum / (n_gt + EPS)
        precision_curve = tp_cum / (tp_cum + fp_cum + EPS)

        ap = _compute_ap_from_pr(recall_curve, precision_curve)

        tp_final = int(tp_cum[-1])
        fp_final = int(fp_cum[-1])
        fn_final = int(n_gt - tp_final)
        p, r, f1 = precision_recall_f1(tp_final, fp_final, fn_final)

        aps[c] = ap
        precisions[c] = p
        recalls[c] = r
        f1s[c] = f1

    return aps, precisions, recalls, f1s


def mean_ap(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    num_classes: int | None = None,
    iou_thresholds: Iterable[float] = np.arange(0.5, 0.96, 0.05),
) -> Dict[str, float]:
    """Compute mAP metrics across one or multiple IoU thresholds."""
    iou_thresholds = list(iou_thresholds)
    if not iou_thresholds:
        raise ValueError("iou_thresholds must not be empty")

    ap_all = []
    p50 = r50 = f150 = None

    for iou in iou_thresholds:
        aps, ps, rs, f1s = ap_per_class(predictions, targets, iou_threshold=iou, num_classes=num_classes)
        ap_all.append(aps)
        if abs(iou - 0.5) < 1e-9:
            p50 = float(np.mean(ps)) if len(ps) else 0.0
            r50 = float(np.mean(rs)) if len(rs) else 0.0
            f150 = float(np.mean(f1s)) if len(f1s) else 0.0

    ap_all_np = np.stack(ap_all, axis=0) if len(ap_all) else np.zeros((0, 0), dtype=np.float32)

    if len(ap_all_np) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
        }

    map50 = float(np.mean(ap_all_np[iou_thresholds.index(0.5)])) if 0.5 in iou_thresholds else 0.0
    map50_95 = float(np.mean(ap_all_np))

    return {
        "precision": p50 if p50 is not None else 0.0,
        "recall": r50 if r50 is not None else 0.0,
        "f1": f150 if f150 is not None else 0.0,
        "map50": map50,
        "map50_95": map50_95,
    }


@dataclass
class SpeedMeter:
    """Track average latency (ms) and throughput (FPS)."""

    total_seconds: float = 0.0
    total_images: int = 0

    def update(self, seconds: float, num_images: int = 1) -> None:
        self.total_seconds += float(seconds)
        self.total_images += int(num_images)

    def latency_ms(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.total_seconds * 1000.0 / self.total_images

    def fps(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return self.total_images / self.total_seconds


class Timer:
    """Context manager helper for timing a block and updating SpeedMeter."""

    def __init__(self, meter: SpeedMeter, num_images: int = 1):
        self.meter = meter
        self.num_images = num_images
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._start
        self.meter.update(elapsed, self.num_images)
        return False
