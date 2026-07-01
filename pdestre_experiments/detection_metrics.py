from typing import List, Sequence, Tuple

import numpy as np


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.copy()
    converted[:, 2] -= converted[:, 0]
    converted[:, 3] -= converted[:, 1]
    return converted


def box_iou_xywh(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_detection_precision(
    detections_xywh: Sequence[np.ndarray],
    detection_scores: Sequence[np.ndarray],
    ground_truth_xywh: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Approximate MATLAB ``evaluateDetectionPrecision`` for xywh boxes."""
    entries = []
    num_gt = 0

    for image_idx, gt_boxes in enumerate(ground_truth_xywh):
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
        num_gt += len(gt_boxes)

        det_boxes = np.asarray(detections_xywh[image_idx], dtype=np.float32)
        det_scores = np.asarray(detection_scores[image_idx], dtype=np.float32)
        if det_boxes.size == 0:
            continue

        order = np.argsort(-det_scores)
        matched_gt = set()

        for det_idx in order:
            best_iou = 0.0
            best_gt_idx = None
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = box_iou_xywh(det_boxes[det_idx], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            is_tp = int(best_iou >= iou_threshold and best_gt_idx is not None)
            if is_tp:
                matched_gt.add(best_gt_idx)
            entries.append((det_scores[det_idx], is_tp))

    if num_gt == 0:
        return 0.0, np.array([]), np.array([])

    if not entries:
        return 0.0, np.array([0.0]), np.array([0.0])

    entries.sort(key=lambda item: item[0], reverse=True)
    tp = np.array([item[1] for item in entries], dtype=np.float32)
    fp = 1.0 - tp
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / num_gt
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float32).eps)

    recall_points = np.concatenate(([0.0], recall, [1.0]))
    precision_points = np.concatenate(([1.0], precision, [0.0]))
    for idx in range(len(precision_points) - 2, -1, -1):
        precision_points[idx] = max(precision_points[idx], precision_points[idx + 1])

    ap = float(np.sum(
        (recall_points[1:] - recall_points[:-1]) * precision_points[1:]
    ))
    return ap, recall, precision
