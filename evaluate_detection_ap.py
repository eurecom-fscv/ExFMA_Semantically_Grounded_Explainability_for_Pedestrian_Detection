#!/usr/bin/env python3
"""Evaluate detector average precision against P-DESTRE ground truth."""

import argparse
import glob
import os

import numpy as np
from scipy.io import loadmat

from pdestre_experiments.annotations import parse_result_basename, read_annotation_boxes
from pdestre_experiments.detection_metrics import evaluate_detection_precision, xyxy_to_xywh
from pdestre_experiments.paths import annotation_dir, detection_output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute detection AP from saved bbox .mat files.'
    )
    parser.add_argument(
        '--detection-dir',
        default=detection_output_dir(),
        help='Directory containing *_bbox.mat detector outputs',
    )
    parser.add_argument(
        '--annotation-dir',
        default=annotation_dir(),
        help='Directory with P-DESTRE annotation .txt files',
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.3,
        help='Minimum detection score kept for evaluation',
    )
    parser.add_argument(
        '--detection-scale',
        type=float,
        default=1.0,
        help='Scale factor applied to detector boxes',
    )
    parser.add_argument(
        '--gt-scale',
        type=float,
        default=4.0,
        help='Scale factor applied to ground-truth boxes',
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold used for true-positive matching',
    )
    return parser.parse_args()


def load_detection_results(
    detection_dir,
    annotation_dir_path,
    score_threshold,
    detection_scale,
    gt_scale,
):
    detections_xywh = []
    detection_scores = []
    ground_truth_xywh = []

    bbox_files = sorted(glob.glob(os.path.join(detection_dir, '*_bbox.mat')))
    for bbox_file in bbox_files:
        video_name, frame_idx = parse_result_basename(bbox_file)
        bboxes = loadmat(bbox_file)['bboxes']
        if bboxes.ndim == 1 or bboxes.shape[0] == 0:
            detections_xywh.append(np.empty((0, 4), dtype=np.float32))
            detection_scores.append(np.empty((0,), dtype=np.float32))
        else:
            kept = bboxes[bboxes[:, 4] > score_threshold]
            xywh = xyxy_to_xywh(kept[:, :4]) / detection_scale
            detections_xywh.append(xywh.astype(np.float32))
            detection_scores.append(kept[:, 4].astype(np.float32))

        annotation_path = os.path.join(annotation_dir_path, f'{video_name}.txt')
        gt_boxes = read_annotation_boxes(annotation_path, frame_idx, scale=gt_scale)
        ground_truth_xywh.append(np.asarray(gt_boxes, dtype=np.float32))

    return detections_xywh, detection_scores, ground_truth_xywh


def main():
    args = parse_args()
    detections_xywh, detection_scores, ground_truth_xywh = load_detection_results(
        args.detection_dir,
        args.annotation_dir,
        args.score_threshold,
        args.detection_scale,
        args.gt_scale,
    )
    ap, recall, precision = evaluate_detection_precision(
        detections_xywh,
        detection_scores,
        ground_truth_xywh,
        iou_threshold=args.iou_threshold,
    )
    mean_recall = float(np.mean(recall)) if recall.size else 0.0
    mean_precision = float(np.mean(precision)) if precision.size else 0.0
    print(
        f'Average precision = {ap * 100:.2f}; '
        f'Recall = {mean_recall * 100:.2f}; '
        f'Precision = {mean_precision * 100:.2f}.'
    )


if __name__ == '__main__':
    main()
