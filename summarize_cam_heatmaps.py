#!/usr/bin/env python3
"""Average CAM heatmaps over annotated pedestrian boxes."""

import argparse
import glob
import os

import cv2
import numpy as np
from scipy.io import loadmat

from pdestre_experiments.annotations import parse_result_basename, read_annotation_boxes
from pdestre_experiments.paths import annotation_dir, cam_output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize CAM heatmaps over P-DESTRE ground-truth boxes.'
    )
    parser.add_argument(
        '--cam-dir',
        default=cam_output_dir(),
        help='Directory containing *_grayscale_cam.mat files',
    )
    parser.add_argument(
        '--annotation-dir',
        default=annotation_dir(),
        help='Directory with P-DESTRE annotation .txt files',
    )
    parser.add_argument(
        '--output-image',
        default='mean_map_head_out.png',
        help='Path to the averaged heatmap image',
    )
    parser.add_argument(
        '--gt-scale',
        type=float,
        default=4.0,
        help='Scale factor applied to ground-truth boxes',
    )
    parser.add_argument(
        '--max-aspect-ratio',
        type=float,
        default=0.6,
        help='Keep boxes with width/height below this value',
    )
    parser.add_argument(
        '--resize-width',
        type=int,
        default=50,
        help='Width used when normalizing cropped heatmaps',
    )
    parser.add_argument(
        '--resize-height',
        type=int,
        default=100,
        help='Height used when normalizing cropped heatmaps',
    )
    return parser.parse_args()


def summarize_cam_heatmaps(
    cam_dir,
    annotation_dir_path,
    output_image,
    gt_scale=4.0,
    max_aspect_ratio=0.6,
    resize_size=(50, 100),
):
    crops = []
    cam_files = sorted(glob.glob(os.path.join(cam_dir, '*cam.mat')))

    for cam_file in cam_files:
        video_name, frame_idx = parse_result_basename(cam_file)
        cam_map = loadmat(cam_file)['grayscale_cam']
        annotation_path = os.path.join(annotation_dir_path, f'{video_name}.txt')

        for x, y, w, h in read_annotation_boxes(annotation_path, frame_idx, scale=gt_scale):
            x, y, w, h = (int(np.floor(value)) for value in (x, y, w, h))
            if w <= 0 or h <= 0 or (w / h) >= max_aspect_ratio:
                continue
            crop = cam_map[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            crops.append(cv2.resize(crop, resize_size))

    if not crops:
        raise RuntimeError(f'No CAM crops found in {cam_dir}')

    avg_map = np.mean(np.stack(crops, axis=0), axis=0)
    os.makedirs(os.path.dirname(output_image) or '.', exist_ok=True)
    cv2.imwrite(output_image, avg_map)


def main():
    args = parse_args()
    summarize_cam_heatmaps(
        args.cam_dir,
        args.annotation_dir,
        args.output_image,
        gt_scale=args.gt_scale,
        max_aspect_ratio=args.max_aspect_ratio,
        resize_size=(args.resize_width, args.resize_height),
    )
    print(f'Wrote summary heatmap to {args.output_image}')


if __name__ == '__main__':
    main()
