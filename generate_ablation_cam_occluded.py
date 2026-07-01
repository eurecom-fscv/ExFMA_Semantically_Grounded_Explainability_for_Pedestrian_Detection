#!/usr/bin/env python3
"""Generate AblationCAM heatmaps with body-part occlusions."""

import argparse
import os

from pdestre_experiments.ablation_cam import run_ablation_cam_dataset
from pdestre_experiments.paths import annotation_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate AblationCAM maps after applying body-part occlusions.'
    )
    parser.add_argument('config', help='MMDetection config file')
    parser.add_argument('checkpoint', help='Detector checkpoint')
    parser.add_argument('input_img_dir', help='Directory with input PNG frames')
    parser.add_argument('output_dir', help='Directory for CAM and bbox outputs')
    parser.add_argument(
        '--gt-annotation-dir',
        default=annotation_dir(),
        help='Directory with P-DESTRE annotation .txt files',
    )
    parser.add_argument(
        '--occlusion-region',
        choices=['head', 'top', 'bottom', 'legs'],
        default='head',
        help='Body region to occlude before running the detector',
    )
    parser.add_argument(
        '--gt-scale',
        type=float,
        default=4.0,
        help='Downscale factor applied to GT boxes before occlusion',
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.3,
        help='Minimum detection score used for CAM targets',
    )
    return parser.parse_args()


def main():
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = parse_args()
    run_ablation_cam_dataset(
        args.config,
        args.checkpoint,
        args.input_img_dir,
        args.output_dir,
        annotation_dir=args.gt_annotation_dir,
        occlusion_region=args.occlusion_region,
        gt_scale=args.gt_scale,
        score_threshold=args.score_threshold,
    )


if __name__ == '__main__':
    main()
