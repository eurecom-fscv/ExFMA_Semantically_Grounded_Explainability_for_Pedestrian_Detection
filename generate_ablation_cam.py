#!/usr/bin/env python3
"""Generate AblationCAM heatmaps and detector outputs for P-DESTRE frames."""

import argparse
import os

from pdestre_experiments.ablation_cam import run_ablation_cam_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate AblationCAM maps and detector bounding boxes.'
    )
    parser.add_argument('config', help='MMDetection config file')
    parser.add_argument('checkpoint', help='Detector checkpoint')
    parser.add_argument('input_img_dir', help='Directory with input PNG frames')
    parser.add_argument('output_dir', help='Directory for CAM and bbox outputs')
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
        score_threshold=args.score_threshold,
    )


if __name__ == '__main__':
    main()
