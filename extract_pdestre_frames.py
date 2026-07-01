#!/usr/bin/env python3
"""Extract annotated P-DESTRE frames from source videos."""

import argparse
import os

import cv2

from pdestre_experiments.annotations import (
    list_annotation_files,
    read_frame_indices,
    select_annotation_subset,
)
from pdestre_experiments.paths import annotation_dir, frame_output_dir, video_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract PNG frames from P-DESTRE videos for annotated frames.'
    )
    parser.add_argument(
        '--annotation-dir',
        default=annotation_dir(),
        help='Directory with P-DESTRE annotation .txt files',
    )
    parser.add_argument(
        '--video-dir',
        default=video_dir(),
        help='Directory with P-DESTRE source videos',
    )
    parser.add_argument(
        '--output-dir',
        default=frame_output_dir(),
        help='Directory where extracted PNG frames are written',
    )
    parser.add_argument(
        '--start-fraction',
        type=float,
        default=0.5,
        help='Process annotations starting from this fraction of the file list',
    )
    parser.add_argument(
        '--video-extension',
        default='.MP4',
        help='Video file extension used when resolving source videos',
    )
    return parser.parse_args()


def extract_frames(
    annotation_files,
    video_dir_path,
    output_dir,
    video_extension='.MP4',
):
    os.makedirs(output_dir, exist_ok=True)

    for annotation_path in annotation_files:
        video_name = os.path.splitext(os.path.basename(annotation_path))[0]
        video_path = os.path.join(video_dir_path, video_name + video_extension)
        if not os.path.exists(video_path):
            video_path = os.path.join(video_dir_path, video_name + video_extension.lower())
        if not os.path.exists(video_path):
            raise FileNotFoundError(f'Video not found for annotation: {video_path}')

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f'Unable to open video: {video_path}')

        for frame_idx in read_frame_indices(annotation_path):
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - 1, 0))
            ok, frame = capture.read()
            if not ok:
                continue
            output_path = os.path.join(
                output_dir,
                f'{video_name}_frame_{frame_idx}.png',
            )
            cv2.imwrite(output_path, frame)

        capture.release()


def main():
    args = parse_args()
    annotation_files = select_annotation_subset(
        list_annotation_files(args.annotation_dir),
        start_fraction=args.start_fraction,
    )
    extract_frames(
        annotation_files,
        args.video_dir,
        args.output_dir,
        video_extension=args.video_extension,
    )


if __name__ == '__main__':
    main()
