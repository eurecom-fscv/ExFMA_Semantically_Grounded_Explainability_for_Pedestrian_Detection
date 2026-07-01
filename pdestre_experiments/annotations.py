import os
from typing import Iterable, List, Tuple


def parse_result_basename(filename: str) -> Tuple[str, str]:
    """Parse ``{video}_frame_{idx}_suffix`` style result filenames."""
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    if len(parts) < 3:
        raise ValueError(f'Unexpected result filename format: {filename}')
    return parts[0], parts[2]


def read_annotation_boxes(
    annotation_path: str,
    frame_idx: str,
    scale: float = 1.0,
) -> List[Tuple[float, float, float, float]]:
    boxes = []
    with open(annotation_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fields = line.split(',')
            if fields[0] != str(frame_idx):
                continue
            x, y, w, h = (max(float(fields[i]), 1.0) for i in range(2, 6))
            boxes.append((x / scale, y / scale, w / scale, h / scale))
    return boxes


def read_frame_indices(annotation_path: str) -> List[int]:
    indices = set()
    with open(annotation_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fields = line.split(',')
            indices.add(int(float(fields[0])))
    return sorted(indices)


def list_annotation_files(annotation_dir: str) -> List[str]:
    return sorted(
        os.path.join(annotation_dir, name)
        for name in os.listdir(annotation_dir)
        if name.endswith('.txt')
    )


def select_annotation_subset(
    annotation_files: Iterable[str],
    start_fraction: float = 0.5,
) -> List[str]:
    files = list(annotation_files)
    start = int(len(files) * start_fraction)
    return files[start:]
