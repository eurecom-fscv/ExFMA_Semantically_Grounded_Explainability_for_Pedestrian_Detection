# Explainability Experiments (P-DESTRE)

This repository contains the code used to run pedestrian-detector explainability
experiments on the P-DESTRE dataset.

## Pipeline

| Step | Python | MATLAB |
|------|--------|--------|
| 1. Extract frames | `extract_pdestre_frames.py` | `extract_pdestre_frames.m` |
| 2. Generate CAMs | `generate_ablation_cam.py` | — |
| 3. Generate CAMs with occlusions | `generate_ablation_cam_occluded.py` | — |
| 4. Evaluate detection AP | `evaluate_detection_ap.py` | `evaluate_detection_ap.m` |
| 5. Summarize heatmaps | `summarize_cam_heatmaps.py` | `summarize_cam_heatmaps.m` |

Shared helpers live in `pdestre_experiments/`.

## Data layout

Place the P-DESTRE dataset locally (not included in this repo):

```
data/P_Destre/
  annotation/   # one .txt file per video
  videos/       # source .MP4 files
  frames/       # extracted PNGs (created by extract_pdestre_frames)
```

Experiment outputs go under `outputs/` (gitignored).

## Environment variables

| Variable | Used by | Default |
|----------|---------|---------|
| `PDESTRE_ANNOTATION_DIR` | all scripts | `data/P_Destre/annotation` |
| `PDESTRE_VIDEO_DIR` | frame extraction | `data/P_Destre/videos` |
| `PDESTRE_FRAME_OUTPUT_DIR` | frame extraction | `data/P_Destre/frames` |
| `PDESTRE_DETECTION_OUTPUT_DIR` | AP evaluation | `outputs/visible_PDestre_bottom_exp_out` |
| `PDESTRE_CAM_OUTPUT_DIR` | heatmap summary | `outputs/visible_PDestre_head_exp_out` |

## Python dependencies

Install Pedestron first (see `INSTALL.md`), then install a fork of
`pytorch-grad-cam` that provides the Pedestron-specific classes:

- `AblationLayerPedestron`
- `pedestron_ablation_reshape_transform`
- `FasterRCNNBoxScoreTarget`

## Example commands

```bash
# 1. Extract frames
python extract_pdestre_frames.py

# 2. Generate CAMs with head occlusion
python generate_ablation_cam_occluded.py \
  configs/elephant/p_destre/cascade_hrnet.py \
  /path/to/checkpoint.pth \
  data/P_Destre/frames \
  outputs/visible_PDestre_head_exp_out \
  --gt-annotation-dir data/P_Destre/annotation \
  --occlusion-region head

# 3. Evaluate detection AP
python evaluate_detection_ap.py \
  --detection-dir outputs/visible_PDestre_bottom_exp_out

# 4. Summarize CAM heatmaps
python summarize_cam_heatmaps.py \
  --cam-dir outputs/visible_PDestre_head_exp_out \
  --output-image mean_map_head_out.png
```

MATLAB equivalents:

```matlab
extract_pdestre_frames
evaluate_detection_ap
summarize_cam_heatmaps
```

## MATLAB requirements

`evaluate_detection_ap.m` calls `evaluateDetectionPrecision` from the MATLAB
Computer Vision Toolbox. The Python version implements an equivalent metric in
`pdestre_experiments/detection_metrics.py`.
