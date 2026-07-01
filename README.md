# Pedestron Explainability Experiments

Public release of the explainability experiment code for pedestrian detection on
the P-DESTRE dataset. This repository is a focused subset built on two existing
codebases:

- [Pedestron](https://github.com/hasanirtiza/Pedestron) — pedestrian detection
  framework (SemanticPedestrianDetection)
- [MMDetection](https://github.com/open-mmlab/mmdetection) — object detection
  toolbox (vendored under `mmdet/`)

It contains only the scripts and dependencies needed to reproduce the paper
experiments. Supplementary material for the grid visualizations (FLAS) is
available at
[https://galdi.eurecom.io/ExFMA2026_supplementary_material.html](https://galdi.eurecom.io/ExFMA2026_supplementary_material.html).

## Contents

| Script | Description |
|--------|-------------|
| `extract_pdestre_frames.py` / `.m` | Extract annotated frames from P-DESTRE videos |
| `generate_ablation_cam.py` | Detector + AblationCAM baseline |
| `generate_ablation_cam_occluded.py` | AblationCAM with body-part occlusions |
| `evaluate_detection_ap.py` / `.m` | Detection average-precision evaluation |
| `summarize_cam_heatmaps.py` / `.m` | CAM heatmap summarization |
| `pdestre_experiments/` | Shared Python helpers |
| `mmdet/` | MMDetection-based detector framework |
| `configs/elephant/p_destre/` | P-DESTRE model configuration |

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Create a Python environment and install Pedestron

```bash
conda create -n pedestron-exp python=3.7 -y
conda activate pedestron-exp

conda install cython
# Install PyTorch + torchvision for your CUDA version: https://pytorch.org/

pip install -v -e .
```

See [INSTALL.md](INSTALL.md) for full system requirements and troubleshooting.

### 3. Install explainability dependencies

```bash
pip install opencv-python scipy pillow
# Install a Pedestron-compatible fork of pytorch-grad-cam that provides:
#   AblationLayerPedestron, pedestron_ablation_reshape_transform,
#   FasterRCNNBoxScoreTarget
```

### 4. Prepare the P-DESTRE dataset

Obtain P-DESTRE separately and place it under:

```
data/P_Destre/
  annotation/   # one .txt file per video
  videos/       # source .MP4 files
  frames/       # created by extract_pdestre_frames (step 1 below)
```

Download or train a detector checkpoint compatible with
`configs/elephant/p_destre/cascade_hrnet.py`.

## How to run

The experiment pipeline has five steps. Python commands are shown below;
MATLAB equivalents are listed in [EXPERIMENTS.md](EXPERIMENTS.md).

### Step 1 — Extract frames

Extract PNG frames for all annotated frames in the second half of the dataset:

```bash
python extract_pdestre_frames.py \
  --annotation-dir data/P_Destre/annotation \
  --video-dir data/P_Destre/videos \
  --output-dir data/P_Destre/frames
```

### Step 2 — Generate baseline CAMs (no occlusion)

```bash
python generate_ablation_cam.py \
  configs/elephant/p_destre/cascade_hrnet.py \
  /path/to/checkpoint.pth \
  data/P_Destre/frames \
  outputs/visible_PDestre_out
```

This writes per-frame `*_grayscale_cam.mat` and `*_bbox.mat` files.

### Step 3 — Generate CAMs with body-part occlusions

Repeat for each body region of interest (`head`, `top`, `bottom`, `legs`):

```bash
python generate_ablation_cam_occluded.py \
  configs/elephant/p_destre/cascade_hrnet.py \
  /path/to/checkpoint.pth \
  data/P_Destre/frames \
  outputs/visible_PDestre_head_exp_out \
  --gt-annotation-dir data/P_Destre/annotation \
  --occlusion-region head
```

Use matching output directories for other regions, e.g.
`outputs/visible_PDestre_top_exp_out`, `outputs/visible_PDestre_bottom_exp_out`,
`outputs/visible_PDestre_legs_exp_out`.

### Step 4 — Evaluate detection average precision

```bash
python evaluate_detection_ap.py \
  --detection-dir outputs/visible_PDestre_bottom_exp_out \
  --annotation-dir data/P_Destre/annotation
```

### Step 5 — Summarize average heatmaps

Average CAM activations over ground-truth pedestrian boxes and save a summary PNG:

```bash
python summarize_cam_heatmaps.py \
  --cam-dir outputs/visible_PDestre_head_exp_out \
  --annotation-dir data/P_Destre/annotation \
  --output-image outputs/mean_map_head_out.png
```

### Optional — MATLAB

If you prefer MATLAB for steps 1, 4, and 5, set the environment variables
described in [EXPERIMENTS.md](EXPERIMENTS.md) and run:

```matlab
extract_pdestre_frames
evaluate_detection_ap
summarize_cam_heatmaps
```

## Results

### Detection performance under anatomical occlusions

Pedestrian detection performance under different anatomically grounded occlusion
settings. The table reports Average Precision (AP), Recall, Precision, and
F1-score, enabling analysis of the contribution of different body regions to
detector performance.

| Method | AP | Recall | Precision | F1 Score |
|--------|-----|--------|-----------|----------|
| Full body — orig. size | 85.68 | 57.94 | 88.79 | 0.701 |
| Full body — resized ¼ | 86.51 | 57.38 | 90.28 | 0.701 |
| Lower body occluded | 68.37 | 49.74 | 80.52 | 0.615 |
| Upper body occluded | 35.25 | 30.22 | 59.96 | 0.401 |
| Head occluded | 81.98 | 55.09 | 87.45 | 0.674 |
| Legs occluded | 79.83 | 54.53 | 86.26 | 0.668 |

Upper-body occlusion causes the largest drop in AP (35.25), while head and legs
occlusions preserve relatively higher performance. Resizing the input to one
quarter of the original resolution has a minor effect compared with anatomical
masking.

### Anatomical partitioning

Body regions used for occlusion are defined from golden-ratio proportions on the
pedestrian bounding box: head (top ⅛), upper body (top ½), lower body (bottom ½),
and legs (bottom ¼).

<p align="center">
  <img src="img_paper/GoldenRatio.png" alt="Golden-ratio anatomical partitioning" width="420"/>
</p>

<p align="center"><em>Anatomical regions derived from golden-ratio proportions on the pedestrian bounding box.</em></p>

### Experimental pipeline

End-to-end workflow from input images and ground-truth boxes, through
anatomically grounded occlusions and Cascade Mask R-CNN detection, to AblationCAM
explanation maps and grid-based visualization (FLAS).

<p align="center">
  <img src="img_paper/CBMI_ExFMA.png" alt="Explainability pipeline overview" width="720"/>
</p>

<p align="center"><em>Overview of the explainability pipeline: input, occlusion variants, detector, AblationCAM maps, and grid visualization.</em></p>

Interactive grid visualizations (full-body and body-part views, including
CAM-thresholded variants) are provided as
[supplementary material](https://galdi.eurecom.io/ExFMA2026_supplementary_material.html).

### Average explanation heatmaps

The figures below are **average AblationCAM heatmaps** (jet colormap) produced by
`summarize_cam_heatmaps`. For each experiment, CAM maps are cropped to
ground-truth pedestrian bounding boxes (aspect ratio &lt; 0.6), resized to a
common 50×100 grid, and averaged over all qualifying detections. Warmer colors
(red/yellow) indicate higher average model importance; cooler colors (blue)
indicate lower importance.

#### Baseline (full body, no occlusion)

<p align="center">
  <img src="img_paper/avg_full_jet.png" alt="Baseline average heatmap" width="120"/>
</p>

<p align="center"><em>avg_full_jet.png — average AblationCAM with no body-part occlusion.</em></p>

#### Occlusion experiments

Each panel shows how average attention shifts when a specific body region is
replaced with its mean color before running the detector and CAM.

<table align="center">
  <tr>
    <td align="center">
      <img src="img_paper/avg_head_jet.png" alt="Head occlusion heatmap" width="120"/><br/>
      <b>Head</b><br/>
      <sub>Top ⅛ occluded</sub>
    </td>
    <td align="center">
      <img src="img_paper/avg_upper_jet.png" alt="Upper body occlusion heatmap" width="120"/><br/>
      <b>Upper body</b><br/>
      <sub>Top half occluded</sub>
    </td>
    <td align="center">
      <img src="img_paper/avg_lower_jet.png" alt="Lower body occlusion heatmap" width="120"/><br/>
      <b>Lower body</b><br/>
      <sub>Bottom half occluded</sub>
    </td>
    <td align="center">
      <img src="img_paper/avg_legs_jet.png" alt="Legs occlusion heatmap" width="120"/><br/>
      <b>Legs</b><br/>
      <sub>Bottom quarter occluded</sub>
    </td>
  </tr>
</table>

<p align="center">
  <em>
    Left to right: head, upper body, lower body, and legs occlusion experiments.
    Under upper-body occlusion, attention shifts toward the lower body; under
    lower-body occlusion, importance concentrates on the head and torso.
  </em>
</p>

To reproduce these figures, run step 3 with the corresponding
`--occlusion-region`, then step 5 pointing `--cam-dir` at the matching output
folder. Apply a jet colormap when preparing figures for publication.

## Further reading

- [EXPERIMENTS.md](EXPERIMENTS.md) — pipeline details, environment variables, legacy script names
- [INSTALL.md](INSTALL.md) — full Pedestron installation guide
- [Pedestron](https://github.com/hasanirtiza/Pedestron) — upstream pedestrian detection repository
- [MMDetection](https://github.com/open-mmlab/mmdetection) — upstream object detection framework
- [Supplementary grid visualizations](https://galdi.eurecom.io/ExFMA2026_supplementary_material.html) — interactive FLAS grid figures from the paper

## Citation

If you use this code, please cite our work:

> *Semantically Grounded Explainability for Pedestrian Detection Using Anatomical Proportions and Grid-Based Visualization*  
> Chiara Galdi and Romain Giot  
> ExFMA special session at CBMI 2026

## License

See [LICENSE](LICENSE).
