import glob
import os

import cv2
import mmcv
import numpy as np
import torch
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.ablation_layer import AblationLayerPedestron
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import pedestron_ablation_reshape_transform
from scipy.io import savemat

from mmdet.apis import init_detector

from pdestre_experiments.annotations import read_annotation_boxes
from pdestre_experiments.occlusions import apply_body_part_occlusion


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True),
    )
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False,
        )
    ]
    return dict(img=[img], img_meta=[img_meta])


def _apply_occlusions(image, basename, annotation_dir, occlusion_region, gt_scale):
    if occlusion_region is None:
        return image

    parts = os.path.basename(basename).split('_')
    frame_idx = parts[2]
    annotation_path = os.path.join(annotation_dir, f'{parts[0]}.txt')
    for x, y, w, h in read_annotation_boxes(annotation_path, frame_idx, scale=gt_scale):
        x, y, w, h = (int(value) for value in (x, y, w, h))
        image = apply_body_part_occlusion(image, x, y, w, h, occlusion_region)
    return image


def run_ablation_cam_dataset(
    config,
    checkpoint,
    input_img_dir,
    output_dir,
    annotation_dir=None,
    occlusion_region=None,
    gt_scale=4.0,
    score_threshold=0.3,
    image_size=(960, 540),
):
    os.makedirs(output_dir, exist_ok=True)
    eval_imgs = sorted(glob.glob(os.path.join(input_img_dir, '*.png')))
    model = init_detector(config, checkpoint, device=torch.device('cuda'))
    cfg = model.cfg
    target_layers = [model.backbone]
    prog_bar = mmcv.ProgressBar(len(eval_imgs))

    for image_name in eval_imgs:
        basename = os.path.splitext(os.path.basename(image_name))[0]
        cam_path = os.path.join(output_dir, f'{basename}_grayscale_cam.mat')
        if os.path.exists(cam_path):
            prog_bar.update()
            continue

        image = cv2.imread(image_name)
        image = cv2.resize(image, image_size)
        if annotation_dir is not None and occlusion_region is not None:
            image = _apply_occlusions(
                image, basename, annotation_dir, occlusion_region, gt_scale
            )

        img_transform = ImageTransform(
            size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg
        )
        device = next(model.parameters()).device
        data = _prepare_data(mmcv.imread(image), img_transform, cfg, device)

        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)
        bbox_result = results[0] if isinstance(results, tuple) else results
        bboxes = np.vstack(bbox_result)
        boxes = bboxes[bboxes[:, 4] > score_threshold][:, :4]
        labels = ['person'] * len(boxes)
        targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]

        cam = AblationCAM(
            model,
            target_layers,
            use_cuda=torch.cuda.is_available(),
            reshape_transform=pedestron_ablation_reshape_transform,
            ablation_layer=AblationLayerPedestron(),
            ratio_channels_to_ablate=1.0,
        )
        grayscale_cam = cam(
            input_tensor=data['img'],
            input_meta=data['img_meta'],
            targets=targets,
            return_loss=False,
            rescale=True,
        )[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, image_size)
        savemat(cam_path, {'grayscale_cam': grayscale_cam})
        savemat(
            os.path.join(output_dir, f'{basename}_bbox.mat'),
            {'bboxes': bboxes},
        )
        prog_bar.update()
