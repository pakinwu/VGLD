# ------------------------------------------------------------------------------
# Copyright (c) 2025 Bojin Wu , realpakinwu@gmail.com
# All rights reserved.
#
# This source code is released under the MIT license, see LICENSE for details.
#
# If you use this code in your research, please cite the corresponding paper:
# @article{wu2025vgld,
#   title={VGLD:Visually-Guided Linguistic Disambiguation for Monocular Depth Scale Recovery},
#   author={Wu, Bojin and Chen, Jing},
#   journal={arXiv preprint arXiv:2505.02704},
#   year={2025}
# }
# ------------------------------------------------------------------------------
import numpy as np
import os
import cv2
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from modules.vgld.loss import Levenberg_Marquardt

def save_infer_image(
                     batch,
                     pred_depth: torch.Tensor,
                     metrics = None,
                     save_path="",
                     text_mode=0,
                     ):
    # Convert tensors to numpy arrays
    def tensor_to_numpy(tensor):
        return tensor.squeeze().cpu().numpy()

    # Flatten metrics and convert tensors to numpy arrays
    # metrics = metrics.flatten().tolist()

    text = batch["text"]
    mask = batch["mask"]
    raw_image = batch["raw_image"]
    gt_depth = batch["gt_depth"]
    rel_depth = batch["rde_image"]
    gt_scale, gt_shift = Levenberg_Marquardt(rel_depth, gt_depth, mask)

    raw_image_np = tensor_to_numpy(raw_image.squeeze(dim=0))
    raw_image_np = np.transpose(raw_image_np, (1, 2, 0))
    raw_image_np = (raw_image_np - raw_image_np.min()) / (raw_image_np.max() - raw_image_np.min())
    if "nyu" in save_path:
        raw_image_np = raw_image_np[45: 472, 43: 608]

    rel_depth_np = tensor_to_numpy(rel_depth.squeeze(dim=0))
    pred_depth_np = tensor_to_numpy(pred_depth.squeeze(dim=0))
    gt_depth_np = tensor_to_numpy(gt_depth.squeeze(dim=0))
    mask_np = tensor_to_numpy(mask.squeeze(dim=0))
    gt_scale = tensor_to_numpy(gt_scale)
    gt_shift = tensor_to_numpy(gt_shift)



    # rel_depth_np = 1 / rel_depth_np
    # rel_depth_np = 1 - (rel_depth_np - rel_depth_np.min()) / (rel_depth_np.max() - rel_depth_np.min())
    LM_depth_np = 1 / (gt_scale * rel_depth_np + gt_shift)

    LM_error_map = np.abs(gt_depth_np - LM_depth_np) / gt_depth_np
    LM_error_map = np.where(np.isinf(LM_error_map), 0, LM_error_map)
    error_map = np.abs(gt_depth_np - pred_depth_np) / gt_depth_np
    error_map = np.where(np.isinf(error_map), 0, error_map)
    if "kitti" in save_path :
        vmin, vmax = 0., 70.
        fontsize = 12
        fig, axes = plt.subplots(2, 3, figsize=(60, 15))
        car_size = 0.2
    elif "ddad" in save_path:
        vmin, vmax = 0., 80.
        fontsize = 18
        fig, axes = plt.subplots(2, 3, figsize=(40, 20))
        car_size = 0.2
        error_map = np.where(~mask_np, 0, error_map)
        LM_error_map = np.where(~mask_np, 0, LM_error_map)
        gt_depth_np = np.where(~mask_np, 0, gt_depth_np)
    else:
        vmin, vmax = 0., 5.
        fontsize = 18
        fig, axes = plt.subplots(2, 3, figsize=(40, 20))
        car_size = 0.2
        error_map = np.where(~mask_np, 0, error_map)
        LM_error_map = np.where(~mask_np, 0, LM_error_map)
        gt_depth_np = np.where(~mask_np, 0, gt_depth_np)

    # Determine color range from GT depth
    # vmin, vmax = gt_depth_np.min(), gt_depth_np.max()
    # if vmin > pred_depth_np.min():
    #     vmin = pred_depth_np.min()
    # if vmax < pred_depth_np.max():
    #     vmax = pred_depth_np.max()

    # Create a figure with 2 rows and 4 columns

    # First row: Original images with original color maps
    axes[0, 0].imshow(raw_image_np)
    axes[0, 0].set_title("RGB Image", fontsize=fontsize)
    axes[0, 0].axis('off')

    fit_error = axes[0, 1].imshow(LM_error_map, cmap=plt.cm.hot, vmin=0, vmax=0.5)
    axes[0, 1].set_title("Fit Error Map", fontsize=fontsize)
    axes[0, 1].axis('off')
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size=car_size, pad=0.1)  # pad 控制间距
    cbar = plt.colorbar(fit_error, cax=cax)
    cbar.set_label("Error (Absrel)", fontsize=fontsize)
    # cbar.set_label("Normalize", fontsize=fontsize, loc="top", rotation=0, labelpad=5)
    # plt.colorbar(rel_depth, ax=axes[0, 1])

    fit_depth = axes[0, 2].imshow(LM_depth_np, cmap=plt.cm.Spectral, vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Fit Depth Map", fontsize=fontsize)
    axes[0, 2].axis('off')
    divider = make_axes_locatable(axes[0, 2])
    cax = divider.append_axes("right", size=car_size, pad=0.1)  # pad 控制间距
    cbar = plt.colorbar(fit_depth, cax=cax)
    cbar.set_label("Depth Value (m)", fontsize=fontsize)

    # fit_depth = axes[0, 2].imshow(rel_depth_np, cmap=plt.cm.Spectral)
    # axes[0, 2].set_title("Rel Depth Map", fontsize=fontsize)
    # axes[0, 2].axis('off')
    # divider = make_axes_locatable(axes[0, 2])
    # cax = divider.append_axes("right", size=car_size, pad=0.1)  # pad 控制间距
    # cbar = plt.colorbar(fit_depth, cax=cax)
    # cbar.set_label("Depth Value (m)", fontsize=fontsize)
    # cbar.ax.tick_params(labelsize=18)

    gt_depth = axes[1, 0].imshow(gt_depth_np, cmap=plt.cm.Spectral, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("GT Depth Map", fontsize=fontsize)
    axes[1, 0].axis('off')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("right", size=car_size, pad=0.1)  # pad 控制间距
    cbar = plt.colorbar(gt_depth, cax=cax)
    cbar.set_label("Depth Value (m)", fontsize=fontsize)
    # cbar.ax.tick_params(labelsize=18)

    error_map = axes[1, 1].imshow(error_map, cmap=plt.cm.hot, vmin=0, vmax=0.5)
    pred_title = "Error Map"
    pred_title += f"\nPrompt: {text}"
    pred_title += f"\nLM_shift: {gt_shift:.5f}"
    pred_title += f"\nLM_scale: {gt_scale:.5f}"
    axes[1, 1].set_title(pred_title, fontsize=fontsize)
    axes[1, 1].axis('off')
    divider = make_axes_locatable(axes[1, 1])
    cax = divider.append_axes("right", size=car_size, pad=0.1)  # pad 控制间距
    cbar = plt.colorbar(error_map, cax=cax)
    cbar.set_label("Error (Absrel)", fontsize=fontsize)

    pred_depth = axes[1, 2].imshow(pred_depth_np, cmap=plt.cm.Spectral, vmin=vmin, vmax=vmax)
    pred_title = "Predicted Depth"
    rmse = metrics["rmse"]
    abs = metrics["abs_rel"]
    d1 = metrics["d_1"]
    pred_scale = metrics["scale"]
    pred_shift = metrics["shift"]
    pred_title += f"\nRMSE: {rmse:.3f}"
    pred_title += f"\nAbsRel: {abs:.3f}"
    pred_title += f"\nD1: {d1:.3f}"
    pred_title += f"\npred_shift: {pred_shift:.5f}"
    pred_title += f"\npred_scale: {pred_scale:.5f}"
    axes[1, 2].set_title(pred_title, fontsize=fontsize)
    axes[1, 2].axis('off')
    divider = make_axes_locatable(axes[1, 2])
    cax = divider.append_axes("right", size=car_size, pad=0.1)  # pad 控制间距
    cbar = plt.colorbar(pred_depth, cax=cax)
    cbar.set_label("Depth Value (m)", fontsize=fontsize)
    # cbar.ax.tick_params(labelsize=18)

    # Save the image
    plt.savefig(save_path[:-4] + '_' + str(text_mode) + ".png", bbox_inches='tight')
    plt.close(fig)

    with open(save_path[:-4] + ".txt", "a") as f:
        f.write(text[0] + '\n')



def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg
