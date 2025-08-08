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
import torch
import os, sys
import argparse
from typing import List, Tuple, Optional, Union
from dataloaders.MDE_dataloader import Create_MDEDataloader
import numpy as np
from scipy.optimize import curve_fit

MAXFEV = 10000


def Levenberg_Marquardt(pred, gt, mask):
    def model(pred, a, b):
        denominator = np.exp(a) * pred + np.exp(b)
        return 1 / denominator

    pred = pred[mask].to(torch.float64)
    gt = gt[mask].to(torch.float64)
    pred_array = pred.detach().cpu().numpy()
    gt_array = gt.detach().cpu().numpy()
    initial_guess = (0.0, 0.0)  # (a_initial, b_initial)

    params_opt, params_cov = curve_fit(
        model,
        pred_array,
        gt_array,
        p0=initial_guess,
        method="lm",
        maxfev=MAXFEV
    )

    a_opt, b_opt = params_opt
    a, b = torch.exp(torch.tensor(a_opt)).to(torch.float32), torch.exp(torch.tensor(b_opt)).to(torch.float32)
    a, b = a.cuda(), b.cuda()
    return a, b

def main(args):
    # Build DataLoader

    rde_ratio = {"dav1_vits": 1, "dav2_vits": 10, "midas_dpt_swin2_large_384": 10000, "midas_dpt_large_384": 10}[
        f"{args.rde_model}"]
    assert type(args.dataset_list) == list
    dataloader_dict = {}
    for dataname in args.dataset_list:
        dataloader_dict[f"{dataname}"] = Create_MDEDataloader(args, dataname, mode=args.mode)

    ss_emb_dict = {}
    for dataset_name in args.dataset_list:
        test_loader = dataloader_dict[dataset_name]
        for idx, batch in enumerate(test_loader):
            mask = batch["mask"].cuda()
            gt_depth = batch["gt_depth"].cuda()

            image_path = batch["image_path"][0][:-4]
            rde_path = image_path.replace('raw_data', f'{args.rde_model}') + ".npy"
            rde_ndarray = np.load(rde_path).astype(np.float32) / rde_ratio
            rde_ndarray = np.where(rde_ndarray < 1e-3, 1e-3, rde_ndarray)
            rde_ndarray = np.where(np.isnan(rde_ndarray), 1e-3, rde_ndarray)
            rde_ndarray = np.where(np.isinf(rde_ndarray), 1e-3, rde_ndarray)
            rde_ndarray = np.expand_dims(rde_ndarray, axis=0)
            rel = torch.from_numpy(rde_ndarray).to(torch.float32).cuda()

            ss_path = f"./data/{dataset_name}/ss_embedding/{args.mode}_ss_{args.rde_model}_maxfev-{MAXFEV}.pth"
            os.system(f"mkdir -p ./data/{dataset_name}/ss_embedding")
            # if dataset_name == "kitti":
            #     gt_height, gt_width = 352, 1216
            #     eval_mask = torch.zeros(mask.shape)
            #     eval_mask[0][0][int(0.40810811 * gt_height):int(0.99189189 * gt_height),
            #     int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            #     mask = torch.logical_and(mask, eval_mask.cuda())
            if len(rel.shape) == 3:
                rel = rel.unsqueeze(dim=0)
                # print(rel.shape)
            gt_scale, gt_shift = Levenberg_Marquardt(rel, gt_depth, mask)
            if idx % 200 == 0:
                print(f'{idx}: {gt_scale}  {gt_shift}, {rel.shape}')
            ss_emb_dict.update({f"{image_path}": {"gt_scale": gt_scale, "gt_shift": gt_shift}})
        torch.save(ss_emb_dict, ss_path)
        print(f"Saving in {ss_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ss_embedding generator")
    parser.add_argument('--rde_model',      type=str,     default='dav2',    help='Relative Depth Estimator type')
    parser.add_argument('--dataset_list',   type=List[str], default=['ddad'])
    parser.add_argument('--mode',   type=str, default="test")
    parser.add_argument('--use_lm',         type=bool,  help='lm loss', default=False)  # In this page , Must be False
    args = parser.parse_args()

    rde_model_dict = {"midas-1": "midas_dpt_swin2_large_384",
                      "midas-2": "midas_dpt_large_384",
                      "dav1": "dav1_vits",
                      "dav2": "dav2_vits",
                      }
    args.rde_model = rde_model_dict[args.rde_model]

    main(args)