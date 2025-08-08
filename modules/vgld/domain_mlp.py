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
import cv2
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from modules.depth_anything_v2.dpt import DepthAnythingV2

from scipy.optimize import curve_fit

class Domain_MLP(nn.Module):
    def __init__(self, args):
        super(Domain_MLP, self).__init__()

        self.args = args
        scale_num = 1
        shift_num = 1

        dim = 512  # We only use ViT-B/16 version : 512,    "RN50": 1024
        if args.fusion == "tci":
            dim *= 2

        self.mlp_classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.cal_global = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256)
        )

        self.cal_shifts_dict = nn.ModuleDict({
            "indoor": nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, shift_num),
            ),
            "outdoor": nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, shift_num),
            ),
        })

        self.cal_scales_dict = nn.ModuleDict({
            "indoor": nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, scale_num),
            ),
            "outdoor": nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, scale_num),
            ),
        })

    def forward(self, text_embeddings, img_embeddings):


        fm = self.args.fusion   # fusion mode
        if fm == "t":
            in_feat = text_embeddings
        elif fm == "i":
            in_feat = img_embeddings
        elif fm == "tci":
            in_feat_list = []
            for (t, i) in zip(text_embeddings, img_embeddings):
                in_feat_list.append(torch.cat([i, t], dim=-1).unsqueeze(0))
            in_feat = torch.cat(in_feat_list, dim=0)
        else:
            raise RuntimeError(f"Invalid fusion args: {fm}")

        domain_logits = self.mlp_classifier(in_feat.squeeze(dim=1))  # [bz, 1, 512] -> [bz, 512] -> [bz, 2]
        domain_vote = torch.softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)
        # Get the path
        pdn = ["indoor", "outdoor"][torch.argmax(domain_vote, dim=-1).squeeze().item()]  # pdn pred_domain_name
        global_feat = self.cal_global(in_feat)

        pred_shift = self.cal_shifts_dict[pdn](global_feat)  # WBJ try to add image_feat
        pred_scale = self.cal_scales_dict[pdn](global_feat)
        pred_scale = torch.exp(pred_scale).reshape(-1, 1, 1, 1)
        pred_shift = torch.exp(pred_shift).reshape(-1, 1, 1, 1)

        return domain_logits, pred_scale, pred_shift


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
        maxfev=10000
    )

    a_opt, b_opt = params_opt
    # print(f"Optimized a: {a_opt:.4f}, Optimized b: {b_opt:.4f}")
    a, b = torch.exp(torch.tensor(a_opt)).to(torch.float32), torch.exp(torch.tensor(b_opt)).to(torch.float32)
    a, b = a.cuda(), b.cuda()
    return a, b
