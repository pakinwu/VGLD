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
import torch, torch.nn as nn

import numpy as np
from scipy.optimize import curve_fit

def Levenberg_Marquardt(preds, gts, masks):
    def model(pred, a, b):
        denominator = np.exp(a) * pred + np.exp(b)
        return 1 / denominator

    a_list = []
    b_list = []
    for (pred, gt, mask) in zip(preds, gts, masks):
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
        a_list.append(a.unsqueeze(dim=0).cuda())
        b_list.append(b.unsqueeze(dim=0).cuda())
    a = torch.cat(a_list, dim=0)
    b = torch.cat(b_list, dim=0)
    return a, b

class SILogLoss(nn.Module):
    """
    SILogLoss
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.alph = 10
        self.lam = 0.85
        self.eps = 1e-8

    def forward(self,
                pred_depth: torch.Tensor,
                gt_depth: torch.Tensor,
                mask: torch.Tensor,
                ):
        # calculate the SILogLoss
        g = torch.log(pred_depth[mask] + self.eps) - torch.log(gt_depth[mask] + self.eps)
        if True in torch.isnan(g):
            print("NAN in SILoss!")
            g = torch.where(torch.isnan(g), torch.tensor(1e-3), g)   # To replace Nan
        Dg = (torch.mean(g ** 2)) - ((self.lam * (torch.mean(g)) ** 2))
        loss = self.alph * torch.sqrt(Dg)

        return loss

class MSELoss(nn.Module):
    """
    SILogLoss
    """
    def __init__(self, cfg=None, alpha=1.0, min_depth=0.001, max_depth=10):
        super().__init__()
        self.cfg = cfg
        self.alpha = alpha
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.loss = torch.nn.L1Loss(reduction='none')

    def forward(self,
                pred_depth: torch.Tensor,
                gt_depth: torch.Tensor,
                mask: torch.Tensor = None,
                ):
        # pred_depth = pred_depth[mask]   # [1,1,h,w] -> [xxx] such as [90787] (valid points)
        # gt_depth = gt_depth[mask]
        # sub = gt_depth - pred_depth
        # mseloss = torch.mean(torch.abs(sub) / gt_depth)
        #
        # return self.alpha * mseloss

        diff = self.loss(pred_depth, gt_depth)  # [bz, 1, h ,w]

        num_pixels = (gt_depth > self.min_depth) * (gt_depth < self.max_depth)


        diff = torch.where(
            (gt_depth > self.min_depth) * (gt_depth < self.max_depth),
            diff,
            torch.zeros_like(diff)
        )

        diff = diff.reshape(diff.shape[0], -1)
        num_pixels = num_pixels.reshape(num_pixels.shape[0], -1).sum(dim=-1) + 1e-6

        loss1 = (diff).sum(dim=-1) / num_pixels

        total_pixels = gt_depth.shape[1] * gt_depth.shape[2] * gt_depth.shape[3]

        weight = num_pixels.to(diff.dtype) / total_pixels

        loss = (loss1 * weight).sum()

        return loss


class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self,
                pred_depth: torch.Tensor,
                gt_depth: torch.Tensor,
                mask: torch.Tensor,
                ):
        # torch.any(torch.isnan(pred_depth))
        pred_depth = pred_depth[mask]   # [1,1,h,w] -> [xxx] such as [90787] (valid points)
        gt_depth = gt_depth[mask]
        pred_depth = torch.log1p(pred_depth)  # log(1 + x)
        if torch.any(torch.isnan(pred_depth)):
            pred_depth = torch.where(torch.isnan(pred_depth), torch.tensor(1e-3), pred_depth)
            print("NANNNNNNNNNNNNNN")
        gt_depth = torch.log1p(gt_depth)  # log(1 + y)
        loss = nn.MSELoss()(pred_depth, gt_depth)
        return loss

class HuberLoss(nn.Module):
    """
    SILogLoss
    """
    def __init__(self, cfg, delta=1.0):
        super().__init__()
        self.cfg = cfg
        self.loss = torch.nn.HuberLoss(reduction = 'none', delta=delta)

    def forward(self,
                pred_depth: torch.Tensor,
                gt_depth: torch.Tensor,
                mask: torch.Tensor,
                ):
        pred_depth = pred_depth[mask]   # [1,1,h,w] -> [xxx] such as [90787] (valid points)
        gt_depth = gt_depth[mask]
        huber_loss = self.loss(pred_depth, gt_depth)
        huber_loss = huber_loss.mean()
        return huber_loss

class InDLoss(nn.Module):
    """
    Inverse Depth Loss
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.alph = 10
        self.lam = 0.85

    def forward(self,
                pred_depth: torch.Tensor,
                gt_depth: torch.Tensor,
                mask: torch.Tensor,
                ):
        # calculate the SILogLoss
        pred_depth = pred_depth[mask]  # [1,1,h,w] -> [xxx] such as [90787] (valid points)
        gt_depth = gt_depth[mask]
        g = 1 / pred_depth - 1 / gt_depth
        Dg = (torch.mean(g ** 2)) - ((self.lam * (torch.mean(g)) ** 2))
        loss = self.alph * torch.sqrt(Dg)

        return loss
