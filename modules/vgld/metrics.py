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
import numpy as np
import pandas as pd
import os, csv
from typing import Union, List, Dict

class Metrics(object):
    def __init__(self, dataset_name_list: list, log_dir):
        # Note: only used in Val dataset
        self.metric_list = ['mse', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog', 'd_1', 'd_2', 'd_3', 'scale', 'shift', 'times',]
        self.metric_list_inf = ['rde_time', 'clip_text_time', 'clip_img_time', 'vgld_time', 'total_time']
        self.metric_path_dict = {}

        # create csv files
        for dataname in dataset_name_list:
            metric_file_path = os.path.join(log_dir, f"metrics/{dataname}")
            self.metric_path_dict[f"{dataname}"] = metric_file_path
            os.makedirs(metric_file_path, exist_ok=True)

    def save_to_csv_simple(self, metrics_dict: Dict, dataset_name: str, rde_model: str, exp_name: str = "", domain: str = "nk", fusion_mode: str = "tci"):
        log_path = f"logs/METRIC_SIMPLE_RESULT/{exp_name}"
        simple_metric_list = ['fusion_mode', 'rde_model', 'dataset', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'd_1', 'd_2', 'd_3']
        os.makedirs(log_path, exist_ok=True)
        metric_file = os.path.join(log_path, f"{domain}_metric_simple.csv")
        if not os.path.exists(metric_file):
            with open(metric_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(simple_metric_list)

        with open(metric_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            m = [metrics_dict[item] for item in simple_metric_list[3:]]
            writer.writerow([fusion_mode, rde_model, dataset_name] + m)

    def save_to_csv(self, cur_step: int, metrics_dict: Dict, dataset_name: str):

        metric_file = os.path.join(self.metric_path_dict[f"{dataset_name}"], f"{dataset_name}_metrics_step-{cur_step}.csv")
        if not os.path.exists(metric_file):
            with open(metric_file, mode='w', newline='' ) as f:
                writer = csv.writer(f)
                writer.writerow(self.metric_list)

        with open(metric_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics_dict.values())

    def save_to_csv_inf_time(self, cur_step: int, metrics_dict: Dict, dataset_name: str):

        metric_file = os.path.join(self.metric_path_dict[f"{dataset_name}"], f"{dataset_name}_metrics_step-{cur_step}.csv")
        if not os.path.exists(metric_file):
            with open(metric_file, mode='w', newline='' ) as f:
                writer = csv.writer(f)
                writer.writerow(self.metric_list_inf)

        with open(metric_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics_dict.values())

    def compute_avg_and_save_to_csv(self, cur_step: int, dataset_name: str):

        metric_file = os.path.join(self.metric_path_dict[f"{dataset_name}"], f"{dataset_name}_metrics_step-{cur_step}.csv")
        data_frame = pd.read_csv(metric_file)
        metric_means = data_frame.mean()

        cur_step = pd.Series([cur_step], index=["step"], dtype=int)
        metric_means = pd.concat([cur_step, metric_means])

        avg_metric_file = os.path.join(self.metric_path_dict[f"{dataset_name}"], f"avg_metrics_{dataset_name}.csv")

        if not os.path.exists(avg_metric_file):
            with open(avg_metric_file, mode='w', newline='' ) as f:
                writer = csv.writer(f)
                writer.writerow(["step"] + self.metric_list)

        with open(avg_metric_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metric_means)

        return metric_means

    def compute(self,
              pred_depth: Union[np.ndarray, torch.Tensor],
              gt_depth: Union[np.ndarray, torch.Tensor],
              mask: Union[np.ndarray, torch.Tensor],
              ):
        """ NOTE : Only use in Val
        Args:
            pred_depth:
            gt_depth:
            mask:
        Return:
            Dict
            - MSE
            - Abs Rel
            - Sq Rel
            - RMSE
            - RMSE Log
            - Log10
            - SILog
            - D1, D2, D3
        """
       # Transform to numpy format
        if type(pred_depth) == torch.Tensor:
            pred_depth = pred_depth.detach().cpu().numpy()
        if type(gt_depth) == torch.Tensor:
            gt_depth = gt_depth.detach().cpu().numpy()
        if type(mask) == torch.Tensor:
            mask = mask.detach().cpu().numpy()

        # Init metric
        bz, _, h, w = gt_depth.shape
        assert bz == 1

        # for pred, gt, time in zip(pred_depth, gt_depth, times):  # because only used in Val, Batch size must be 1.
        gt_depth = gt_depth[0, :, :]  # [1, h, w] -> [h, w]
        pred_depth = pred_depth[0, :, :]
        mask = mask[0, :, :]

        # mask = (gt_depth > self.min_depth) & (gt_depth < self.max_depth)
        pred_valid = pred_depth[mask]  # [h, w] -> [xx](valid num points)
        pred_valid = np.clip(pred_valid, a_min=1e-3, a_max=np.inf)  # To prevent negative number
        gt_valid = gt_depth[mask]

        ## Higher is better
        thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
        d_1 = np.mean(thresh < 1.25)
        d_2 = np.mean(thresh < 1.25 ** 2)
        d_3 = np.mean(thresh < 1.25 ** 3)

        ## Lower is better
        sub = gt_valid - pred_valid
        logsub = np.log(pred_valid) - np.log(gt_valid)
        silog = np.sqrt(np.mean(logsub ** 2) - np.mean(logsub) ** 2) * 100  # Standard SILoss, Refer to Eigen

        mse = np.mean(np.abs(sub) ** 2)
        abs_rel = np.mean(np.abs(sub) / gt_valid)
        sq_rel = np.mean((sub ** 2) / gt_valid)

        rmse = np.sqrt(mse)
        rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
        log10 = np.mean(np.abs(np.log10(gt_valid) - np.log10(pred_valid)))


        metrics_dict = {
            "mse": mse,
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "log10": log10,
            "silog": silog,
            "d_1": d_1,
            "d_2": d_2,
            "d_3": d_3,
        }

        # print(metrics_dict)
        return metrics_dict
