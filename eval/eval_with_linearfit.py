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
import numpy as np
import argparse
from typing import List, Tuple, Optional, Union
from tensorboardX import SummaryWriter

from dataloaders.MDE_dataloader import Create_MDEDataloader
from modules.vgld.mlp import Least_Squares, Levenberg_Marquardt
from modules.vgld.metrics import Metrics
import random

def main(args):

    rde_ratio = {"dav1_vits": 1, "dav2_vits": 10, "midas_dpt_swin2_large_384": 10000, "midas_dpt_large_384": 10}[
        f"{args.rde_model}"]
    # Set seed
    seed = 666
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Preliminary setting
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # Build DataLoader
    test_loader_dict = {}
    for dataname in args.val_dataset:
        test_loader_dict[f"{dataname}"] = Create_MDEDataloader(args, dataname=dataname, mode="test")

    # Logging
    tb_logger = SummaryWriter(args.log_dir, flush_secs=30)

    # Metrics Init
    metrics = Metrics(dataset_name_list=args.val_dataset, log_dir=args.log_dir)

    with torch.no_grad():
        for dataset_name in args.val_dataset:
            print(f"Starting Evaluation {dataset_name}")
            test_loader = test_loader_dict[dataset_name.split('_')[0]]
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
                rde_ndarray = np.expand_dims(rde_ndarray, axis=0)
                rde_image = torch.from_numpy(rde_ndarray).to(torch.float32).cuda()


                if dataset_name == "kitti":
                    gt_height, gt_width = 352, 1216
                    eval_mask = torch.zeros(mask.shape)
                    eval_mask[0][0][int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                    mask = torch.logical_and(mask, eval_mask.cuda())

                if args.eval_type == "ls":
                    gt_scale, gt_shift = Least_Squares(rde_image, gt_depth, mask)
                    pred_depth = rde_image * gt_scale + gt_shift
                elif args.eval_type == "lm":
                    gt_scale, gt_shift = Levenberg_Marquardt(rde_image, gt_depth, mask)
                    pred_depth = 1 / (rde_image * gt_scale + gt_shift)

                result = metrics.compute(pred_depth, gt_depth, mask)
                result.update({"scale": gt_scale.item(), "shift": gt_shift.item()})

                metrics.save_to_csv(metrics_dict=result, cur_step=-1, dataset_name=dataset_name)
                if idx % 200 == 0:
                    print(f"Eval_{dataset_name}: {idx} / {len(test_loader)}")
            avg_result = metrics.compute_avg_and_save_to_csv(cur_step=-1, dataset_name=dataset_name)
            metrics.save_to_csv_simple(avg_result, dataset_name, args.rde_model,
                                       exp_name=args.exp_name+'_'+args.eval_type,
                                       domain="none",
                                       fusion_mode="none")
            print(f"Avg result in {dataset_name} : {metrics.metric_list} \n {avg_result}")
            tb_logger.add_scalar(f"{dataset_name}_AbsRel", avg_result.abs_rel, -1)
            tb_logger.add_scalar(f"{dataset_name}_rmse", avg_result.rmse, -1)
            tb_logger.add_scalar(f"{dataset_name}_d1", avg_result.d_1, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VGLD_eval_with_linearfit")

    parser.add_argument('--rde_model',      type=str,     default='dav2',    help='Relative Depth Estimator type')
    # parser.add_argument('-vd', '--val_dataset',   type=List[str], default=['nyu', 'kitti', 'diml', 'ddad', 'sunrgbd'])
    parser.add_argument('-vd', '--val_dataset',   type=List[str], default=['ddad'])
    parser.add_argument('--mode',   type=str, default="test")
    parser.add_argument('--use_lm',         type=bool,  help='lm loss', default=False)  # In this page , Must be False
    parser.add_argument('--eval_type',         type=str,  help='lm or least square', default="ls")  # In this page , Must be False

    # Log and save
    parser.add_argument('--exp_name',          type=str,   help='model name', default='0711')
    parser.add_argument('--log_dir',  type=str,   help='directory to save checkpoints and summaries', default='./logs')
    args = parser.parse_args()

    rde_model_dict = {"midas-1": "midas_dpt_swin2_large_384",
                      "midas-2": "midas_dpt_large_384",
                      "dav1": "dav1_vits",
                      "dav2": "dav2_vits",
                      }
    args.rde_model = rde_model_dict[args.rde_model]

    # extra_path = '%s' % (datetime.now().strftime('%y%m%d_%H%M%S'))
    args.exp_name = f'{args.exp_name}_{args.eval_type}_{args.rde_model}'
    args.log_dir = os.path.join(args.log_dir, args.exp_name)

    main(args)

    # ## EVAL ALL
    # for rde_model in ["midas-2", "dav1", "dav2"]:
    #     args.rde_model = rde_model_dict[rde_model]
    #     args.log_dir = os.path.join('./logs', f'{args.exp_name}_{args.eval_type}/{args.rde_model}')
    #     main(args)