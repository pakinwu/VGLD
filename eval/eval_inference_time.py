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
import torch.nn as nn
import os, sys
import numpy as np
import time
import argparse
from typing import List
from tensorboardX import SummaryWriter

from modules.depth_anything.dpt import DepthAnything
from modules.vgld.domain_mlp import Domain_MLP
from modules.vgld.metrics import Metrics

from dataloaders.MDE_dataloader import Create_MDEDataloader
import random
import clip
import cv2
from PIL import Image



def do_kb_crop(image: np.ndarray):
    height, width, _ = image.shape
    top_margin = int(height - 352)
    left_margin = int((width - 1216) / 2)
    image = image[top_margin : top_margin + 352, left_margin : left_margin + 1216, :]

    return image

def main(args):
    # Set seed
    seed = 0
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


    # Model Init
    model = Domain_MLP(args).cuda()
    model.eval()

    # Dav1 init
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    version = 'vits'  # or 'vits', 'vitb', 'vitg'
    dav1_model = DepthAnything(model_configs[version])
    dav1_model.load_state_dict(torch.load(f'./checkpoints/depth_anything_{version}14.pth'))
    dav1_model = dav1_model.cuda().eval()

    # CLIP Init

    clip_model, preprocess = clip.load("ViT-B/16", device="cuda")
    clip_model = clip_model.eval()

    # Metrics Init
    metrics = Metrics(dataset_name_list=args.val_dataset, log_dir=args.log_dir)

    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint)
    else:
        raise RuntimeError("Request checkpoints!")

    global_step = 1
    # Logging
    tb_logger = SummaryWriter(args.log_dir, flush_secs=30)

    with torch.no_grad():
        for dataset_name in args.val_dataset:
            print(f"Starting Evaluation {dataset_name}, global step = {global_step}")
            test_loader = test_loader_dict[dataset_name]
            for idx, batch in enumerate(test_loader):
                total_t = time.time()

                for item in batch:
                    if type(batch[item]) == torch.Tensor:
                        batch[item] = batch[item].cuda()
                mask = batch["mask"]

                rde_st = time.time()
                raw_img = batch["raw_image"]
                raw_img_h, raw_img_w = raw_img.shape[-2], raw_img.shape[-1]
                if "kitti" in dataset_name:
                    size = (350, 1204)
                elif "ddad" in dataset_name:
                    size = (602, 966)
                else:
                    size = (420, 560)
                raw_img = torch.nn.functional.interpolate(
                    raw_img,
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                )
                relative_depth = dav1_model(raw_img).unsqueeze(dim=1)  # [1, 1, 384, 384]
                relative_depth = torch.nn.functional.interpolate(
                    relative_depth,
                    size=(raw_img_h, raw_img_w),
                    mode="bicubic",
                    align_corners=False,
                )

                rel_depth = relative_depth.to(torch.float32)
                rel_depth = torch.where(rel_depth < 1e-3, 1e-3, rel_depth)
                rel_depth = torch.where(torch.isnan(rel_depth), 1e-3, rel_depth)
                rde_image = torch.where(torch.isinf(rel_depth), 1e-3, rel_depth)
                rde_et = time.time()

                # CLIP inference
                clip_t_st = time.time()
                text_token = clip.tokenize(batch["text"], context_length=77, truncate=True).cuda()
                text_features = clip_model.encode_text(text_token).unsqueeze(dim=1)
                clip_t_et = time.time()

                clip_i_st = time.time()
                image_features_list = []
                for img_path in batch["image_path"]:
                    raw_img = cv2.imread(img_path)
                    raw_img = do_kb_crop(raw_img)  # KITTI
                    image = preprocess(Image.fromarray(raw_img)).unsqueeze(0).cuda()
                    image_features = clip_model.encode_image(image)
                    image_features_list.append(image_features.unsqueeze(0))
                image_features = torch.cat(image_features_list, dim=0)
                clip_i_et = time.time()

                vgld_st = time.time()
                domain_logits, pred_scale, pred_shift = model(text_features.to(torch.float32),
                                                              image_features.to(torch.float32))
                pred_depth = 1 / (rde_image * pred_scale + pred_shift)
                vgld_et = time.time()

                total_e = time.time()

                rde_time = rde_et - rde_st
                clip_text_time = clip_t_et - clip_t_st
                clip_img_time = clip_i_et - clip_i_st
                vgld_time = vgld_et - vgld_st
                total_time = total_e - total_t

                result = {"scale": pred_scale.item(),
                           "shift": pred_shift.item(),
                           "rde_time": rde_time,
                           "clip_text_time": clip_text_time,
                           "clip_img_time": clip_img_time,
                           "vgld_time": vgld_time,
                           "total_time": total_time,
                           }
                metrics.save_to_csv_inf_time(metrics_dict=result, cur_step=global_step, dataset_name=dataset_name)
            avg_result = metrics.compute_avg_and_save_to_csv(cur_step=global_step, dataset_name=dataset_name)
            print(f"Avg result in {dataset_name} : {metrics.metric_list} \n {avg_result}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VGLD", fromfile_prefix_chars='@')
    # Model
    parser.add_argument('--rde_model',      type=str,     default='dav1',    help='Relative Depth Estimator type')
    parser.add_argument('-vd', '--val_dataset',   type=List[str], default=['nyu'])
    # parser.add_argument('-vd', '--val_dataset',   type=List[str], default=['sunrgbd'])
    parser.add_argument('--mode',   type=str, default="test")
    parser.add_argument('--use_lm',         type=bool,  help='lm loss', default=True)  # In this page , Must be False

    # Log and save
    parser.add_argument('--exp_name',          type=str,   help='model name', default='EVAL_Inf_time')
    parser.add_argument('--log_dir',  type=str,   help='directory to save checkpoints and summaries', default='./logs')
    parser.add_argument('--ckpt',  type=str,  default="checkpoints/VGLD/vgld_dav1_nk_tci.pth", help='checkpoint path, i.t. checkpoints/VGLD/vgld_dav1_nk_tci.pth')
    parser.add_argument('--fusion',  type=str,   help='MLP fusion mode', default='tci')
    args = parser.parse_args()

    rde_model_dict = {"midas-1": "midas_dpt_swin2_large_384",
                      "midas-2": "midas_dpt_large_384",
                      "dav1": "dav1_vits",
                      "dav2": "dav2_vits",
                      }

    args.rde_model = rde_model_dict[args.rde_model]
    args.log_dir = os.path.join(args.log_dir, args.exp_name)
    main(args)