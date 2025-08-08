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
from typing import List
from tensorboardX import SummaryWriter

from tools.utils import convert_arg_line_to_args
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
    rde_ratio = {"dav1_vits": 1, "dav2_vits": 10, "midas_dpt_swin2_large_384": 10000, "midas_dpt_large_384": 10}[
        f"{args.rde_model}"]
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
    assert type(args.val_dataset) == list

    # Build DataLoader
    test_loader_dict = {}
    for dataname in args.val_dataset:
        test_loader_dict[f"{dataname}"] = Create_MDEDataloader(args, dataname=dataname, mode="test")

    # Model Init
    model = Domain_MLP(args).cuda()
    model.eval()

    # CLIP Init
    clip_model, preprocess = clip.load("ViT-B/16", device="cuda")   # We only use ViT-B/16 version
    clip_model = clip_model.eval()

    global_step = 1
    # Logging
    tb_logger = SummaryWriter(args.log_dir, flush_secs=30)

    # Metrics Init
    metrics = Metrics(dataset_name_list=args.val_dataset, log_dir=args.log_dir)

    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint)
    else:
        raise RuntimeError("Request checkpoints!")

    with torch.no_grad():
        for dataset_name in args.val_dataset:
            print(f"Starting Evaluation {dataset_name}, global step = {global_step}")
            test_loader = test_loader_dict[dataset_name]
            for idx, batch in enumerate(test_loader):
                text = batch["text"]
                mask = batch["mask"].cuda()
                gt_depth = batch["gt_depth"].cuda()
                image_path = batch["image_path"][0][:-4]
                rde_path = image_path.replace('raw_data', f'{args.rde_model}') + ".npy"
                rde_ndarray = np.load(rde_path).astype(np.float32) / rde_ratio
                rde_ndarray = np.where(rde_ndarray < 1e-3, 1e-3, rde_ndarray)
                rde_ndarray = np.where(np.isnan(rde_ndarray), 1e-3, rde_ndarray)
                rde_ndarray = np.where(np.isinf(rde_ndarray), 1e-3, rde_ndarray)
                rde_ndarray = np.expand_dims(rde_ndarray, axis=0)
                if len(rde_ndarray.shape) == 3:
                    rde_ndarray = np.expand_dims(rde_ndarray, axis=0)
                rde_image = torch.from_numpy(rde_ndarray).to(torch.float32).cuda()

                if dataset_name == "kitti":
                    gt_height, gt_width = 352, 1216
                    eval_mask = torch.zeros(mask.shape)
                    eval_mask[0][0][int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                    mask = torch.logical_and(mask, eval_mask.cuda())

                # CLIP Testual Emb
                text_token = clip.tokenize(text, context_length=77, truncate=True).cuda()
                text_features = clip_model.encode_text(text_token).unsqueeze(dim=0).to(torch.float32)

                # CLIP Visual Emb
                image_features_list = []
                for img_path in batch["image_path"]:
                    raw_img = cv2.imread(img_path)
                    if dataset_name == "kitti":
                        raw_img = do_kb_crop(raw_img)  # KITTI
                    image = preprocess(Image.fromarray(raw_img)).unsqueeze(0).cuda()
                    image_features = clip_model.encode_image(image)
                    image_features_list.append(image_features.unsqueeze(0))
                image_features = torch.cat(image_features_list, dim=0)

                domain_logits, pred_scale, pred_shift = model(text_features.to(torch.float32),
                                                              image_features.to(torch.float32))
                pred_depth = 1 / (rde_image * pred_scale + pred_shift)

                pred_scale, pred_shift = np.squeeze(pred_scale.detach().cpu().numpy()), np.squeeze(
                    pred_shift.detach().cpu().numpy())

                result = metrics.compute(pred_depth, gt_depth, mask)
                result.update({"scale": pred_scale.item(), "shift": pred_shift.item()})
                metrics.save_to_csv(metrics_dict=result, cur_step=-1, dataset_name=dataset_name)
                if idx % 200 == 0:
                    print(f"Eval_{dataset_name}: {idx} / {len(test_loader)}")
            avg_result = metrics.compute_avg_and_save_to_csv(cur_step=-1, dataset_name=dataset_name)
            metrics.save_to_csv_simple(avg_result, dataset_name, args.rde_model,
                                       exp_name=args.exp_name,
                                       domain=args.domain,
                                       fusion_mode=args.fusion)
            print(f"Avg result in {dataset_name} : {metrics.metric_list} \n {avg_result}")
            tb_logger.add_scalar(f"{dataset_name}_AbsRel", avg_result.abs_rel, -1)
            tb_logger.add_scalar(f"{dataset_name}_rmse", avg_result.rmse, -1)
            tb_logger.add_scalar(f"{dataset_name}_d1", avg_result.d_1, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VGLD", fromfile_prefix_chars='@')
    # Model
    parser.add_argument('--rde_model',      type=str,     default='dav2',    help='Relative Depth Estimator type')
    parser.add_argument('-vd', '--val_dataset',   type=List[str], default=['nyu', 'kitti', 'diml', 'ddad', 'sunrgbd'])
    # parser.add_argument('-vd', '--val_dataset',   type=List[str], default=['sunrgbd'])
    parser.add_argument('--mode',   type=str, default="test")
    parser.add_argument('--use_lm',         type=bool,  help='lm loss', default=True)  # In this page , Must be False

    # Log and save
    parser.add_argument('--exp_name',          type=str,   help='model name', default='EVAL_VGLD_ALLINONE_0712')
    parser.add_argument('--log_dir',  type=str,   help='directory to save checkpoints and summaries', default='./logs')
    parser.add_argument('--ckpt',  type=str,   required=False, help='checkpoint path, i.t. checkpoints/VGLD/vgld_dav1_nk_tci.pth')
    parser.add_argument('--fusion',  type=str,   help='MLP fusion mode', default='tci')
    args = parser.parse_args()


    rde_model_dict = {"midas-1": "midas_dpt_swin2_large_384",
                      "midas-2": "midas_dpt_large_384",
                      "dav1": "dav1_vits",
                      "dav2": "dav2_vits",
                      }
    domain_dataset_dict = { "n": ["nyu"],
                            "k": ["kitti"],
                            "nk": ['ddad']}
                            # "nk": ['nyu', 'kitti', 'diml', 'ddad', 'sunrgbd']}
    # Choise fusion_mode
    fusion_mode = args.ckpt[:-4].split('_')[-1]
    if fusion_mode not in ["t", "i", "tci"]:
        raise RuntimeError(f"Invalid fusion mode. ckpt:{args.ckpt}")
    args.fusion = fusion_mode

    # Choise rde_model
    rde_model = args.ckpt[:-4].split('_')[1]
    if rde_model not in ["midas-1", "midas-2", "dav1", "dav2"]:
        raise RuntimeError(f"Invalid rde_model. ckpt:{args.ckpt}")

    args.rde_model = rde_model_dict[rde_model]

    # extra_path = '%s' % (datetime.now().strftime('%y%m%d_%H%M%S'))
    args.log_dir = os.path.join(args.log_dir, f'{args.exp_name}_{args.rde_model}')

    main(args)

    # ## EVAL VGLD ALL IN ONE!!
    # # parser ckpt required -> False
    # for val_domain in ["n", "k", "nk"]:
    #     args.val_dataset = domain_dataset_dict[val_domain]
    #     args.domain = val_domain
    #     for fusion_mode in ["t", "i", "tci"]:
    #         args.fusion = fusion_mode
    #         for rde_model in ["midas-1", "midas-2", "dav1", "dav2"]:
    #             args.rde_model = rde_model_dict[rde_model]
    #             args.ckpt = f"checkpoints/VGLD/vgld_{rde_model}_{val_domain}_{fusion_mode}.pth"
    #             args.log_dir = os.path.join("./logs", f'{args.exp_name}/{val_domain}_{fusion_mode}_{args.rde_model}')
    #             main(args)