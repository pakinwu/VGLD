import torch
import os, sys
import numpy as np
import time
import argparse
from typing import List, Tuple, Optional, Union
from tensorboardX import SummaryWriter
import clip
from tools.utils import convert_arg_line_to_args
from dataloaders.MDE_dataloader import Create_MDEDataloader
from modules.rsa.lanscale import LanScaleModel
from modules.vgld.metrics import Metrics
import random

def main(args):
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

    # Model Init
    # LanScale Model
    model = LanScaleModel(
        text_feat_dim=1024
    )
    model.eval()
    model.cuda()
    print("== LanScale Model Initialized")

    if args.load_model == 'da':
        checkpoint = torch.load("checkpoints/RSA/da_ckpt")
    elif args.load_model == 'midas':
        checkpoint = torch.load("checkpoints/RSA/midas_ckpt")
    # elif args.load_model == 'dpt':
    #     checkpoint = torch.load("checkpoints/RSA/dpt_ckpt")
    else:
        raise RuntimeError(f"Invalid {args.load_model}!")

    model.load_state_dict(checkpoint['model'])

    clip_model, preprocess = clip.load("RN50", device="cuda")


    # Logging
    tb_logger = SummaryWriter(args.log_dir, flush_secs=30)

    # Metrics Init
    metrics = Metrics(dataset_name_list=args.val_dataset, log_dir=args.log_dir)

    model.eval()
    with torch.no_grad():
        for dataset_name in args.val_dataset:
            print(f"Starting Evaluation {dataset_name}")

            test_loader = test_loader_dict[dataset_name]
            for idx, batch in enumerate(test_loader):
                # if idx in need_list:
                text = batch["text"]
                mask = batch["mask"].cuda()
                gt_depth = batch["gt_depth"].cuda()
                image_path = batch["image_path"][0][:-4]
                rde_path = image_path.replace('raw_data', f'{args.rde_model}') + ".npy"
                rde_ndarray = np.load(rde_path).astype(np.float32)
                rde_ndarray = np.where(rde_ndarray < 1e-3, 1e-3, rde_ndarray)
                rde_ndarray = np.where(np.isnan(rde_ndarray), 1e-3, rde_ndarray)
                rde_ndarray = np.where(np.isinf(rde_ndarray), 1e-3, rde_ndarray)
                rde_ndarray = np.expand_dims(rde_ndarray, axis=0)
                rde_image = torch.from_numpy(rde_ndarray).to(torch.float32).cuda()

                if dataset_name == "kitti":
                    gt_height, gt_width = 352, 1216
                    eval_mask = torch.zeros(mask.shape)
                    eval_mask[0][0][int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                    mask = torch.logical_and(mask, eval_mask.cuda())

                text_token = clip.tokenize(text, context_length=77, truncate=True).cuda()
                text_features = clip_model.encode_text(text_token).squeeze(dim=0).to(torch.float32)

                rsa_st = time.time()
                pred_scale, pred_shift = model(text_features)
                pred_depth = 1 / (pred_scale * rde_image + pred_shift)
                rsa_et = time.time()

                pred_scale, pred_shift = np.squeeze(pred_scale.detach().cpu().numpy()), np.squeeze(pred_shift.detach().cpu().numpy())

                result = metrics.compute(pred_depth, gt_depth, mask)

                rsa_time = rsa_et - rsa_st
                result.update({"scale": pred_scale, "shift": pred_shift, "times": rsa_time})
                # if idx in need_list:
                os.system(f"mkdir -p {args.log_dir}/output_image_rsa/{dataset_name}")
                metrics.save_to_csv(metrics_dict=result, cur_step=-1, dataset_name=dataset_name)
                if idx % 200 == 0:
                    print(f"Eval_{dataset_name}: {idx} / {len(test_loader)}")
            avg_result = metrics.compute_avg_and_save_to_csv(cur_step=-1, dataset_name=dataset_name)

            metrics.save_to_csv_simple(avg_result, dataset_name, args.rde_model,
                                       exp_name=args.exp_name,
                                       domain="nk",
                                       fusion_mode="t")   # We only eval RSA-NK model
            print(f"Avg result in {dataset_name} : {metrics.metric_list} \n {avg_result}")
            tb_logger.add_scalar(f"{dataset_name}_AbsRel", avg_result.abs_rel, -1)
            tb_logger.add_scalar(f"{dataset_name}_rmse", avg_result.rmse, -1)
            tb_logger.add_scalar(f"{dataset_name}_d1", avg_result.d_1, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VGLD", fromfile_prefix_chars='@')
    # Model
    parser.add_argument('--load_model',      type=str,     default='da',    help='da or midas')
    # parser.add_argument('-vd', '--val_dataset',   type=List[str], default=['nyu', 'kitti', 'diml', 'ddad', 'sunrgbd'])
    parser.add_argument('-vd', '--val_dataset',   type=List[str], default=['sunrgbd'])
    parser.add_argument('--mode',   type=str, default="test")
    parser.add_argument('--use_lm',         type=bool,  help='lm loss', default=False)  # In this page , Must be False

    # Log and save
    parser.add_argument('--exp_name',          type=str,   help='model name', default='EVAL_RSA_0710')
    parser.add_argument('--log_dir',  type=str,   help='directory to save checkpoints and summaries', default='./logs')
    args = parser.parse_args()

    if args.load_model == "da":
        args.rde_model = "dav1_vits"
    elif args.load_model == "midas":
        args.rde_model = "midas_dpt_swin2_large_384"
    else:
        raise RuntimeError("warning")

    # extra_path = '%s' % (datetime.now().strftime('%y%m%d_%H%M%S'))
    args.log_dir = os.path.join(args.log_dir, f'{args.exp_name}_{args.rde_model}')

    main(args)
