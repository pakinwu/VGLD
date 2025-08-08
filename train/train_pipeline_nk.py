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
from torchvision.utils import save_image
import os, sys
import copy
import json, csv
import numpy as np
import time
import clip, cv2
import argparse
from typing import List, Tuple, Optional, Union
from tensorboardX import SummaryWriter

from tools.utils import convert_arg_line_to_args
from dataloaders.MDE_dataloader import Create_MDEDataloader
from modules.vgld.domain_mlp import Domain_MLP

from modules.vgld.loss import MSELoss, SILogLoss, MSLELoss, Levenberg_Marquardt
from modules.vgld.metrics import Metrics
import random
from PIL import Image

def print_grad(grad):
    print("Gradient:", grad)


def isnan_in_grad(grad):
    if True in torch.isnan(grad):
        print("Grad_NAN.")
        # print("Gradient:", grad)

def do_kb_crop(image: np.ndarray):
    height, width, _ = image.shape
    top_margin = int(height - 352)
    left_margin = int((width - 1216) / 2)
    image = image[top_margin : top_margin + 352, left_margin : left_margin + 1216, :]

    return image

def main(args):

    rde_ratio = {"dav1_vits": 1, "dav2_vits": 10, "midas_dpt_swin2_large_384": 10000, "midas_dpt_large_384": 10}[
        f"{args.rde_model}"]
    # def test_pred():
    #     print("GT", GT_BUFFER.detach().cpu().numpy()[0][0][:80, 150])
    #     print("PRED", PRED_BUFFER.detach().cpu().numpy()[0][0][:80, 150])
    # Set seed
    seed = args.seed
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

    train_loader_0 = Create_MDEDataloader(args, dataname=args.train_dataset[0], mode="train")
    train_loader_1 = Create_MDEDataloader(args, dataname=args.train_dataset[1], mode="train")
    test_loader_dict = {}
    for dataname in args.val_dataset:
        test_loader_dict[f"{dataname}"] = Create_MDEDataloader(args, dataname=dataname, mode="test")

    # Model Init
    model = Domain_MLP(args)
    model.train()
    model.cuda()


    # CLIP Init
    clip_model, preprocess = clip.load("ViT-B/16", device="cuda")   # We only use ViT-B/16 version
    clip_model = clip_model.eval()

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(isnan_in_grad)

    global_step = 1

    # TODO
    # Load checkpoints
    # resume

    # TODO
    # Multi GPU

    # Logging
    tb_logger = SummaryWriter(args.log_dir, flush_secs=30)

    # Metrics Init
    metrics = Metrics(dataset_name_list=args.val_dataset, log_dir=args.log_dir)

    # Loss Init
    CE_Loss = nn.CrossEntropyLoss()
    SS_Loss = nn.MSELoss()
    nyu_Loss = MSELoss(max_depth=10.0)
    kitti_Loss = MSELoss(max_depth=80.0)

    # Training Init
    # optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=args.lr, weight_decay=args.wd)
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr)

    steps_per_epoch = len(train_loader_0)
    num_total_steps = args.epochs * steps_per_epoch
    cur_epoch = global_step // steps_per_epoch
    print("Total Steps:", num_total_steps)
    print("Start Training!")

    # Eval Before Training
    # model.eval()
    # with torch.no_grad():
    #     for dataset_name in args.val_dataset:
    #         print(f"Before Training, starting testing Evaluation {dataset_name} in two step.")
    #         test_loader = test_loader_dict[dataset_name]
    #         for step, batch in enumerate(test_loader):
    #             if step > 1:
    #                 break
    #             else:
    #                 for k, v in batch.items():
    #                     if type(v) == torch.Tensor:
    #                         batch[k] = batch[k].cuda()
    #
    #                 pred_depth, pred_scale, pred_shift = model(batch)
    #                 result = metrics.compute(pred_depth, batch["gt_depth"], batch["mask"])
    #                 metrics.save_to_csv(metrics_dict=result, cur_step=-1, dataset_name=dataset_name)
    #         avg_result = metrics.compute_avg_and_save_to_csv(cur_step=-1, dataset_name=dataset_name)
    #         print(f"Avg result in {dataset_name} : {metrics.metric_list} \n {avg_result}")
    # model.eval()


    # Start Training!
    iterator_train_1 = iter(train_loader_1)
    nyu_the_best_absrel = 100.
    kitti_the_best_absrel = 100.
    while cur_epoch < args.epochs:
        start_time = time.time()
        for idx, batch in enumerate(train_loader_0):
            ############### START TRAINING DATA 0
            optimizer.zero_grad()
            text = batch["text"]
            mask = batch["mask"].cuda()
            gt_depth = batch["gt_depth"].cuda()

            # RDE Depth
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

            # CLIP Testual Emb
            text_token = clip.tokenize(text, context_length=77, truncate=True).cuda()
            text_features = clip_model.encode_text(text_token).unsqueeze(dim=1).to(torch.float32)

            # CLIP Visual Emb
            image_features_list = []
            for img_path in batch["image_path"]:
                raw_img = cv2.imread(img_path)
                image = preprocess(Image.fromarray(raw_img)).unsqueeze(0).cuda()
                image_features = clip_model.encode_image(image)
                image_features_list.append(image_features.unsqueeze(0))
            image_features = torch.cat(image_features_list, dim=0)

            domain_logits, pred_scale, pred_shift = model(text_features.to(torch.float32),
                                                          image_features.to(torch.float32))
            pred_depth = 1 / (rde_image * pred_scale + pred_shift)

            loss_nyu = nyu_Loss(pred_depth=pred_depth, gt_depth=gt_depth)
            loss_nyu_domain = CE_Loss(domain_logits, batch["domain_index"].cuda())
            loss_nyu_ss = 10 * SS_Loss(pred_scale, batch["gt_scale"].cuda()) + SS_Loss(pred_shift, batch["gt_shift"].cuda())


            ############### START TRAINING DATA 1
            try:
                batch = next(iterator_train_1)
            except:
                iterator_train_1 = iter(train_loader_1)
                batch = next(iterator_train_1)

            text = batch["text"]
            mask = batch["mask"].cuda()
            gt_depth = batch["gt_depth"].cuda()

            # RDE Depth
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

            # CLIP Testual Emb
            text_token = clip.tokenize(text, context_length=77, truncate=True).cuda()
            text_features = clip_model.encode_text(text_token).unsqueeze(dim=1).to(torch.float32)

            # CLIP Visual Emb
            image_features_list = []
            for img_path in batch["image_path"]:
                raw_img = cv2.imread(img_path)
                raw_img = do_kb_crop(raw_img)  # Only KITTI need !
                image = preprocess(Image.fromarray(raw_img)).unsqueeze(0).cuda()
                image_features = clip_model.encode_image(image)
                image_features_list.append(image_features.unsqueeze(0))
            image_features = torch.cat(image_features_list, dim=0)

            domain_logits, pred_scale, pred_shift = model(text_features.to(torch.float32), image_features.to(torch.float32))
            pred_depth = 1 / (rde_image * pred_scale + pred_shift)

            loss_kitti = kitti_Loss(pred_depth=pred_depth, gt_depth=gt_depth)
            loss_kitti_domain = CE_Loss(domain_logits, batch["domain_index"].cuda())
            loss_kitti_ss = 10 * SS_Loss(pred_scale, batch["gt_scale"].cuda()) + SS_Loss(pred_shift, batch["gt_shift"].cuda())

            total_ss_loss = loss_nyu_ss + loss_kitti_ss
            total_domain_loss = loss_nyu_domain + loss_kitti_domain
            total_loss = args.alpha * loss_nyu + (1 - args.alpha) * loss_kitti + 0.1 * total_domain_loss + args.beta * total_ss_loss
            total_loss.backward()
            optimizer.step()

            # Log
            if global_step % args.log_freq == 0:
                tb_logger.add_scalar("NYU Loss", loss_nyu.item()/args.batch_size, int(global_step))
                tb_logger.add_scalar("KITTI Loss", loss_kitti.item() / args.batch_size, int(global_step))
                tb_logger.add_scalar("SS Loss", total_ss_loss.item() / args.batch_size, int(global_step))

            # Eval
            if global_step % args.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    for dataset_name in args.val_dataset:
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
                                                   domain="nk",
                                                   fusion_mode=args.fusion)
                        print(f"Avg result in {dataset_name} : {metrics.metric_list} \n {avg_result}")
                        tb_logger.add_scalar(f"{dataset_name}_AbsRel", avg_result.abs_rel, -1)
                        tb_logger.add_scalar(f"{dataset_name}_rmse", avg_result.rmse, -1)
                        tb_logger.add_scalar(f"{dataset_name}_d1", avg_result.d_1, -1)

                        # Model save
                        if dataset_name == "nyu":
                            nyu_ckpt_save_name = 'nyu_abs{:.4f}'.format(avg_result.abs_rel)
                            if avg_result.abs_rel < nyu_the_best_absrel:
                                nyu_the_best_absrel = avg_result.abs_rel
                                nyu_save = True
                            else:
                                nyu_save = False
                        elif dataset_name == 'kitti':
                            kitti_ckpt_save_name = 'kitti_abs{:.4f}'.format(avg_result.abs_rel)
                            if avg_result.abs_rel < kitti_the_best_absrel:
                                kitti_the_best_absrel = avg_result.abs_rel
                                kitti_save = True
                            else:
                                kitti_save = False

                    if nyu_save or kitti_save:
                        model_save_name = f'/model_{nyu_ckpt_save_name}_{kitti_ckpt_save_name}_step{global_step}.pth'
                        torch.save(model.state_dict(), args.log_dir + model_save_name)
                model.train()
            if global_step % 200 == 0:
                print(f"epoch{cur_epoch} : {global_step} / {num_total_steps} nyu_Loss {loss_nyu} ; kitti_Loss {loss_kitti} ")
            global_step += 1
        cur_epoch += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VGLD", fromfile_prefix_chars='@')

    # Model
    parser.add_argument('-td', '--train_dataset', type=List[str], default=['nyu', 'kitti'], help='dataset to train on')
    parser.add_argument('-vd', '--val_dataset',   type=List[str], default=['nyu', 'kitti'], help='dataset to validate on')
    parser.add_argument('--rde_model',            type=str,     default='dav1',    help='Relative Depth Estimator type')
    parser.add_argument('--use_lm',         type=bool,  help='lm loss', default=True)  # In this page , Must be False


    # Training
    parser.add_argument('--wd',              type=float, default=1e-3, help='weight decay factor for optimization')
    parser.add_argument('--batch_size',                type=int,    default=8, help='batch size')
    parser.add_argument('--num_worker',               type=int,   help='number of threads to use for data loading', default=8)
    parser.add_argument('--epochs',                type=int,   help='number of epochs', default=12)
    parser.add_argument('--lr',             type=float, help='initial learning rate', default=3e-4)
    parser.add_argument('--fusion',         type=str,  help='end learning rate', default="tci")
    parser.add_argument('--alpha',         type=float,  help='', default=0.7)
    parser.add_argument('--seed',         type=int,  help='', default=0)
    parser.add_argument('--beta',         type=float,  help='', default=0.1)
    # Log and save
    parser.add_argument('--exp_name',          type=str,   help='model name', default=None)
    parser.add_argument('--log_dir',  type=str,   help='directory to save checkpoints and summaries', default='./logs')
    parser.add_argument('--log_freq',         type=int,   help='Logging frequency in global steps', default=100)
    parser.add_argument('--eval_freq',           type=int,   help='Eval frequency in global steps', default=2000)
    args = parser.parse_args()



    rde_model_dict = {"midas-1": "midas_dpt_swin2_large_384",
                      "midas-2": "midas_dpt_large_384",
                      "dav1": "dav1_vits",
                      "dav2": "dav2_vits",
                      }

    args.rde_model = rde_model_dict[args.rde_model]

    if args.exp_name is None:
        lr = "{:.0e}".format(args.lr).replace('-0', '')
        args.exp_name = f"nk_ss_{args.rde_model}_lr{lr}_bz{args.batch_size}_{args.fusion}"

    extra_path = f"seed_{args.seed}"
    args.log_dir = os.path.join(args.log_dir, args.exp_name, extra_path)
    os.system(f"mkdir -p {args.log_dir}")

    main(args)
