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
"""A loader for the labeled DDAD dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms_v2
from PIL import Image, ImageOps
import os.path as osp
import cv2
import glob

import os

class DDAD(Dataset):

    def __init__(self,
                 args,
                 data_path: str = "./data/ddad",
                 mode: str = "test",
                 **kwarg

                 ):
        """
        DDAD : Unit meters
        Args:
            path:
            img_size:
            stage:
            prompt_base:
            prompt_base_norm:
            debug:
        """

        if mode != 'test':
            raise RuntimeError("DDAD must be Validation!")

        self.raw_data_path = os.path.join(data_path, "raw_data/ddad_val")  # diode need "val"
        self.mode = mode
        # self.depth_scale = 65536 / 250
        self.depth_scale = 65536 / 250
        self.max_depth = 80
        split_file_path = f'{data_path}/splits_files/ddad_val_split_files.txt'
        with open(split_file_path, 'r') as f:
            self.samples_pth = f.readlines()

    def __len__(self):
        return len(self.samples_pth)

    def __getitem__(self, idx):

        ## 1. load image,gt

        image_path = self.raw_data_path + '/' + self.samples_pth[idx].strip().split(" ")[0]
        depth_path = self.raw_data_path + '/' + self.samples_pth[idx].strip().split(" ")[1]

        raw_image = Image.open(image_path)
        raw_image = np.asarray(raw_image, dtype=np.float32) / 255.0
        raw_image = raw_image.transpose(2, 0, 1)
        raw_image = torch.from_numpy(raw_image).to(torch.float32)

        # GT Depth

        gt_depth = Image.open(depth_path)
        gt_depth = np.asarray(gt_depth, dtype=np.float32)

        gt_depth = np.expand_dims(gt_depth, axis=0)
        gt_depth /= self.depth_scale
        gt_depth = torch.from_numpy(gt_depth)

        # mask
        mask = torch.logical_and(gt_depth > 1e-3, gt_depth < self.max_depth)

        batch_dict = {"raw_image": raw_image,
                      "gt_depth": gt_depth,
                      "mask": mask,
                      "domain_index": 1,  # 0 for indoor scene, 1 for outdoor scene
                      "image_path": image_path,
                      "depth_path": depth_path,
                      }
        return batch_dict
