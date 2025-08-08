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
"""A loader for the labeled DIML dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms_v2
from PIL import Image, ImageOps
import os.path as osp
import cv2

import os

class DIML(Dataset):

    def __init__(self,
                 args,
                 data_path: str = "./data/diml",
                 mode: str = "test",
                 **kwargs
                 ):
        """
        we only use indoor dataset!
        DIML LR and HR   (Low resolution and Hight resolution)
        indoor [288, 512] -> [288, 384]   mm Unit, the max: 6m  depth_scale: 10m
        outdoor [384, 640] -> [384, 512]   mm Unit, the max: 65m  depth_scale: 80m
        Args:
            path:
            img_size:
            stage:
            prompt_base:
            prompt_base_norm:
            debug:
        """

        if mode != 'test':
            raise RuntimeError("DIML must be Validation!")
        self.raw_data_path = os.path.join(data_path, "raw_data")
        self.mode = mode
        self.depth_scale = 1000
        self.max_depth = 10.0
        split_file_path = f'{data_path}/splits_files/diml_indoor_test_split_files.txt'
        with open(split_file_path, 'r') as f:
            self.samples_pth = f.readlines()

    def __len__(self):
        return len(self.samples_pth)

    def __getitem__(self, idx):

        image_path = self.raw_data_path + '/' + self.samples_pth[idx].strip().split("--")[0]
        depth_path = self.raw_data_path + '/' + self.samples_pth[idx].strip().split("--")[1]

        raw_image = Image.open(image_path)
        raw_image = np.asarray(raw_image, dtype=np.float32) / 255.0
        raw_image = raw_image.transpose(2, 0, 1)
        raw_image = torch.from_numpy(raw_image).to(torch.float32)

        # GT Depth
        gt_depth = Image.open(depth_path)
        gt_depth = np.asarray(gt_depth, dtype=np.float32)
        gt_depth = np.expand_dims(gt_depth, axis=0)
        gt_depth = gt_depth / self.depth_scale
        gt_depth = np.where(np.isinf(gt_depth), self.max_depth, gt_depth)
        gt_depth = np.where(gt_depth < 1e-3, 1e-3, gt_depth)
        gt_depth = torch.from_numpy(gt_depth).to(torch.float32)  # TODO All use float16

        # mask
        mask = torch.logical_and(gt_depth > 1e-3, gt_depth < self.max_depth)


        batch_dict = {"raw_image": raw_image,
                      "gt_depth": gt_depth,
                      "mask": mask,
                      "domain_index": 0,  # 0 for indoor scene, 1 for outdoor scene
                      "image_path": image_path,
                      "depth_path": depth_path,
                      }
        return batch_dict
