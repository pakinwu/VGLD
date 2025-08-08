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

"""A loader for the labeled KITTI dataset."""
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms_v2
from PIL import Image, ImageOps
from torchvision.transforms import RandomCrop
import json
import os

class KITTI(Dataset):
    def __init__(self,
                 args,
                 data_path: str = "./data/kitti",
                 mode: str = "test",
                 **kwargs
                 ):

        """
        KITTI max 22000, 22000/256 = 85.9Meters
        Args:
            path:
            img_size:
            stage:
            debug:
            prompt_base:
            prompt_base_norm:
        """
        self.raw_data_path = os.path.join(data_path, "raw_data")
        self.mode = mode
        self.depth_scale = 256
        self.max_depth = 80.0
        self.top_margin = int(375 - 352)
        self.left_margin = int((1242 - 1216) / 2)
        split_file_path = f'{data_path}/splits_files/eigen_{mode}_files_with_gt.txt'

        with open(split_file_path, 'r') as f:
            self.samples_pth = f.readlines()


    def __len__(self):
        return len(self.samples_pth)

    def kitti_benchmark_crop(self, image, gt_depth):
        height = image.height
        width = image.width
        top_margin = int(height - 352)  # NOTE: Is it right?  int(height - 352)    A: maybe Yeap
        left_margin = int((width - 1216) / 2)
        gt_depth = gt_depth.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
        image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
        return image, gt_depth

    def __getitem__(self, idx):

        image_path = self.raw_data_path + '/rgb/' + self.samples_pth[idx].split()[0]
        depth_path = self.raw_data_path + '/trainval_gt/' + self.samples_pth[idx].split()[1]

        # Crop Raw Image
        raw_image = Image.open(image_path)
        gt_depth = Image.open(depth_path)
        raw_image, gt_depth = self.kitti_benchmark_crop(raw_image, gt_depth)

        # # Raw_Image
        raw_image = np.asarray(raw_image, dtype=np.float32) / 255.0
        raw_image = torch.from_numpy(raw_image.transpose(2, 0, 1)).to(torch.float32)

        # GT Depth
        gt_depth = np.asarray(gt_depth, dtype=np.float32)
        gt_depth = np.expand_dims(gt_depth, axis=0)
        gt_depth = gt_depth / self.depth_scale
        gt_depth = torch.from_numpy(gt_depth)

        # mask
        mask = torch.logical_and(gt_depth > 1e-3, gt_depth < self.max_depth)

        gt_height, gt_width = 352, 1216
        eval_mask = torch.zeros(mask.shape)
        eval_mask[0][int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
        mask = torch.logical_and(mask, eval_mask)

        gt_max = gt_depth[mask].max()
        gt_min = gt_depth[mask].min()


        batch_dict = {"raw_image": raw_image,
                      "gt_depth": gt_depth,
                      "mask": mask,
                      "domain_index": 1,  # 0 for indoor scene, 1 for outdoor scene
                      "gt_max": gt_max,
                      "gt_min": gt_min,
                      "image_path": image_path,
                      "depth_path": depth_path,
                      }
        return batch_dict


