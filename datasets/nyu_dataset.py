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
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json

class NYU(Dataset):
    def __init__(self,
                 args,
                 data_path: str = "./data/nyu",
                 mode: str = "test",
                 **kwargs
                ):
        self.raw_data_path = os.path.join(data_path, "raw_data")
        self.mode = mode
        self.depth_scale = 1000
        self.max_depth = 10.0

        # split file
        split_file_path = f'{data_path}/splits_files/nyu_{mode}.txt'
        with open(split_file_path,'r') as f:
            self.samples_pth = f.readlines()



    def __len__(self):
        return len(self.samples_pth)


    def __getitem__(self, idx):
        if self.mode == 'test':
            image_path = self.raw_data_path + '/official_splits/test/' + self.samples_pth[idx].split()[0]
            depth_path = self.raw_data_path + '/official_splits/test/' + self.samples_pth[idx].split()[1]
        else:
            image_path = self.raw_data_path + '/sync' + self.samples_pth[idx].split()[0]
            depth_path = self.raw_data_path + '/sync' + self.samples_pth[idx].split()[1]

        raw_image = Image.open(image_path)
        raw_image = np.asarray(raw_image, dtype=np.float32) / 255.0
        raw_image = raw_image.transpose(2, 0, 1)
        raw_image = torch.from_numpy(raw_image).to(torch.float32)

        # GT Depth
        gt_depth = Image.open(depth_path)
        gt_depth = np.array(gt_depth)
        mask = np.zeros_like(gt_depth)
        mask[45:472, 43:608] = 1
        gt_depth[mask == 0] = 0
        gt_depth = Image.fromarray(gt_depth)
        gt_depth = np.asarray(gt_depth, dtype=np.float32)
        gt_depth = np.expand_dims(gt_depth, axis=2)
        gt_depth /= self.depth_scale

        gt_depth = torch.from_numpy(gt_depth.transpose((2, 0, 1)))
        # mask
        mask = torch.logical_and(gt_depth > 1e-3, gt_depth < self.max_depth)

        gt_max = gt_depth[mask].max()
        gt_min = gt_depth[mask].min()

        batch_dict = {"raw_image": raw_image,
                      "gt_depth": gt_depth,
                      "mask": mask,
                      "domain_index": 0,  # 0 for indoor scene, 1 for outdoor scene

                      "gt_max": gt_max,
                      "gt_min": gt_min,

                      "image_path": image_path,
                      "depth_path": depth_path,
                      }
        return batch_dict
