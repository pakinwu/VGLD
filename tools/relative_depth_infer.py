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
import cv2
import torch
import PIL
import numpy as np
import glob
import os
import argparse
from modules.depth_anything.dpt import DepthAnything
from modules.depth_anything_v2.dpt import DepthAnythingV2
from modules.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from modules.midas.model_loader import load_model
from torchvision.transforms import Compose

from dataloaders.MDE_dataloader import Create_MDEDataloader

from torchvision.utils import save_image
def save_img_from_ndarray(ndarray, file_name):
    img = torch.from_numpy(ndarray).to(torch.float32)
    save_image(img, file_name)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def do_kb_crop(image: np.ndarray):
    height, width, _ = image.shape
    top_margin = int(height - 352)
    left_margin = int((width - 1216) / 2)
    image = image[top_margin : top_margin + 352, left_margin : left_margin + 1216, :]

    return image


def image2tensor(raw_image, resize_h, resize_w):
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    h, w = raw_image.shape[:2]

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    image = image.to(DEVICE)

    return image, (h, w)


class dav1(object):
    def __init__(self, args):
        self.args = args

        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        self.version = 'vits'  # or 'vits', 'vitb', 'vitg'  We only use vits version
        model = DepthAnything(model_configs[self.version])
        model.load_state_dict(torch.load(f'./checkpoints/depth_anything_{self.version}14.pth'))
        self.model = model.to(DEVICE).eval()


    def infer(self, batch, dn, save_output=True):

        img_path = batch["image_path"][0]
        raw_img = batch["raw_image"].cuda()
        raw_img_h, raw_img_w = raw_img.shape[-2], raw_img.shape[-1]

        if "kitti" in dn:
            size = (350, 1204)
        elif "ddad" in dn:
            size = (602, 966)
        else:
            size = (420, 560)

        raw_img = torch.nn.functional.interpolate(
            raw_img,
            size=size,
            mode="bicubic",
            align_corners=False,
        )
        relative_depth = self.model(raw_img).unsqueeze(dim=0)  # [1, 1, 384, 384]
        relative_depth = torch.nn.functional.interpolate(
            relative_depth,
            size=(raw_img_h, raw_img_w),
            mode="bicubic",
            align_corners=False,
        )


        depth_ndarray = relative_depth.squeeze(0).squeeze(0).cpu().detach().numpy()

        depth_ndarray = depth_ndarray.astype(np.float16())  # [raw_h, raw_w]

        if save_output:
            directory, filename = os.path.split(img_path)
            rel_path = os.path.relpath(directory, f'./data/{dn}/raw_data')
            if dn == "kitti" or dn == "vkitti":
                save_dir = f"data/{dn}/{self.args.rde_model}/rgb/" + rel_path
            else:
                save_dir = f"data/{dn}/{self.args.rde_model}/" + rel_path

            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir + "/" + filename[:-4] + ".npy"
            np.save(save_path, depth_ndarray)
            print(f"Saving:  {save_path}")
            print(f"Size = {depth_ndarray.shape}")
        else:
            print("Note : Do Not Saving!")
            print(f"Size = {depth_ndarray.shape}"
                  )


class dav2(object):
    def __init__(self, args):
        self.args = args
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 129, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.version = 'vits'  # or 'vits', 'vitb', 'vitg'  We only use vits version

        model = DepthAnythingV2(**model_configs[self.version])
        model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{self.version}.pth', map_location='cpu'))
        self.model = model.to(DEVICE).eval()
    
    def infer(self, batch, dn, save_output=True):

        img_path = batch["image_path"][0]
        raw_img = batch["raw_image"].cuda()
        raw_img_h, raw_img_w = raw_img.shape[-2], raw_img.shape[-1]

        if "kitti" in dn:
            size = (350, 1204)
        elif "ddad" in dn:
            size = (602, 966)
        else:
            size = (420, 560)

        raw_img = torch.nn.functional.interpolate(
            raw_img,
            size=size,
            mode="bicubic",
            align_corners=False,
        )
        relative_depth = self.model(raw_img).unsqueeze(dim=0)  # [1, 1, 384, 384]
        relative_depth = torch.nn.functional.interpolate(
            relative_depth,
            size=(raw_img_h, raw_img_w),
            mode="bicubic",
            align_corners=False,
        )

        depth_ndarray = relative_depth.squeeze(0).squeeze(0).cpu().detach().numpy()

        depth_ndarray = depth_ndarray.astype(np.float16())  # [raw_h, raw_w]

        if save_output:
            directory, filename = os.path.split(img_path)
            rel_path = os.path.relpath(directory, f'./data/{dn}/raw_data')
            if dn == "kitti" or dn == "vkitti":
                save_dir = f"data/{dn}/{self.args.rde_model}/rgb/" + rel_path
            else:
                save_dir = f"data/{dn}/{self.args.rde_model}/" + rel_path

            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir + "/" + filename[:-4] + ".npy"
            np.save(save_path, depth_ndarray)
            print(f"Saving:  {save_path}")
            print(f"Size = {depth_ndarray.shape}")
        else:
            print("Note : Do Not Saving!")
            print(f"Size = {depth_ndarray.shape}"
                  )


class midas(object):
    def __init__(self, args):
        self.args = args
        self.version = "dpt_swin2_large_384" if args.rde_model == "midas_dpt_swin2_large_384" else "dpt_large_384"
        model_path = f"./checkpoints/{self.version}.pt"  # dpt_swin2_large_384.pt
        self.model, transform, net_w, net_h = load_model(DEVICE, model_path, self.version, optimize=False, height=None,
                                                    square=False)
        self.model = self.model.to(DEVICE).eval()

    def infer(self, batch, dn, save_output=True):
        img_path = batch["image_path"][0]
        raw_img = batch["raw_image"].cuda()
        raw_img_h, raw_img_w = raw_img.shape[-2], raw_img.shape[-1]
        raw_img = torch.nn.functional.interpolate(
            raw_img,
            size=(384, 384),
            mode="bicubic",
            align_corners=False,
        )
        relative_depth = self.model(raw_img).unsqueeze(dim=0)  # [1, 1, 384, 384]
        depth_ndarray = torch.nn.functional.interpolate(
            relative_depth,
            size=(raw_img_h, raw_img_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0).cpu().detach().numpy()

        depth_ndarray = depth_ndarray.astype(np.float16())  # [raw_h, raw_w]

        if save_output:
            directory, filename = os.path.split(img_path)
            rel_path = os.path.relpath(directory, f'./data/{dn}/raw_data')
            if dn == "kitti" or dn == "vkitti":
                save_dir = f"data/{dn}/{self.args.rde_model}/rgb/" + rel_path
            else:
                save_dir = f"data/{dn}/{self.args.rde_model}/" + rel_path

            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir + "/" + filename[:-4] + ".npy"
            np.save(save_path, depth_ndarray)
            print(f"Saving:  {save_path}")
            print(f"Size = {depth_ndarray.shape}")
        else:
            print("Note : Do Not Saving!")
            print(f"Size = {depth_ndarray.shape}")

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='RDE_infer')
    parser.add_argument('--rde_model',      type=str,     default='dav1',    help='Relative Depth Estimator type')
    parser.add_argument('--dataset',   type=str, default='sunrgbd')
    parser.add_argument('--mode',   type=str, default="test")
    parser.add_argument('--use_lm',         type=bool,  help='lm loss', default=False)  # In this page , Must be False
    args = parser.parse_args()

    rde_model_dict = {"midas-1": "midas_dpt_swin2_large_384",
                      "midas-2": "midas_dpt_large_384",
                      "dav1": "dav1_vits",
                      "dav2": "dav2_vits",
                      }
    args.rde_model = rde_model_dict[args.rde_model]

    if "midas" in args.rde_model:
        model = midas(args)
    elif args.rde_model == "dav1_vits":
        model = dav1(args)
    elif args.rde_model == "dav2_vits":
        model = dav2(args)



    # DataLoader
    test_loader = Create_MDEDataloader(args, dataname=args.dataset, mode="test")

    total_num = len(test_loader)
    for idx, batch in enumerate(test_loader):
        model.infer(batch, args.dataset)  # Infer & Saving All in One!
        print(f"{args.dataset}: {idx} / {total_num}")
