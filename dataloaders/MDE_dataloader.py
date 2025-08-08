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
from torch.utils.data import Dataset, DataLoader
import os
import random
import time
from datasets.nyu_dataset import NYU
from datasets.kitti_dataset import KITTI
from datasets.diml_dataset import DIML
from datasets.sunrgbd_dataset import SUNRGBD
from datasets.ddad_dataset import DDAD
import json
import datasets as ALL_DATASETS

ALL_DATASETS = {"nyu": NYU,
                "kitti": KITTI,
                "diml": DIML,
                "ddad": DDAD,
                "sunrgbd": SUNRGBD,
                }


class MDEDataset(Dataset):
    def __init__(
            self,
            args,
            root : str = "./data",
            dataset_name: str = "",
            mode: str = "test",  # "train" "test"
            rde_model: str = "dav2",
            use_lm : bool = True,
            **kwargs,
    ) -> None:
        super(Dataset, self).__init__()
        self.args = args
        data_path = os.path.join(root, dataset_name)
        self.use_lm = use_lm
        if self.use_lm:
            self.ss_emb_path = os.path.join(data_path, f"ss_embedding/{mode}_ss_{rde_model}_maxfev-10000.pth")
            self.ss_emb = torch.load(self.ss_emb_path, map_location="cpu")

        self.dataset = ALL_DATASETS[dataset_name](
            args=args,
            data_path=data_path,
            mode=mode,
            **kwargs
        )
        if mode == "test":
            text_json_list = [f"text_{dataset_name}_llava-v1.6-vicuna-7b_0.json", f"text_{dataset_name}_llava-v1.6-mistral-7b_0.json"]
        else:
            text_json_list = [f"text_{dataset_name}_llava-v1.6-vicuna-7b_{i}.json" for i in range(6)] + [f"text_{dataset_name}_llava-v1.6-mistral-7b_{i}.json" for i in range(6)]
        self.text_dict_list = []
        for json_path in text_json_list:
            with open(os.path.join(data_path, "text", json_path), 'r') as f:
                self.text_dict_list.append(json.load(f))


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        batch_dict = self.dataset[index]

        # Load image_embedding
        image_path = batch_dict["image_path"]
        idx = random.randint(0, len(self.text_dict_list) - 1)
        text = self.text_dict_list[idx][image_path[:-4]]['text'][0]
        if self.use_lm:
            ss_emb_dict = self.ss_emb[image_path[:-4]]
            gt_shift = ss_emb_dict["gt_shift"].to(torch.float32)
            gt_scale = ss_emb_dict["gt_scale"].to(torch.float32)

        # Update dict
        if self.use_lm:
            update_dict = {
                "text": text,
                "gt_scale": gt_scale,
                "gt_shift": gt_shift,
            }
        else:
            update_dict = {
                "text": text,
            }
        batch_dict.update(update_dict)

        return batch_dict

def Create_MDEDataloader(args, dataname, mode):

    dataset = MDEDataset(args,
                        root="./data",
                        dataset_name=dataname,
                        mode=mode,
                        rde_model=args.rde_model,
                        use_lm=args.use_lm,
                       )

    if mode == "train":
        shuffle = True
        bz = args.batch_size
        nw = args.num_worker
    elif mode == "test":
        shuffle = False
        bz = 1
        nw = 1
    else:
        raise RuntimeError(f"Invalid mode:{mode}")

    dataloader = DataLoader(dataset, batch_size=bz, shuffle=shuffle, num_workers=nw, drop_last=True)

    return dataloader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test_Dataloader")
    # parser.add_argument('--rde_model',            type=str,     default='dav1',    help='Relative Depth Estimator type')
    parser.add_argument('--use_lm',         type=bool,  help='lm loss', default=False)
    args = parser.parse_args()

    rde_model_dict = {"midas-1": "midas_dpt_swin2_large_384",
                      "midas-2": "midas_dpt_large_384",
                      "dav1": "dav1_vits",
                      "dav2": "dav2_vits",
                      }
    test_model = ["midas-1", "midas-2", "dav1", "dav2"]
    train_data = ["nyu", "kitti"]
    test_data = ["nyu", "kitti", "diml", "ddad", "sunrgbd"]

    for mode in ["train", "test"]:
        for rde_model in test_model:
            test_rde_model = rde_model_dict[rde_model]
            for dataname in test_data:
                if mode == "train" and dataname not in train_data:
                    break
                else:
                    test_dataloader = MDEDataset(
                                            args,
                                            root="./data",
                                            dataset_name=dataname,
                                            rde_model=test_rde_model,
                                            mode=mode,
                                            )
                    idx = 0
                    num = len(test_dataloader)
                    for i, batch in enumerate(test_dataloader):
                        idx += 1
                        if idx > 20:
                            idx = 0
                            print(mode, rde_model, dataname, "ok, next", )
                            break

    print("ok")
