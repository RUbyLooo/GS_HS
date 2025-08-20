import pathlib
from pathlib import Path
import os,sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from torch.utils.data import DataLoader
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset


class MixLeRobotDataset(LeRobotDataset):
    def __init__(self, *lrd_args, name: str, **lrd_kwargs):
        super().__init__(*lrd_args, **lrd_kwargs)
        self.name = name
        print(f"[Init] dataset_name={name} "
                f"scene_pt={scene_pt}, subtask_pt={subtask_pt}")
     


    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # item["scene_des"] = self.scene_pt
        # item["subtask_des"] =  self.subt_pt
        return item 

class MultiMixLeRobotDataset(MultiLeRobotDataset):
    def __init__(self,
                 repo_ids: list[str],
                 name: list[str],
                 root: list[Path],
                 **kwargs):
   
        super().__init__(repo_ids=repo_ids, root=root, **kwargs)
        self.name = name
        for idx, name in enumerate(self.name):
            print(f"[Init] dataset_name={name} "
                f"scene_pt={self.scene_pt_list[idx]}, subtask_pt={self.subtask_pt_list[idx]}")

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        ds_idx = int(item["dataset_index"].item())  # 哪个子数据集
        # item["scene_des"]   = self.scene_embs[ds_idx]
        # item["subtask_des"] = self.subtask_embs[ds_idx]


        return item
