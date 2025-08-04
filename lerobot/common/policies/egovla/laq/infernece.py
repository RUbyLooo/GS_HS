import torch
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse

from lerobot.common.policies.egovla.laq.laq_model import LatentActionQuantization



def load_image(img_path1, img_path2):
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)
    return img1, img2

def inference():



    codebook_size = 8
    spatial_depth = 8
    temporal_depth = 8
    code_seq_len=4
    laq_checkpoint = "/home/cfy/cfy/smovlm/LAPA/laq/results/vae.100000.pt"


    laq = LatentActionQuantization(
        dim=1024,
        quant_dim=32,
        codebook_size=codebook_size,
        image_size=256,
        patch_size=32,
        spatial_depth=spatial_depth,
        temporal_depth=temporal_depth,
        dim_head=64,
        heads=16,
        code_seq_len=code_seq_len,
    ).cuda()
    laq.load(laq_checkpoint)

    print(f"laq load success !!! ")

    img_first_path = "/home/cfy/data/icra_data/data/fold_cloth/images/observation.images.top/episode_000038/frame_000211.png"
    img_last_path = "/home/cfy/data/icra_data/data/fold_cloth/images/observation.images.top/episode_000038/frame_000261.png"

    img_first, img_last = load_image(img_first_path, img_last_path)

    image_size = 256
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])

    img_first_tensor = transform(img_first)  # shape: [3, 256, 256]
    img_last_tensor = transform(img_last)    # shape: [3, 256, 256]

    # 在 dim=1 上插入“时间轴”维度
    img_first_tensor = img_first_tensor.unsqueeze(1)  # [3, 1, 256, 256]
    img_last_tensor = img_last_tensor.unsqueeze(1)    # [3, 1, 256, 256]

    # 拼接两个时间帧 → 结果是 [3, 2, 256, 256]
    image_pair_tensor = torch.cat([img_first_tensor, img_last_tensor], dim=1)
    # 增加上 batch 维度
    image_pair_tensor = image_pair_tensor.unsqueeze(0)  # [1, 3, 2, 256, 256]


    with torch.no_grad():
        index_batch, fir_img_concat = laq(image_pair_tensor.cuda(), return_only_codebook_ids=True)

    print(f"index_batch : {index_batch}")

    codes_token, recon_img = laq.decode_from_codebook_indices(index_batch, fir_img_concat)
    print(f"codes shape : {codes_token.shape}")




if __name__ == "__main__":
    inference()