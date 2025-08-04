import os
import io
import base64
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np

class ParquetTrajectoryDataset(Dataset):
    def __init__(self, folder, image_size=256, offset=5, input_mode="top_cam_only"):
        super().__init__()
        self.folder = folder
        self.parquet_files = sorted([
            f for f in os.listdir(folder) if f.endswith('.parquet')
        ])
        self.image_size = image_size
        self.offset = offset

        # 解析 input_mode
        supported_modes = ["top_cam_only", "all_cam_only", "all_cam_state", "top_cam_state"]
        assert input_mode in supported_modes, f"Unsupported mode: {input_mode}"
        self.input_mode = input_mode

        # 设置字段
        self.image_keys = []
        self.include_state = False

        if input_mode in ["top_cam_only", "top_cam_state"]:
            self.image_keys = ['observation.images.top']
        elif input_mode in ["all_cam_only", "all_cam_state"]:
            self.image_keys = [
                'observation.images.top',
                'observation.images.wrist_left',
                'observation.images.wrist_right'
            ]

        if input_mode in ["all_cam_state", "top_cam_state"]:
            self.include_state = True

        self.load_columns = self.image_keys.copy()
        if self.include_state:
            self.load_columns.append('observation.state')

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.parquet_files)

    def _load_image(self, img_data):
        try:
            if isinstance(img_data, dict) and 'bytes' in img_data:
                with Image.open(io.BytesIO(img_data['bytes'])) as img:
                    return img.convert('RGB').copy()
            elif isinstance(img_data, str):
                if os.path.exists(img_data):
                    with Image.open(img_data) as img:
                        return img.convert('RGB').copy()
                else:
                    with Image.open(io.BytesIO(base64.b64decode(img_data))) as img:
                        return img.convert('RGB').copy()
            elif isinstance(img_data, bytes):
                with Image.open(io.BytesIO(img_data)) as img:
                    return img.convert('RGB').copy()
            else:
                raise ValueError("Unsupported image format")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")

    def __getitem__(self, index):
        tries = 0
        max_tries = 3

        while tries < max_tries:
            try:
                parquet_path = os.path.join(self.folder, self.parquet_files[index])
                df = pd.read_parquet(parquet_path, columns=self.load_columns)
                num_frames = len(df)

                if num_frames < 2:
                    raise ValueError(f"Too few frames in {parquet_path}")

                first_idx = random.randint(0, num_frames - 2)
                second_idx = min(first_idx + self.offset, num_frames - 1)

                images_first = []
                images_second = []

                for key in self.image_keys:
                    img1_data = df.iloc[first_idx][key]
                    img2_data = df.iloc[second_idx][key]
                    img1 = self._load_image(img1_data)
                    img2 = self._load_image(img2_data)
                    img1_t = self.transform(img1).unsqueeze(1)  # [C,1,H,W]
                    img2_t = self.transform(img2).unsqueeze(1)
                    images_first.append(img1_t)
                    images_second.append(img2_t)

                # 拼接图像
                all_imgs_first = torch.cat(images_first, dim=0)  # [C*n_views, 1, H, W]
                all_imgs_second = torch.cat(images_second, dim=0)
                cat_img = torch.cat([all_imgs_first, all_imgs_second], dim=1)  # [C*n_views, 2, H, W]

                # 加上状态信息（Optional）
                if self.include_state:
                    # 获取 offset + 1 个状态（从 first_idx 到 second_idx，含）
                    states = df.iloc[first_idx:second_idx + 1]['observation.state'].values

                    # 转为单个 tensor，形状为 [offset + 1, state_dim]
                    state_tensor = torch.tensor(np.stack(states), dtype=torch.float32)

                    return cat_img, state_tensor
                else:
                    return cat_img

            except Exception as e:
                print(f"[WARN] Failed loading index {index}: {e}")
                index = random.randint(0, len(self.parquet_files) - 1)
                tries += 1

        raise RuntimeError("Failed to load data after multiple retries.")
