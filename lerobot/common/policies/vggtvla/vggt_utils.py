import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import torch.nn.functional as F

def load_and_preprocess_images(tensor_image_list,img_masks, mode="crop"):

    # Check for empty list
    assert len(tensor_image_list) > 0, "Image list must not be empty"
    B = tensor_image_list[0].shape[0]
    S = len(tensor_image_list)
    processed_imgs = []
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for img in tensor_image_list:
        # Open image
        img = img * 2.0 - 1.0

        B, C, H, W = img.shape

        if mode == "crop":
            new_width = target_size
            new_height = round(H * (target_size / W) / 14) * 14

            img_resized = F.interpolate(img, size=(new_height, new_width), mode="bicubic", align_corners=False)

            # Center crop height if needed
            if new_height > target_size:
                top = (new_height - target_size) // 2
                img_resized = img_resized[:, :, top:top+target_size, :]
        elif mode == "pad":
            # Resize while preserving aspect ratio
            if W >= H:
                new_width = target_size
                new_height = round(H * (target_size / W) / 14) * 14
            else:
                new_height = target_size
                new_width = round(W * (target_size / H) / 14) * 14

            img_resized = F.interpolate(img, size=(new_height, new_width), mode="bicubic", align_corners=False)

            # Pad to [target_size, target_size]
            pad_h = target_size - new_height
            pad_w = target_size - new_width
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            img_resized = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        merged_mask = torch.stack(img_masks, dim=0).any(dim=0)  # [2, B] → [B]
        processed_imgs.append(img_resized)  # shape [B, 3, H, W]

    # Stack along S dimension: [S, B, 3, H, W] → [B, S, 3, H, W]
    stacked = torch.stack(processed_imgs, dim=1)
    return stacked, merged_mask