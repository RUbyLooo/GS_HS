#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoProcessor

from lerobot.common.constants import ACTION, OBS_STATE
from lerobot.common.policies.normalize import (
    Normalize,
    Unnormalize,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.qwenvla.configuration_qwenvla import QwenVLAConfig
# from lerobot.common.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.common.policies.utils import (
    populate_queues,
)
from lerobot.common.utils.utils import get_safe_dtype
from vggt.models.aggregator import Aggregator
from vggt.models.vggt import VGGT
from lerobot.common.policies.vggtvla.vggt_utils import load_and_preprocess_images
from typing import Tuple
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import logging

from torchvision.transforms import ToPILImage

def extract_all_batch_images_and_captions(image_dict: dict, max_views: int = 4):
    """
    从图像字典中提取 batch 中所有样本的多视角图像和对应 caption。
    
    参数:
        image_dict: dict，键是摄像头名（如 observation.image.front），值是 [B, T, C, H, W] tensor
        max_views: 每个样本最多取多少个视角

    返回:
        - image_lists: List[List[PIL.Image]]，batch_size 个 image list
        - caption_lists: List[List[str]]，batch_size 个 caption list
    """
    to_pil = ToPILImage()
    batch_size = next(iter(image_dict.values())).shape[0]

    image_lists = [[] for _ in range(batch_size)]
    caption_lists = [[] for _ in range(batch_size)]

    view_count = 0

    for key in sorted(image_dict.keys()):
        img_tensor = image_dict[key]  # [B, T, C, H, W]
        if img_tensor.ndim != 5:
            print("!!!!!!!!!!!!!!!!!!!!")
            continue

        for b in range(batch_size):
            img = img_tensor[b, -1]  # 取每个样本最后一帧
            img = img.to(torch.uint8) if img.dtype == torch.int64 else img
            img = img.clamp(0, 1).cpu()
            pil_img = to_pil(img)

            image_lists[b].append(pil_img)
            caption_lists[b].append(f"camera view: {key.split('.')[-1]}")

        view_count += 1
        if view_count >= max_views:
            break

    return image_lists, caption_lists  # [B][N], [B][N]


def _resize_images(images, size=(224, 224)):
    return [img.resize(size) for img in images]
# def combine_multi_view_images(image_dict: dict, batch_idx=0, max_views=4):
  
#     to_pil = ToPILImage()
#     image_list = []
#     captions = []
#     count = 0

#     for key in sorted(image_dict.keys()):
#         img_tensor = image_dict[key]
#         if img_tensor.ndim == 5:
#             img_tensor = img_tensor[batch_idx, -1]  # 取最后一帧
#             img_tensor = img_tensor.to(torch.uint8) if img_tensor.dtype == torch.int64 else img_tensor

        
#             img_tensor = img_tensor.clamp(0, 1).cpu()
#             pil_image = to_pil(img_tensor)
#             image_list.append(pil_image)
#             captions.append(f"camera view: {key.split('.')[-1]}")  # e.g., "front", "left_wrist", etc.

#         count += 1
#         if count >= max_views:
#             break

#     return image_list, captions


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


import threading

class LangCacheManager:
    def __init__(self, vlm, device):
        self.vlm = vlm
        self.device = device
        self.lang_cache = None
        self.task = None
        self.loop = asyncio.new_event_loop()
        t = threading.Thread(target=self.loop.run_forever, daemon=True)
        t.start()
        self.future: asyncio.Future | None = None
        # self.ready = asyncio.Event()

    def get_cache(self):
        return self.lang_cache

    async def _run_vlm(self, image_list, captions, goal):
        logging.info("VLM running")
        with torch.no_grad():
            lang_emb = self.vlm.get_batch_prefix_embeddings(image_list, captions, goal)  # [1, L, D]
            # lang = self.vlm.
        self.lang_cache = lang_emb.detach().to(self.device)
            # self.ready.set()

    def launch(self, image_list, captions, goal):
        # if self.task is None or self.task.done():
        if self.future is None or self.future.done():
            # self.ready.clear()
            # try:
            #     loop = asyncio.get_running_loop()
            # except RuntimeError:
            #     loop = asyncio.new_event_loop()
            #     asyncio.set_event_loop(loop)

            # self.task = loop.create_task(self._run_vlm(image_list, captions, goal))
            self.future = asyncio.run_coroutine_threadsafe(
                self._run_vlm(image_list, captions, goal),
                self.loop
            )

            # self.task = asyncio.create_task(self._run_vlm(image_list, captions, goal))
    def is_ready(self) -> bool:
        return self.future is not None and self.future.done()

    def wait(self, timeout=None):
        """阻塞直到最新一次的 VLM 推理结束，抛异常会往外抛。"""
        if self.future is not None:
            # .result() 会阻塞直到完成
            self.future.result(timeout)



from termcolor import colored
from transformers import AutoModelForVision2Seq
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor

class LangReducer(nn.Module):
    def __init__(self, dim_model, num_latents=5, n_heads=4):
        super().__init__()
        # 5 个可训练的查询
        self.latent_queries = nn.Parameter(torch.randn(num_latents, dim_model))
        self.cross_attn = nn.MultiheadAttention(dim_model, n_heads, batch_first=True)

    def forward(self, lang_embs):
        # lang_embs: [B, L, D]
        B, L, D = lang_embs.shape
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 5, D]
        # 把 lang_embs 当作 KV，用 queries 作为 Q
        out, _ = self.cross_attn(query=queries, key=lang_embs, value=lang_embs)
        return out  # [B, 5, D]






class QwenVLM:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
        logging.info(colored("initializing Qwen-VL",attrs=["bold"]))

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            quantization_config=quant_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        # self.model = AutoModelForVision2Seq.from_pretrained(model_name, quantization_config=quant_config,trust_remote_code=True).to(device)
        self.device = device

    def _build_message_content(self, images: list[Image.Image], image_captions: list[str], prompt: str):
        content = []
        for i, (img, caption) in enumerate(zip(images, image_captions)):
            content.append({"type": "text", "text": f"{caption}"})
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        return content

    @torch.no_grad()
    def generate_task_description(self, images: list[Image.Image], image_captions: list[str], goal: str, max_new_tokens: int = 128) -> str:
        """
        输入图像和最终任务目标，输出格式化 token 描述：[目标任务]... [细分任务]...[当前场景]... [任务进度]...
        """
        # 构造多模态输入：图像 + prompt
        # prompt = (
        #     f"请根据图像内容(包括第三视角和腕部相机）和任务目标，请尤其关注腕部相机图片中和目标的空间关系，保证任务顺利完成，生成一个任务状态描述，格式为："
        #     f"[目标任务]{goal}[当前场景]...[细分任务]... [任务进度]..."
        #     f"细分任务的示例："
        #     f"1.接近目标"
        #     f"2.打开夹爪"
        #     f"3.抓取物体"
        #     f"任务进度需要和你的细分任务相关"
        #     f"细分任务数量为1-5"
            
        # )
        prompt = (
            f"根据图像内容(包括第三视角和腕部相机）和目标任务，生成一个低层级的任务规划，格式为："
            f"[目标任务]{goal}\n"
            f"[低层级任务]:...\n"
            f"[目前在执行的低层级任务]:...\n"
            f"[下一个低层级任务]:..."
            
        )
       
        
 
        content = self._build_message_content(images, image_captions, prompt)
        
        messages = [{"role": "user", "content": content}]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=images, return_tensors="pt").to(self.model.device)
        # inputs.update({"images": image})

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,remove_invalid_values=True,)
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text

    @torch.no_grad()
    def get_prefix_embedding(self, images: list[Image.Image],image_captions: list[str], goal: str) -> torch.Tensor:
        """
        获取图像 + 任务目标组合生成的 prefix token embedding（多模态 embedding）
        output: [1, L, D]
        """
        # 文本 prompt 和图像同样构建
        # prompt = (
        #     f"请根据图像内容(包括第三视角和腕部相机）和任务目标，请尤其关注腕部相机图片中和目标的空间关系，保证任务顺利完成，生成一个任务状态描述，格式为："
        #     f"[目标任务]{goal}[当前场景]...[细分任务]... [任务进度]..."
        #     f"细分任务的示例："
        #     f"1.接近目标"
        #     f"2.打开夹爪"
        #     f"3.抓取物体"
        #     f"任务进度需要和你的细分任务相关"
        #     f"细分任务数量为1-5"
            
        # )
        prompt = (
            f"根据图像内容(包括第三视角和腕部相机）和目标任务，生成一个低层级的任务规划，格式为："
            f"[目标任务]{goal}\n"
            f"[低层级任务]:...\n"
            f"[目前在执行的低层级任务]:...\n"
            f"[下一个低层级任务]:..."
            
        )

        # logging.info(colored("run vlm for 1 episode"))
        content = self._build_message_content(images, image_captions, prompt)

        messages = [{"role": "user", "content": content}]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(text=text_prompt, images=images, return_tensors="pt").to(self.model.device)

        # forward 获取 hidden states
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return outputs.hidden_states[-1]  # [1, L, D]

    def extract_prefix(self, image: Image.Image, goal: str, verbose=False) -> torch.Tensor:
        """
        高层封装：输入图像和目标 → 输出 prefix embedding: [1, L, D]
        """
        text = self.generate_task_description(image, goal)
        if verbose:
            print("[Generated Text]:", text)
        return self.get_prefix_embedding(text)
    
    from typing import List
    @torch.no_grad()
    def get_batch_prefix_embeddings(self, image_lists: List[List[Image.Image]], captions:List[str], goal_texts: List[str]) -> torch.Tensor:
        """
        给定 batch 中每个样本的多视角图像和目标文本，逐个调用 VLM，返回拼接好的 [B, L, D] 特征。
        """
        # logging.info(colored("run vlm for 1 batch"))
        all_embs = []
        for images,caption, goal in zip(image_lists, captions, goal_texts):
            emb = self.get_prefix_embedding(images=_resize_images(images), image_captions=caption, goal=goal)  # [1, L, D]
            all_embs.append(emb)
            # desc = self.generate_task_description(
            #     images=_resize_images(images),
            #     image_captions=caption,
            #     goal=goal,
            # )
            # print(f"\n=== Generated for goal '{goal}' ===\n{desc}\n")

        # 对不同 L 做 padding，使得 [B, L_max, D]
        max_len = max(e.shape[1] for e in all_embs)
        dim = all_embs[0].shape[2]

        padded_embs = []
        for e in all_embs:
            pad_len = max_len - e.shape[1]
            if pad_len > 0:
                pad = torch.zeros((1, pad_len, dim), dtype=e.dtype, device=e.device)
                e = torch.cat([e, pad], dim=1)
            padded_embs.append(e)

        return torch.cat(padded_embs, dim=0)  # [B, L_max, D]

class KVCacheDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory_kv, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt: [B, S, D]
        # memory_kv: (memory_k, memory_v) each [B, L, D]
        memory_k, memory_v = memory_kv

        # Self Attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # Cross Attention
        tgt2, _ = self.cross_attn(tgt, memory_k, memory_v)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # Feedforward
        tgt2 = self.linear2(F.gelu(self.linear1(tgt)))
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt
    

class KVCacheTransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            KVCacheDecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])

    def forward(self, tgt, memory_kv, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt: [B, S, D]
        # memory_kv: (memory_k, memory_v): each [B, L, D]

        for layer in self.layers:
            tgt = layer(tgt, memory_kv, tgt_mask, tgt_key_padding_mask)
        return tgt

class ExpertTransformer(nn.Module):
    def __init__(self, action_dim:int, dim_model: int, num_layers: int, n_heads: int):
        super().__init__()
        self.decoder = KVCacheTransformerDecoder(
            num_layers=num_layers,
            d_model=dim_model,
            nhead=n_heads,
            dropout=0.1,
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2 * dim_model, dim_model),
            nn.SiLU(),
            nn.Linear(dim_model, dim_model),
        )

        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)

        self.out_proj = nn.Linear(dim_model, action_dim)

    def build_kv_cache(self, memory: torch.Tensor):
        # memory: [B, L, D]
        return self.k_proj(memory), self.v_proj(memory)

    def forward(self, 
                prefix_kv_cache: tuple[torch.Tensor, torch.Tensor],  # (k,v) 
                x_t: torch.Tensor, 
                time_emb: torch.Tensor
                ):
        """
        Args:
            prefix_embs: [B, L1, D]
            x_t: [B, S, D]
            time_emb: [B, S, D]
        Returns:
            v_t: [B, S, D]
        """
        suffix_input = torch.cat([x_t, time_emb], dim=-1).float()  
        suffix_input = self.fuse_mlp(suffix_input)
        out = self.decoder(tgt=suffix_input, memory_kv=prefix_kv_cache)
       

        return out  





class QwenVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = QwenVLAConfig
    name = "qwenvla"

    def __init__(
        self,
        config: QwenVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoProcessor.from_pretrained(self.config.vlm_model_name).tokenizer
        self.model = QwenVLA(config)
        
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        # 设置模型为评估模式
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        batch = self.normalize_inputs(batch)

        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            for k in batch:
                if k in self._queues:
                    batch[k] = torch.stack(list(self._queues[k]), dim=1)
            # images, img_masks = self.prepare_images(batch)
            images, img_masks = self.prepare_images_vggt(batch)
            image_for_vggt, merged_masks = load_and_preprocess_images(images, img_masks)
            state = self.prepare_state(batch)
            # lang_tokens, lang_masks = self.prepare_language(batch)
            lang_mgr = self.model.lang_cache_mgr
            lang_cache = lang_mgr.get_cache()

            image_dict = {k: v for k, v in batch.items() if k.startswith("observation.images")}
            image_lists, caption_lists = extract_all_batch_images_and_captions(image_dict, max_views=4)  # 返回 List[PIL.Image]
            
            goal_texts=batch["task"]


            if lang_cache is None:
            # 第一次：启动并等待
                if lang_mgr.future is None:
                    print("[SLOW LOOP] (first): launching VLM")
                    lang_mgr.launch(image_lists, caption_lists, goal_texts)
                print("[SLOW LOOP] (first): waiting for VLM to finish")
                lang_mgr.wait()
                lang_cache = lang_mgr.get_cache()
                assert lang_cache is not None, "第一次 VLM 推理失败"
            else:
                # 后续：如果上一次任务已经完成，就再异步启动一次，不等待
                if lang_mgr.future is not None and lang_mgr.future.done():
                    print("[SLOW LOOP] (refresh): re-launching VLM in background")
                    lang_mgr.launch(image_lists, caption_lists, goal_texts)

            print("[FAST LOOP] sampling actions using current lang_cache")

            lang_random = torch.randn_like(lang_cache)

            # actions_random = self.model.sample_actions(
            #     image_for_vggt, merged_masks, lang_embeds = lang_random , state = state, noise=noise
            # )


            actions = self.model.sample_actions(
                image_for_vggt, merged_masks, lang_embeds = lang_cache.to(state.device) , state = state, noise=noise
            )
            print(f"[FAST LOOP] got actions tensor with shape {actions.shape}")
            # l2_diff = (actions - actions_random).norm(p=2, dim=-1).mean().item()
            # cos_sim = F.cosine_similarity(actions, actions_random, dim=-1).mean().item()
            # print(f"[DEBUG] L2 diff: {l2_diff:.4f}, Cosine similarity: {cos_sim:.4f}")


            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images_vggt(batch)
        image_for_vggt, merged_masks = load_and_preprocess_images(images, img_masks)

    
        state = self.prepare_state(batch)
        # lang_tokens, lang_masks = self.prepare_language(batch)


        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        lang_embs = batch["lang_embs"]
        assert lang_embs is not None, "Missing lang_embs in fast path! Ensure slow loop has called VLM."
        # if self.model.lang_cache_mgr.task is None or self.model.lang_cache_mgr.task.done():
        #     image_dict = {k: v for k, v in batch.items() if k.startswith("observation.image")}
        #     # current_image_list, current_images_captions = extract_all_batch_images_and_captions(image_dict, batch_idx=0)
        #     # current_goal = batch["task"]
        #     lang_embs = self.model.lang_cache_mgr.get_cache()
        #     assert lang_embs is not None, "lang_cache not ready! Please launch() in slow loop."
        #     # self.model.lang_cache_mgr.launch(current_image_list, current_images_captions, current_goal)


        losses = self.model.forward(image_for_vggt, merged_masks, lang_embs, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For backward pass
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images_vggt(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        """
        �� 功能概述：
        该函数的作用是对输入的图像数据进行标准化和预处理，包括：
            调整尺寸（保持宽高比）
            填充(padding)
            归一化像素值到 [-1, 1] 范围
            处理缺失图像通道（模拟空摄像头）
        Args:
            batch: 输入的数据字典，例如：
                {
                    "observation.image": Tensor[B, C, H, W],
                    "observation.image_padding_mask": Tensor[B]
                }
        最终输出是：
            图像张量列表 images
            对应的掩码列表 img_masks(表示哪些图像是真实的, 哪些是填充的)
        """
        images = []
        img_masks = []
        # 步骤 1：收集存在的图像键与缺失的图像键
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        # 步骤 2：处理每个存在的图像模态
        """
        如果图像是时间序列(shape 为 [B, T, C, H, W]），只取最后一帧（最新观测）。
        否则直接使用原始图像
        """
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            ## 尺寸调整（保持比例 + 填充）
            # if self.config.resize_imgs_with_padding is not None:
            #     img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            ## 归一化图像范围 [0,1] → [-1,1]
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            ## 构建图像掩码
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        # images = torch.stack(images, dim=0).permute(1, 0, 2, 3, 4)
        return images, img_masks


    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.dim_model)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor

class PreceiverResample(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_latents: int, num_layers:int, n_heads: int, num_obs:int, dropout=0.1):
        super().__init__()
        self.latent_tokens = nn.Parameter(torch.randn(num_latents, latent_dim))  # [N_latent, D]
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim * 4),
                nn.GELU(),
                nn.Linear(latent_dim * 4, latent_dim)
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)
        if input_dim != latent_dim:
            self.input_proj = nn.Linear(input_dim, latent_dim)
        else:
            self.input_proj = nn.Identity()
        self.num_time = num_obs
        self.time_embed = nn.Embedding(num_obs, latent_dim)  # 可学习时间编码



    def forward(self, x: torch.Tensor):
        """
        x: [S, B, C]  # e.g., image tokens, S=T*H*W
        return: [N_latent, B, latent_dim]
        """
        x = self.input_proj(x)  # [S, B, D]
        assert x.shape[0] % self.num_time == 0, "Token数不能被时间步整除，可能 flatten 顺序有问题(in resample)"

        S = x.shape[0]
        device = x.device
        P = S // self.num_time
        B = x.shape[1]
        t_ids = torch.arange(self.num_time, device=device).repeat_interleave(P)  # [S]
        t_embeds = self.time_embed(t_ids).unsqueeze(1)  # [S, 1, D]
        x = x + t_embeds

        latents = self.latent_tokens.unsqueeze(1).expand(-1, B, -1)  # [N_latent, B, D]
        
        kv = torch.cat([latents, x], dim=0)
        for attn, ffn in zip(self.cross_attn, self.ffn):
            attn_out, _ = attn(query=latents, key=kv, value=kv)  # [N_latent, B, D]
            latents = latents + attn_out
            latents = latents + ffn(self.norm(latents))

        return latents
    
class LangImageFusion(nn.Module):
    def __init__(self, dim_model: int, lang_dim:int, n_heads:int, dropout:float):
        super().__init__()
        self.scene_to_image_proj = nn.Linear(lang_dim, dim_model)
        self.lang_proj = nn.Linear(lang_dim, dim_model)

        # subtask to image attn
        self.lang_img_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout )
        self.img_tolang__attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=n_heads, dropout=dropout)
       
        self.lang_norm = nn.LayerNorm(dim_model)
       

    def forward(self, lang_tensor: torch.Tensor, img_features:torch.Tensor):
        """
        Args:
            subtask_tensor: [N, subtask_embed_dim]
            img_features: [B, 5, 960]
        Returns:
            fused_token: [1, B, D]  # scene description embedding with image feature
            subtask_for_decoder: [B, N, D]   # each subtask token for action decoder
        """
        B, T_img, D_img = img_features.shape
        
       
        lang_feat = lang_tensor.permute(1,0,2)  # [ N, B, D]
        # expand for attn (Q: [Tq, B, D], K/V: [Tv, B, D])

        

        # scene ↔ image
        # image_tokens = img_features.permute(2, 0, 1)   # [5, 4, D]
        image_tokens = img_features.permute(1,0,2) # [5, 4, D]
        lang_w_image_feat = self.lang_img_attn(query=lang_feat, key=image_tokens, value=image_tokens)[0]
        lang_w_image_feat = self.lang_norm(lang_w_image_feat + lang_feat)

        
        return lang_w_image_feat.permute(1,0,2)


class QwenVLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # TODO 这里维度需要检查
        self.vlm_backbone = QwenVLM(model_name="Qwen/Qwen2-VL-7B-Instruct", device="cuda")
        self.lang_emb_cache = None
        self.expert_transformer = ExpertTransformer(action_dim=config.max_action_dim, dim_model=config.dim_model, num_layers=8 , n_heads=8)
        self.lang_cache_mgr = LangCacheManager(self.vlm_backbone, device=torch.device("cuda"))
        self.lang_proj = nn.Linear(self.lang_cache_mgr.vlm.model.config.hidden_size, self.config.text_embed_dim)

        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.config.text_embed_dim
        )
        self.state_proj_embed = nn.Linear(
            self.config.text_embed_dim, self.config.dim_model
        )
        self.lang_token_proj = nn.Linear(  self.config.text_embed_dim, self. config.dim_model)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.expert_embed_dim)
        self.action_out_proj = nn.Linear(self.config.expert_embed_dim, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.config.expert_embed_dim * 2, self.config.expert_embed_dim
        )
        self.action_time_mlp_out = nn.Linear(
            self.config.expert_embed_dim, self.config.expert_embed_dim
        )
        self.lang_resampler = LangReducer(dim_model=self.config.text_embed_dim, num_latents=5)
        self.set_requires_grad()


        self.prefix_length = self.config.prefix_length
        self.vggt_url = config.vggt_url
        self.vggt_img_size = config.vggt_img_size
        self.vggt_patch_size= config.vggt_patch_size
        self.vggt_embed_dim = config.vggt_embed_dim
        self.load_aggregator()
        for param in self.aggregator.parameters():
            param.requires_grad = False
        self.PreceiverResample = PreceiverResample(input_dim = 2 * config.vggt_embed_dim, 
                                                   latent_dim=config.dim_model, 
                                                   num_latents=5, 
                                                   num_layers=8, 
                                                   n_heads=8,
                                                   num_obs=config.n_obs_steps)
        self.LangImageFusion = LangImageFusion(
            dim_model = self.config.dim_model,
            lang_dim=self.lang_cache_mgr.vlm.model.config.hidden_size,
            # fusion_dim=config.resample_dim_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        
        for param in self.vlm_backbone.model.parameters():
            param.requires_grad = False


    def load_aggregator(self):
        vggt_url = self.vggt_url
        self.aggregator = Aggregator(img_size=self.vggt_img_size, patch_size=self.vggt_patch_size, embed_dim=self.vggt_embed_dim)
        state_dict = torch.hub.load_state_dict_from_url(
           vggt_url
        )
        self.aggregator.load_state_dict(state_dict, strict=False)
        # print(msg)  # 显示 missing_keys 和 unexpected_keys


    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, 
        images:torch.Tensor,
        img_masks:torch.Tensor,
        lang_embeds:torch.Tensor,
        state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        embs = []
        pad_masks = []
        att_masks = []
        # Step 1: 处理每张图像

        # vggt aggregator
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        tokens = aggregated_tokens_list[-1]  # [B, S, P, 2C]
        patch_tokens = tokens[:, :, patch_start_idx:, :]  # [B, S, N_patch, 2C]
        B, S, N_patch, C = patch_tokens.shape
        flattened = patch_tokens.contiguous().view(B, S * N_patch, C) # [B, S * N_patch, C]
        flattened = flattened.permute(1, 0, 2)  # [S*N_patch, B, C]
        image_tokens = self.PreceiverResample(flattened)  # [5, B, 64]
        image_tokens = image_tokens.permute(1, 0, 2)  # → [B, 5, 64]
        # image_tokens =self.image_token_proj(image_tokens)

        embs.append(image_tokens)
        bsize, num_img_embs = image_tokens.shape[:2]
        image_mask = img_masks[:, None].expand(-1, num_img_embs)  # [B, 5]

        pad_masks.append(image_mask)
        att_masks += [0] * (num_img_embs)


        lang_embeds = lang_embeds
        if lang_embeds is not None:
            lang_emb =  lang_embeds.expand(B, -1, -1)  # [B, L, D]
            lang_emb = lang_emb.to(dtype=torch.float32)
            lang_emb = self.lang_proj(lang_emb)
        else:
            lang_emb = torch.zeros((B, 6, self.config.text_embed_dim), device=image_tokens.device)  # fallback

        lang_emb = self.lang_resampler(lang_emb)
        lang_emb = self.lang_token_proj(lang_emb)
        #lang_emd 融合
        embs.append(lang_emb)
        fused_lang_emb = self.LangImageFusion(lang_emb, image_tokens)
        embs.append(fused_lang_emb)

       
        # Step 4: 处理状态信息
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        state_emb = self.state_proj_embed(state_emb)
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""

        """
        �� 功能概述：
            将当前的动作噪声和时间步融合编码，生成 suffix embeddings, 作为扩散模型去噪步骤中的动态输入
        Args:
            参数名	                  类型	                  描述
            noisy_actions	         Tensor[B, S, A]	    当前噪声动作序列(batch_size x chunk_size x action_dim)
            timestep	             Tensor[B]	            当前时间步（广播后）
        """
        # Step 1: 初始化容器
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        # Step 2: 动作嵌入
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # Step 3: 时间嵌入
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)
        ## 扩展时间嵌入并与动作合并
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_embeds, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions # add noise
        u_t = noise - actions


        prefix_embs, _,_ = self.embed_prefix(
            images, img_masks, lang_embeds, state=state
        )
        memory_k, memory_v = self.expert_transformer.build_kv_cache(prefix_embs)
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.config.dim_model,
            self.config.min_period,
            self.config.max_period,
            device=actions.device,
        )
        time_emb = time_emb[:, None, :].expand_as(x_t)
    
        v_t = self.expert_transformer(prefix_kv_cache = (memory_k, memory_v), 
                                        x_t = x_t,
                                        time_emb = time_emb)
        print(f"x_t shape: {x_t.shape}")

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_embeds, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        """
        功能概述：
            使用扩散模型从初始噪声开始逐步去噪，最终输出一个动作序列。
            该函数会：
                嵌入图像、语言和状态信息
                构建注意力掩码和位置 ID
                调用 VLM 模型获取 prefix 的 key-value 缓存
                在每个时间步调用 denoise_step 去噪
                最终返回动作张量

        Args:
            参数名	            类型	           描述
            images	           List[Tensor]	     图像特征列表，可能来自多个摄像头
            img_masks	       List[Tensor]	     图像掩码，表示哪些图像是有效的
            lang_tokens	       Tensor[B, L]	     语言 token IDs
            lang_masks	       Tensor[B, L]	     语言 attention mask
            state	           Tensor[B, D]	     状态向量（关节角度）
            noise	           Optional[Tensor]	 初始噪声张量
        """
        # Step 1: 获取 batch size 和 device
        B = state.shape[0]
        device = state.device
        # Step 2: 初始化噪声（如果未提供）
        if noise is None:
            # actions_shape = (B, self.config.chunk_size, self.config.max_action_dim)
            actions_shape = (B, self.config.chunk_size, self.config.resample_dim_model)
            noise = self.sample_noise(actions_shape, device)
        # Step 3: 嵌入前缀信息
        """
        将图像、语言、状态等信息编码为统一的嵌入向量
        输出包括：
            prefix_embs: 嵌入张量 [B, L, D]
            prefix_pad_masks: padding 掩码 [B, L]
            prefix_att_masks: attention 掩码 [B, L]
        """

        # language random




        prefix_embs, _,_ = self.embed_prefix(
            images, img_masks, lang_embeds, state=state
        )
        memory_k, memory_v = self.expert_transformer.build_kv_cache(prefix_embs)
       
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        # Step 8: 初始化变量 x_t 和 time
        x_t = noise
        t = torch.tensor(1.0, dtype=torch.float32, device=device)
        # Step 9: 去噪循环
        """
        循环直到时间小于 -dt/2(即接近 0)
        每一步：
            调用 denoise_step() 计算当前噪声梯度方向 v_t
            使用欧拉方法更新 x_t(相当于去噪一步)
        """
        while t >= -dt / 2:
            # 构造 time embedding
            t_expand = t.expand(B)
            time_emb = create_sinusoidal_pos_embedding(
                t_expand,
                self.config.resample_dim_model,
                self.config.min_period,
                self.config.max_period,
                device=device,
            )  # [B, D_model]
            time_emb = time_emb[:, None, :].expand_as(x_t)  # [B, S, D_model]

            # 预测 v_t
            v_t = self.expert_transformer(prefix_kv_cache = (memory_k, memory_v),
                                           x_t = x_t, 
                                           time_emb = time_emb)  # [B, S, D_model]

            # Euler 更新
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t  # [B, S, A]
      