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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
python lerobot/scripts/train.py \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
python lerobot/scripts/train.py \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

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
from lerobot.common.policies.egovla.configuration_egovla import EgoVLAConfig
from lerobot.common.policies.egovla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.common.policies.utils import (
    populate_queues,
)
from lerobot.common.utils.utils import get_safe_dtype
from lerobot.common.policies.egovla.laq.laq_model import LatentActionQuantization

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


class EgoVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class =EgoVLAConfig
    name = "egovla"

    def __init__(
        self,
        config:EgoVLAConfig,
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
        self.model = VLAFlowMatching(config)
        self.reset()

        # —— 模块化加载 LAQ/VQVAE 教师模型 —— #
        self.laq = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.laq_supervision:
            self.laq = LatentActionQuantization(**config.laq_model_kwargs).to(self.device)
            self.laq.load(config.laq_ckpt)
            self.laq.eval()

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
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )
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
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}

        deltas = self.config.observation_delta_indices
        t0_idx   = deltas.index(0)
        tN_idx   = deltas.index(self.config.laq_temporal_offset) \
                    if (self.config.laq_supervision and self.config.laq_temporal_offset in deltas) \
                    else None
        
        # —— 模式1: 仅 SmolVLM 微调 —— #
        if self.config.train_smolvlm_only and self.config.laq_supervision:
            total_laq = torch.tensor(0.0, device=self.device)
            for cam in self.config.laq_cameras:
                seq = batch[cam]  # [B, T, C, H, W]
                # 拿出两个时间步
                img0 = seq[:, t0_idx]    # 当前帧
                imgN = seq[:, tN_idx]    # +offset 帧
                pair = torch.stack([img0, imgN], dim=2).to(self.device)  # [B,C,2,H,W]

                with torch.no_grad():
                    index_batch, fir_img_concat = self.laq(
                        pair, return_only_codebook_ids=True
                    )  # [B, chunk_size，query_seq_length, latent_dim] ([B,N,L,D])
                    codes_token, recon_img = self.laq.decode_from_codebook_indices(index_batch, fir_img_concat)
                B, CS, L, D = codes_token.shape # CS for chunksize
                N = CS * L
                decoder_input_ids = torch.full((B, N), fill_value=self.language_tokenizer.bos_token_id, device=self.device)
                decoder_attn_mask = torch.ones_like(decoder_input_ids)
                pred = self.model.vlm_with_expert.forward_latents(pair, decoder_input_ids, decoder_attn_mask)
                # pred: [B, T, L, D]
                loss_latent = F.mse_loss(pred, codes_token)

            loss_laq = loss_latent / len(self.config.laq_cameras)
            loss = self.config.laq_loss_weight * loss_laq
            return loss, {"loss_laq": loss_laq.item(), "loss": loss.item()}


         # ——— 模式2/E2E diffusion ———
        images, img_masks       = self.prepare_images(batch)
        state                   = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions                 = self.prepare_action(batch)
        actions_is_pad          = batch.get("actions_id_pad")

        # diffusion 主 loss
        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks,
            state, actions, noise, time
        )
        loss_dict["losses_after_forward"] = losses.clone()
        if actions_is_pad is not None:
            mask   = ~actions_is_pad
            losses = losses * mask.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()
        loss_main = losses[:, :, : self.config.max_action_dim].mean()
        loss_dict["loss_main"] = loss_main.item()

        # ——— LAQ token 监督 —— #
        loss_laq = torch.tensor(0.0, device=self.device)
        if self.config.laq_supervision:
            total_laq = torch.tensor(0.0, device=self.device)
            for cam in self.config.laq_cameras:
                seq = batch[cam]            # [B, T, C, H, W]
                img0 = seq[:, t0_idx]       # 当前帧
                imgN = seq[:, tN_idx]       # +offset 帧
                pair = torch.stack([img0, imgN], dim=2).to(self.device)  # [B, C, 2, H, W]

                with torch.no_grad():
                    index_batch, fir_img_concat = self.laq(
                        pair, return_only_codebook_ids=True
                    )  # [B, chunk_size，query_seq_length, latent_dim] ([B,N,L,D])
                    codes_token, recon_img = self.laq.decode_from_codebook_indices(index_batch, fir_img_concat)
                B, CS, L, D = codes_token.shape # CS for chunksize
                N = CS * L
                decoder_input_ids = torch.full((B, N), fill_value=self.language_tokenizer.bos_token_id, device=self.device)
                decoder_attn_mask = torch.ones_like(decoder_input_ids)
                pred = self.model.vlm_with_expert.forward_latents(pair, decoder_input_ids, decoder_attn_mask)
                # pred: [B, T, L, D]
                loss_latent = F.mse_loss(pred, codes_token)


            loss_laq = loss_latent / len(self.config.laq_cameras)
            loss_dict["loss_laq"] = loss_laq.item()

        # —— 总 loss —— #
        loss = loss_main + self.config.laq_loss_weight * loss_laq
        loss_dict["loss"] = loss.item()
        return loss, loss_dict


    def prepare_images(self, batch):
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
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

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
            # img shape : torch.Size([B, 3, 512, 512])
            images.append(img)
            # img mask shape : torch.Size([B]) tensor([True, True, True, True, True, True, True, True], device='cuda:0')
            img_masks.append(mask)
            print(f"img mask : {mask}")

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        # 步骤 1：获取设备信息
        device = batch[OBS_STATE].device
        # 步骤 2：复制任务指令到整个 batch
        tasks = batch["task"]
        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]
        # 步骤 3：强制添加换行符
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
        # 步骤 4：调用 Tokenizer
        """
        使用指定的 tokenizer 对任务描述进行编码。
            设置参数：
                padding: 是否填充到固定长度（由 pad_language_to 控制）。
                padding_side: 右侧填充（不影响语义）。
                max_length: 最大 token 数。
                return_tensors="pt": 返回 PyTorch 张量。
        """
        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        # 步骤 5：提取 token IDs 和 attention mask
        """
        input_ids: token ID 序列
        attention_mask: 指示哪些位置是真实内容（非填充）
        lang_tokens: Tensor[B, L]，表示 token ID 序列
        lang_masks: Tensor[B, L]，布尔型，表示 attention mask
        """
        """
        lang_tokens : tensor([[ 7177,   274, 11244,   198],
            [ 7177,   274, 11244,   198],
            [ 7177,   274, 11244,   198],
            [ 7177,   274, 11244,   198],
            [ 7177,   274, 11244,   198],
            [ 7177,   274, 11244,   198],
            [ 7177,   274, 11244,   198],
            [ 7177,   274, 11244,   198]], device='cuda:0')
        """
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        """
        langlang_masks : tensor([[True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True]], device='cuda:0')
        """
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

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
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
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


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            train_smolvlm_only=self.config.train_smolvla_only,
            lora_smolvlm=self.config.lora_smolvlm,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

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
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """

        """
        �� 功能概述：
            将图像、语言文本和状态信息嵌入到共享的语义空间中，并拼接成一个完整的 prefix 序列，作为扩散模型去噪过程的上下文输入。
        Args:
            参数名	            类型	           描述
            images	           List[Tensor]	     图像特征列表，可能包含多个摄像头视角
            img_masks	       List[Tensor]	     对应每个图像的有效掩码
            lang_tokens	       Tensor[B, L]	     语言 token IDs
            lang_masks	       Tensor[B, L]	     语言 attention mask
            state	           Tensor[B, D]	     状态向量（机器人关节角度）
        """
        # Step 1: 初始化容器
        """
        embs       : 存储各模态的嵌入向量
        pad_masks  : 存储 padding 掩码（哪些位置是有效的）
        att_masks  : 存储 attention 掩码（控制哪些 token 可以相互关注）
        """
        embs = []
        pad_masks = []
        att_masks = []
        # Step 2: 处理每张图像

        # 🔁 遍历图像和图像掩码
        """
        1) 遍历 images 和 img_masks, 通过 enumerate 获取索引
        2) zip (images, img_masks, strict=False) 是 Python 3.10+ 的用法, strict=False 意味着允许其中一个序列更长，不会报错
        3) 每个 img 是一个批量图像, img_mask 是图像的有效区域掩码。

        """
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            ## 添加图像起始特殊 token（optional）
            # 如果启用了 add_image_special_tokens，则在每张图像前插入一个特殊的开始 token，作为多模态输入的分界
            """
            🌟 将全局图像开始 token 送入 embed_language_tokens 方法进行嵌入，得到其向量表示
            🌟 unsqueeze(0)                 : 在 batch 维度前添加一维
            🌟 expand(img.shape[0], -1, -1) : 将 token 扩展成整个 batch (每个样本一个)
            """
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                # 创建与 image_start_token 相匹配的 mask, 值全为 1（即有效位置）
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                # 将 image start token 的 attention mask 填入（注意，这里是 0，可能是后续用非零值区分语言 token)
                att_masks += [0] * (image_start_mask.shape[-1])
                # 把图像开始 token 和其掩码加入序列
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
            ## 图像嵌入（使用 SigLIP）
            img_emb = self.vlm_with_expert.embed_image(img) # [B, 64, 960]
            img_emb = img_emb

            # Normalize image embeddings
            ## 归一化图像嵌入
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            ## 构造图像掩码, 将布尔掩码扩展为与图像嵌入维度一致
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs) #[B,64]

            # img_emb shape : torch.Size([B, 64, 960])
            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            ## 添加图像 end token（可选）
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])

        # Step 3: 处理语言文本
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        ## 归一化语言嵌入, 同样做缩放，保持与图像嵌入在同一尺度
        lang_emb_dim = lang_emb.shape[-1]
        # lang_emb shape : torch.Size([B, 4, 960])
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # Step 4: 处理状态信息
        state_emb = self.state_proj(state)
        # state_emb shape : torch.Size([8, 1, 960])
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1) #[4, 135,960]
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
        action_time_emb = self.action_time_mlp_out(action_time_emb) # [B, S, D]

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
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions


        # TODO： mask??
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        """
        �� 功能概述：
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
        bsize = state.shape[0]
        device = state.device
        # Step 2: 初始化噪声（如果未提供）
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        # Step 3: 嵌入前缀信息
        """
        将图像、语言、状态等信息编码为统一的嵌入向量
        输出包括：
            prefix_embs: 嵌入张量 [B, L, D]
            prefix_pad_masks: padding 掩码 [B, L]
            prefix_att_masks: attention 掩码 [B, L]
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        # Step 4: 构造二维 attention mask
        """ 构建标准 Transformer 所需的二维 attention mask [B, 1, L]，用于屏蔽无效位置 """
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        # Step 5: 构造 position_ids
        """
        生成位置索引(position_ids), 用于标识每个 token 的位置
        因为 pad_mask 是布尔值, cumsum 可以得到非填充位置的顺序编号
        """
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        # Step 6: 前向传播 VLM 获取缓存（KV Cache）
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        # Step 7: 设置扩散时间步长
        """
        时间步长 dt 为负数，因为时间是从 1.0 向 0.0 流动
        num_steps 控制去噪的迭代次数
        """
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        # Step 8: 初始化变量 x_t 和 time
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        # Step 9: 去噪循环
        """
        循环直到时间小于 -dt/2(即接近 0)
        每一步：
            调用 denoise_step() 计算当前噪声梯度方向 v_t
            使用欧拉方法更新 x_t(相当于去噪一步)
        """
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """
        �� 功能概述：
            给定当前噪声 x_t 和时间步 timestep, 利用 VLM 模型预测去噪方向 v_t
            该函数是扩散模型中的核心组件之一, 负责每一步的噪声估计
        Args:
            参数名	             类型	                描述
            prefix_pad_masks	Tensor[B, L]	      prefix 的 padding mask
            past_key_values	    Tuple	              已经计算好的 key-value 缓存
            x_t	                Tensor[B, S, D]	      当前噪声状态
            timestep	        Tensor[B]	          当前时间步（广播后的）
        """
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # Step 1: 嵌入后缀信息
        """ 将当前噪声 x_t 和时间步 timestep 编码为 suffix embeddings """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
        # Step 2: 获取各种长度信息
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        # Step 3: 构造 prefix 的扩展 mask
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
