#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from collections import deque

import torch
from torch import nn


def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
):
    """
    👉 将当前 batch 的输入张量添加到一个滑动窗口队列(deque)中, 用于构建时间序列输入, 如 state、images、language_tokens 等，供模型预测时使用
    Args:
        参数名	                类型	                    作用
        queues	               dict[str, deque]	          每个 key (例如 state) 对应一个有长度限制的 deque (双端队列)
        batch	               dict[str, torch.Tensor]	  当前时间步的观测 batch (例如 {"state": tensor(...), "images": ...})
        exclude_keys	       list[str]	              None
    """
    if exclude_keys is None:
        exclude_keys = []

    # step 1 : 遍历当前输入的 batch 的所有 keys
    """
    (1) 遍历所有键 (如 "state", "observation.image", "language_tokens")
    (2) 对每个观测键判断是否需要加入对应的队列
    """
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        """
        如果这个 key :
            不在 queues 里  :     就不管（比如 queues 只存想要的字段）
            被显式排除      :      也不处理（如 "action" 是模型输出，不需要做滑动窗口）
        """
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module (nn.Module): a PyTorch module
        input_shape (tuple): A tuple representing the input shape, e.g., (batch_size, channels, height, width)

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)
