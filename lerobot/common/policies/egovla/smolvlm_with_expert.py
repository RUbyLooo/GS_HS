# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import copy
from typing import List, Optional

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model

def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    """
    Function:
        用于计算 Transformer 模型中前馈神经网络 (FFN) 层的中间维度 (intermediate size)
        该函数首先将输入维度缩小为原来的 2/3, 再乘以一个扩展倍数 (如 4) 最后将结果向上取整为
        `multiple_of` 的倍数, 以提升在硬件 (GPU) 上的执行效率

    Args:
        hidden_dim (int): 
            输入的隐藏维度，通常是 Transformer 中的 embedding size
        ffn_dim_multiplier (int, optional): 
            用于扩展隐藏维度的乘法系数。默认为 4
        multiple_of (int, optional): 
            输出维度将被向上取整为该值的倍数, 目的是对齐硬件特性以提高计算性能, 默认为 256

    Returns:
        int: 计算后的中间层维度, 可用于构建 FFN 模块

    Example:
        >>> get_intermediate_size(480)
        1280
    """
    # step 1 : 对输入维度做一次压缩                  hidden_dim = int(2 * 480 / 3) = 320
    hidden_dim = int(2 * hidden_dim / 3)
    # step 2 : 将压缩后的维度按一定倍数扩展,          hidden_dim = int(4 * 320) = 1280
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    # step 3 : 向上取整到 nearest multiple 的倍数,  hidden_dim = 1280
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = False,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        train_smolvlm_only: bool = False,
        lora_smolvlm:bool = False,    # 是否使用 LoRA
        lora_r:int  = 8,
        lora_alpha:float= 16,
        lora_dropout:float= 0.05,
        laq_code_seq_length:int=8, 
        laq_latent_dim:int=1024,
        chunk_size:int=50,

    ):
        super().__init__()
        if load_vlm_weights:
            print(f"Loading  {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)          # 16
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size                                 # 维度是 960
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # 480
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier)) # 1280
        lm_expert_config.num_hidden_layers = self.num_vlm_layers # 16
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)
        """
        lm_expert : 
        LlamaConfig {
            "_flash_attn_2_enabled": true,  # 启用FlashAttention-2优化, 加速注意力计算并减少内存占用
            "architectures": [
                "VLlama3ForCausalLM"        # 模型架构为视觉-语言LLaMA 3的因果语言模型变体
            ],
            "attention_bias": false,        # 注意力机制中不使用偏置项
            "attention_dropout": 0.0,       # 注意力层不应用dropout
            "bos_token_id": 1,              # 开始标记(BOS)的ID为1
            "eos_token_id": 2,              # 结束标记(EOS)的ID为2
            "head_dim": 64,                 # 每个注意力头的维度大小
            "hidden_act": "silu",           # 隐藏层激活函数使用SiLU (Swish)
            "hidden_size": 720,             # 隐藏层维度大小
            "initializer_range": 0.02,      # 权重初始化的范围
            "intermediate_size": 2048,      # MLP层中间维度大小
            "is_llama_config": true,        # 标识为LLaMA系列模型配置
            "max_position_embeddings": 8192, # 最大位置编码, 支持8192长度的序列
            "mlp_bias": false,              # MLP层不使用偏置项
            "model_type": "llama",          # 模型类型为LLaMA
            "neftune_noise_alpha": 0.0,     # 不启用NEFTune噪声正则化
            "num_attention_heads": 15,      # 注意力头的数量
            "num_hidden_layers": 16,        # 隐藏层(Transformer块)的数量
            "num_key_value_heads": 5,       # 使用Grouped Query Attention, 每组5个头
            "pad_token_id": 2,              # 填充标记(PAD)的ID为2
            "perceiver_config": {           # 视觉感知器相关配置
                "_attn_implementation_autoset": false,
                "_name_or_path": "",
                "add_cross_attention": false,  # 不添加交叉注意力
                "architectures": null,
                "attention_dropout": 0.0,
                "bad_words_ids": null,
                "begin_suppress_tokens": null,
                "bos_token_id": null,
                "chunk_size_feed_forward": 0,
                "cross_attention_hidden_size": null,
                "decoder_start_token_id": null,
                "diversity_penalty": 0.0,
                "do_sample": false,
                "early_stopping": false,
                "encoder_no_repeat_ngram_size": 0,
                "eos_token_id": null,
                "exponential_decay_length_penalty": null,
                "finetuning_task": null,
                "forced_bos_token_id": null,
                "forced_eos_token_id": null,
                "hidden_act": "silu",
                "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
                },
                "is_decoder": false,
                "is_encoder_decoder": false,
                "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
                },
                "length_penalty": 1.0,
                "max_length": 20,
                "min_length": 0,
                "model_type": "vllama3",
                "no_repeat_ngram_size": 0,
                "num_beam_groups": 1,
                "num_beams": 1,
                "num_key_value_heads": 1,     # 感知器配置中的KV头数量
                "num_return_sequences": 1,
                "output_attentions": false,
                "output_hidden_states": false,
                "output_scores": false,
                "pad_token_id": null,
                "prefix": null,
                "problem_type": null,
                "pruned_heads": {},
                "qk_layer_norms_perceiver": false,
                "remove_invalid_values": false,
                "repetition_penalty": 1.0,
                "resampler_depth": 6,         # 视觉重采样器的层数
                "resampler_head_dim": 96,     # 视觉重采样器的头维度
                "resampler_n_heads": 16,      # 视觉重采样器的头数量
                "resampler_n_latents": 64,    # 视觉重采样器的潜在变量数量
                "return_dict": true,
                "return_dict_in_generate": false,
                "sep_token_id": null,
                "suppress_tokens": null,
                "task_specific_params": null,
                "temperature": 1.0,
                "tf_legacy_loss": false,
                "tie_encoder_decoder": false,
                "tie_word_embeddings": true,
                "tokenizer_class": null,
                "top_k": 50,
                "top_p": 1.0,
                "torch_dtype": null,
                "torchscript": false,
                "transformers_version": "4.46.0",
                "typical_p": 1.0,
                "use_bfloat16": false
            },
            "pixel_shuffle_factor": 4,      # 像素洗牌因子，用于处理视觉输入
            "pretraining_tp": 1,            # 预训练时不使用张量并行
            "qk_layer_norms": false,        # 不使用查询和键的层归一化
            "rms_norm_eps": 1e-05,          # RMSNorm的epsilon值
            "rope_interleaved": false,      # RoPE位置编码不使用交错排列
            "rope_scaling": null,           # 不使用RoPE缩放
            "rope_theta": 100000,           # RoPE的theta值, 控制位置编码的周期
            "tie_word_embeddings": false,   # 不共享输入和输出词嵌入
            "torch_dtype": "bfloat16",      # 使用bfloat16数据类型进行计算
            "transformers.js_config": {     # Transformers.js的配置
                "kv_cache_dtype": {
                "fp16": "float16",
                "q4f16": "float16"
                }
            },
            "transformers_version": "4.50.3", # 模型使用的Transformers库版本
            "use_cache": true,                # 启用KV缓存以加速生成
            "use_resampler": false,           # 不使用视觉重采样器
            "vocab_size": 49280               # 词表大小为49,280个token
            }
        """

        self.num_expert_layers = len(self.lm_expert.layers)      # 16
        self.self_attn_every_n_layers = self_attn_every_n_layers # 2
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            # 遍历语言模型的每一层
            for layer_idx in range(len(self.lm_expert.layers)):
                # 如果当前层索引是self_attn_every_n_layers的倍数（例如 0, 2, 4...），则跳过这一层。这些层将保留原始的自注意力机制
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                # 对于需要修改的层，重新定义其键投影层（k_proj）。输入维度使用文本配置中的头数和头维度的乘积，输出维度使用语言模型专家配置中的对应值。这样做是为了使语言模型的键投影维度与视觉语言模型（VLM）兼容
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                # 同样地，重新定义值投影层（v_proj），使其输入输出维度与键投影层一致，确保模型能够正确处理交叉注意力计算
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads # 15
        self.num_key_value_heads = self.config.text_config.num_key_value_heads # 5


        # laq 参数
        self.chunk_size = chunk_size
        self.laq_code_seq_length = laq_code_seq_length
        self.laq_latent_dim = laq_latent_dim
        self.latent_head = nn.Linear(
            self.config.text_config.hidden_size,
            self.laq_latent_dim  
        )

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.train_smolvlm_only = train_smolvlm_only
        self.lora_smolvlm = lora_smolvlm
        if self.lora_smolvlm:
            lora_cfg = LoraConfig(
                r = lora_r,
                lora_alpha = lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj","o_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            # 我只对smolvlm lora ， expert始终全量
            self.vlm = get_peft_model(self.vlm, lora_cfg)

        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size
        self.set_requires_grad()

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        # —— 模式 A：只训练 SmolVLM（冻结 Expert） —— #
        if self.train_smolvlm_only:
            # 先冻结 VLM 的所有参数
            for p in self.vlm.parameters():
                p.requires_grad = False
            # 再根据是否用 LoRA，解冻对应参数
            if self.lora_smolvlm:
               
                for n, p in self.vlm.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = True
            else:
                # 全量微调 VLM
                for p in self.vlm.parameters():
                    p.requires_grad = True
            # 冻结 Expert（动作头）
            for p in self.lm_expert.parameters():
                p.requires_grad = False
            # 可选：冻结视觉 encoder
            if self.freeze_vision_encoder:
                vision = self.get_vlm_model().vision_model
                vision.eval()
                for p in vision.parameters():
                    p.requires_grad = False
            return

        # 如果冻结 vlm vision 层 
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        # 如果只训练动作头
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False

       
        # To avoid unused params issue with distributed training
        # "lm_head" 通常指的是语言模型（Language Model）输出层，将隐藏状态映射为词汇表上的概率分布的线性层
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)
    

    def forward_latents(
        self,
        pixel_values: torch.FloatTensor,   # [B, C, 2, H, W]
        input_ids:     torch.LongTensor,    # [B, N]
        attention_mask:torch.LongTensor,    # [B, N]
    ) -> torch.FloatTensor:
        """
        给定与 LAQ 教师相同的 pixel_values 和 decoder 输入 (全 BOS/全 PAD)，
        输出 [B, N, C, D] 的预测 latents，用于 MSE/CE 监督。
        """   
        out = self.get_vlm_model().text_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]              # [B, N, H]
        B,N,H = last_hidden.shape
        assert N == self.chunk_size * self.laq_code_seq_length
        flat = self.latent_head(last_hidden) # [B, N=chunksize * L, D]
        latents = flat.view(B, self.chunk_size, self.laq_code_seq_length, self.laq_latent_dim) # [B, chunksize, L, D]

        return latents
    """
    💡 注意力流程图

              Q (query_states)          K (key_states)         V (value_states)
          ┌──────────────┐         ┌──────────────┐       ┌──────────────┐
          │ (B, L, H, D) │         │ (B, L, H, D) │       │ (B, L, H, D) │
          └─────┬────────┘         └─────┬────────┘       └─────┬────────┘
                │                          │                      │
         transpose(1,2)             transpose(1,2)               │
                ↓                          ↓                      ↓
         (B, H, L, D)                (B, H, L, D)             (B, L, H, D)
                │                          │                      │
                └── Q × Kᵀ = att_weights   │                      │
                           ↓              │                      │
                    (B, H, L, L)          │                      │
                           ↓              │                      ↓
                 softmax(masked)          │             V.permute(0,2,1,3) → (B, H, L, D)
                           ↓              │                      ↓
                probs = attention scores  └─────── matmul ───────┘
                           ↓
                      (B, H, L, L)
                           ↓
               att_output = probs @ V  →  (B, H, L, D)
                           ↓
             reshape → (B, L, H*D) = final att_output ✅

    where:
        B: batch_size
        L: token 数量 (即 total_seq_len)
        H: 注意力头数 = num_attention_heads
        D: 每个头的维度 = head_dim
        hidden_dim = H × D
    """
    def forward_attn_layer(
        self,
        model_layers,                         # 模型的层集合（可能包含多个模型的层，视觉文本/动作专家）
        inputs_embeds,                        # 输入嵌入列表（每个元素是一组模态的嵌入向量，prefix/suffix）
        layer_idx,                            # 当前处理的层索引（在16层中定位到第n层）
        position_ids,                         # 位置编码ID（用于RoPE位置编码）
        attention_mask,                       # 注意力掩码（控制哪些位置可被关注）
        batch_size,                           # 批次大小
        head_dim,                             # 每个注意力头的维度（如64）
        use_cache: bool = True,               # 是否启用KV缓存（生成式任务加速）
        fill_kv_cache: bool = True,           # 是否填充KV缓存（首次计算为True，后续拼接为False）
        past_key_values=None,                 # 历史KV缓存（存储之前步骤的key/value状态）
    ) -> list[torch.Tensor]:                  # 返回注意力输出和更新后的KV缓存

        query_states = []                     # 存储所有输入模态的查询（Query）状态
        key_states = []                       # 存储所有输入模态的键（Key）状态
        value_states = []                     # 存储所有输入模态的值（Value）状态

        """
        inputs_embeds 是一个 list
            inputs_embeds[0]: prefix     [B, L1, 960]
            inputs_embeds[1]: suffix     [B, L2, 720]
        """
        for i, hidden_states in enumerate(inputs_embeds):
            # 获取对应模态在当前层（layer_idx）的Transformer层
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue

            # 对输入进行层归一化（自注意力前的预处理）
            hidden_states = layer.input_layernorm(hidden_states)

            # 计算隐藏状态的形状：(batch_size, seq_len, ...) → 保留前两维（batch和序列长度）
            input_shape = hidden_states.shape[:-1]
            # 重塑形状为 (batch_size, seq_len, num_heads, head_dim)，用于后续注意力头拆分 head_dim 都是 64

            # hidden_shape : (B, L, -1, 64)
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            # 确保输入与投影层权重的数据类型一致（如bfloat16）
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)

            # 通过线性投影层计算Q、K、V，并重塑为注意力头格式
            """
            vlm:
                Q: (B=8, L=197, H=15, D=64)  → 总维度 960
                K: (B=8, L=197, H=5,  D=64)  → 总维度 320
                V: (B=8, L=197, H=5,  D=64)  → 总维度 320
            lm_expert:
                query_state shape  : torch.Size([8, 50, 15, 64])
                key_state shape    : torch.Size([8, 50, 5, 64])
                value_state shape  : torch.Size([8, 50, 5, 64])
            """
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)



            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens

        # 拼接所有模态的 Q/K/V → 在序列长度维度拼接，而不是 head 数维度
        query_states = torch.cat(query_states, dim=1)   # (B, L_total, H_q, D)
        key_states = torch.cat(key_states, dim=1)       # (B, L_total, H_k, D)
        value_states = torch.cat(value_states, dim=1)   # (B, L_total, H_k, D)
        # 获取拼接后的总序列长度
        seq_len = query_states.shape[1]
        # 调整位置编码和注意力掩码的长度（确保与拼接后的序列长度匹配）
        if seq_len < position_ids.shape[1]:
            # 若拼接后序列更短，截断位置编码和掩码
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            # 否则使用原始位置编码和掩码
            _position_ids = position_ids
            _attention_mask = attention_mask

        # 重命名变量
        attention_mask_ = _attention_mask
        position_ids_ = _position_ids
        # 对Q和K应用旋转位置编码（RoPE），注入位置信息
        query_states = apply_rope(query_states, position_ids_)
        key_states = apply_rope(key_states, position_ids_)

        # 初始化KV缓存（若启用缓存且缓存为空）
        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                # 首次计算：将当前层的K、V存入缓存
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # 后续生成步骤：将新的K、V与历史缓存拼接（如 autoregressive 生成时累加序列）
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        # 获取注意力计算接口
        attention_interface = self.get_attention_interface()

        # 调用注意力接口计算注意力输出
        att_output = attention_interface(
            attention_mask_,  # 注意力掩码
            batch_size,       # 批次大小
            head_dim,         # 头维度
            query_states,     # (B, Lq, Hq, D)
            key_states,       # (B, Lk, Hk, D)
            value_states      # (B, Lv, Hk, D)
        )
        # 返回注意力输出（列表形式，便于后续处理）和更新后的KV缓存
        return [att_output], past_key_values



    """
    Prefix 模态 (模态1):
        Q_prefix (B, L1, H1, D)         K_prefix (B, L1, H1, D)        V_prefix (B, L1, H1, D)
            ┌──────────────┐             ┌──────────────┐             ┌──────────────┐
            │              │             │              │             │              │
            │  input_layernorm             │  input_layernorm             │  input_layernorm
            │  + q_proj                    │  + k_proj                    │  + v_proj
            └───────┬─────────────────────┘ └───────┬─────────────────────┘ └───────┬─────────────────────┘
                    │                            │                            │
                transpose(1,2)                transpose(1,2)                (no transpose)
                    ↓                            ↓                            ↓
                (B, H1, L1, D)                (B, H1, L1, D)                (B, L1, H1, D)

                Q_prefix × K_prefixᵀ  → att_weights_prefix
                    ↓
                (B, H1, L1, L1)
                    ↓
                softmax(masked)
                    ↓
                att_output_prefix = att_weights_prefix @ V_prefix
                    ↓
                (B, H1, L1, D)
                    ↓
                reshape → (B, L1, H1*D)


    Expert 模态 (模态2):
        Q_expert (B, L2, H2, D)         K_expert (B, L1, H2, D)         V_expert (B, L1, H2, D)
            ┌──────────────┐             ┌──────────────┐             ┌──────────────┐
            │  input_layernorm             │ key_states from prefix     │ value_states from prefix
            │  + q_proj (expert)           │ + expert k_proj           │ + expert v_proj
            └───────┬─────────────────────┘ └───────┬─────────────────────┘ └───────┬─────────────────────┘
                    │                            │                            │
                transpose(1,2)                transpose(1,2)                (no transpose)
                    ↓                            ↓                            ↓
                (B, H2, L2, D)                (B, H2, L1, D)                (B, L1, H2, D)

                Q_expert × K_expertᵀ  → att_weights_expert
                    ↓
                (B, H2, L2, L1)
                    ↓
                softmax(masked)
                    ↓
                att_output_expert = att_weights_expert @ V_expert
                    ↓
                (B, H2, L2, D)
                    ↓
                reshape → (B, L2, H2*D)

    """

    
    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        attention_interface = self.get_attention_interface()

        att_outputs = []
        assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None and not fill_kv_cache), (
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        )

        if len(inputs_embeds) == 2 and not past_key_values:
            # 场景1：有两个输入模态且无历史缓存（首次计算）
            # Prefix attention

            # 分割位置编码和注意力掩码（inputs_embeds[0]是前缀，inputs_embeds[1]是专家）
            seq_len = inputs_embeds[0].shape[1]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            # 获取第一层（前缀层）并处理
            layer = model_layers[0][layer_idx]

            # 层归一化和QKV投影
            hidden_states = layer.input_layernorm(inputs_embeds[0])

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)

            # 计算QKV并应用RoPE位置编码
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            # 计算自注意力（前缀内部的注意力）
            att_output = attention_interface(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        else:
            # 场景2：使用缓存或仅有一个输入模态
            expert_position_id = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2], -1
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )  # k_proj should have same dim as kv

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )  # start from 0
            expert_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :
            ]  # take into account kv

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        # att_output = att_output.to(dtype=models[i].dtype)
        return att_outputs, past_key_values

    def get_model_layers(self, models: list) -> list:
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]




    """
    🎯 合并总结：
        | Layer | Self / Cross | Expert 存在？ | 执行函数                     | 说明                                          
        | ----- | ------------ | ----------   | --------------------------- | ----------------------------------------        
        | 0     | Self         | ✅           | forward\_attn\_layer        | vlm和expert都做self-attn, 各自独立                 
        | 1     | Cross        | ✅           | forward\_cross\_attn\_layer | expert用vlm的kv计算cross-attn (query来自expert)   
        | 2     | Self         | ✅           | forward\_attn\_layer        | 同上                                             
        | 3     | Cross        | ✅           | forward\_cross\_attn\_layer | 同上                                             
        | ...   | ...          | ...          | ...                         | ...                                              
        | 14    | Self         | ✅           | forward\_attn\_layer        | 同上                                       
        | 15    | Cross        | ✅           | forward\_cross\_attn\_layer | 同上                                       

    """
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        """
        🔍 含义：
        获取多个模型的文本部分组成的列表：一个来自 VLM (Vision-Language Model), 另一个是 LM Expert (动作专家)
        调用 get_model_layers() 方法获取每个 Transformer 层，形成嵌套结构：
            model_layers = [
                [layer_0_model_0, layer_1_model_0, ..., layer_n_model_0],
                [layer_0_model_1, layer_1_model_1, ..., layer_n_model_1],
            ]
        """
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        # 获取 batchsize
        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # RMSNorm
        # vlm backbone 的层数
        num_layers = self.num_vlm_layers                        # num_layers = 16
        head_dim = self.vlm.config.text_config.head_dim         # head_dim = 64
        """
        Layer 0:  self-attn
        Layer 1:  cross-attn
        Layer 2:  self-attn
        Layer 3:  cross-attn
        ...
        Layer 15: cross-attn
        """
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                print
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )  # in case of self_attn
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        attention_interface = self.eager_attention_forward
        return attention_interface

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
