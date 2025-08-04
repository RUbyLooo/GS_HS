from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, pack
from einops.layers.torch import Rearrange
from einops import repeat

from lerobot.common.policies.egovla.laq.laq_model.attention import Transformer, ContinuousPositionBias
from lerobot.common.policies.egovla.laq.laq_model.nsvq import NSVQ
# from lerobot.common.policies.egovla.laq.laq_model.piper_arm import PiperArm
# from scipy.spatial.transform import Rotation as R

def exists(val):
    return val is not None


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


class LatentActionQuantization(nn.Module):
    def __init__(
        self,
        *,
        dim,                        # Transformer 中的特征维度
        quant_dim,                  # 量化嵌入向量维度
        codebook_size,              # 向量量化的码本大小
        image_size,                 # 输入图像大小
        patch_size,                 # 分割成 patch 的大小
        spatial_depth,              # 空间方向 Transformer 层数
        temporal_depth,             # 时间方向 Transformer 层数
        dim_head = 64,
        heads = 8,
        channels = 3,
        attn_dropout = 0.,
        ff_dropout = 0.,
        code_seq_len = 1,           #  量化代码长度
        is_use_action_state = False,
        is_use_fk_state = False,    # 是否使用 FK 状态
        state_dim = 16,         # action state 的维度
        chunksize = 50,
        ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporalNSVQ patch size
        """

        super().__init__()

        self.code_seq_len = code_seq_len
        self.chunksize = chunksize
        self.state_dim    = state_dim
        """
        pair:
            如果 x 已经是一个形如 (height, width) 的 tuple, 就直接返回
            如果 x 是一个 单独的整数 (比如 256), 就将其转换成 (256, 256) 的形式
        """
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)

        patch_height, patch_width = self.patch_size

        # 连续空间相对位置偏置子模块
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        # 保证图像的宽高不为 0
        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        """
        to_patch_emb_first_frame
            输入:
            [b, c, 1, H, W]

            Step 1: Rearrange 切分 patch
            [b, c, 1, H, W]
                => Rearrange (b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2))

                h = H//p1 w = W//p2
            [b, 1, h, w, (c * p1 * p2)]

            Step 2: LayerNorm 对最后一维 [..., 3072] 做 标准化
            [b, 1, h, w, (c * p1 * p2)]

            Step 3: Linear映射
            [b, 1, h, w, dim]

            Step 4: LayerNorm
            [b, 1, h, w, dim]

            输出:
            [b, 1, h, w, dim]
        """
        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )
        

        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )
        
        transformer_with_action_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
            has_cross_attn = True,
            dim_context = dim,
        )

        """ 空间 Transformer """
        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        """ 时间 Transformer """
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)

        """ 向量量化模块 """
        self.vq = NSVQ(
            dim=dim,
            num_embeddings=codebook_size,      # 8
            embedding_dim=quant_dim,           # 32
            device='cuda',
            code_seq_len=code_seq_len,         # 4
            patch_size=patch_size,             # 32
            image_size=image_size              # 256
        )
            
        # 解码模块
        self.dec_spatial_transformer = Transformer(depth = spatial_depth, **transformer_with_action_kwargs)
        # [b, 1, h, w, dim] -> [b, c, 1, h * p1, w * p2]
        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1 = patch_height, p2 = patch_width)
        )

        # 如果使用 action state
        self.is_use_action_state = is_use_action_state
        self.is_use_fk_state = is_use_fk_state
        # if is_use_action_state == True:
        #     self.action_state_proj = nn.Linear(16, 1024)
        if self.is_use_action_state:
            # 先把第一帧的原始 state 投影到和 patch-emb 同维度
            self.action_state_proj = nn.Linear(state_dim, dim)

            self.dim = dim

            # State → token 序列的投影
            self.state_to_tokens = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * code_seq_len),
                nn.ReLU()
            )

            self.state_cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
           
            # —— **新增：FiLM 条件生成器** ——
            # 输入 state_dim，输出 2×dim 的 gamma/beta
            self.film_generator = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(),
            )
            # —— **End FiLM** ——
            # ---- state decoder ----
            self.state_decoder = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, chunksize * state_dim),
                nn.Sigmoid()  # 强制归一化到 [0, 1]
            )

            self.state_fk_decoder = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, chunksize * 14),
                nn.Sigmoid()  # 强制归一化到 [0, 1]
            )

            self.is_use_delta_action_state = True


    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs, strict = False)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        pt = {k.replace('module.', '') if 'module.' in k else k: v for k, v in pt.items()}
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebooks[indices]

        return codes

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens,
        state_token=None
    ):
        # b   : batch size
        b = tokens.shape[0]
        d = tokens.shape[-1]
        # h, w: patch grid 的高度和宽度（比如 8×8）
        h, w = self.patch_height_width
        # video_shape: 记录 [b, t, h, w] 方便后续 reshape 用
        video_shape = tuple(tokens.shape[:-1])
        first_state_token = state_token[:,:self.dim]
        last_state_token = state_token[:,self.dim:]

        state_token = None


        """
        📌 目的：把每一帧当作一个图像处理，送入空间 Transformer
                把每一帧 flatten 成 patch sequence:
                    输入：[b, t, h, w, d] → 输出：[(b×t), (h×w), d]
                举例:
                    如果 b=1, t=2, h=8, w=8, d=1024
                    就会变成 [2, 64, 1024]
        """
        # Film 融合tokens和第一帧的state ： https://arxiv.org/abs/1709.07871 FiLM: Visual Reasoning with a General Conditioning Layer
        if state_token is not None:
            # state_token: [B, D] →  [B, 1, D]
            first_ctx = first_state_token.unsqueeze(1)
            first_vis_flat = rearrange(tokens[:,:1], 'b t h w d -> b (t h w) d')  # [B, L, D], L=h·w·t
            first_fuse, _ = self.state_cross_attn(first_vis_flat, first_ctx, first_ctx)

            last_ctx = last_state_token.unsqueeze(1)
            last_vis_flat = rearrange(tokens[:,1:], 'b t h w d -> b (t h w) d')  # [B, L, D], L=h·w·t
            last_fuse, _ = self.state_cross_attn(last_vis_flat, last_ctx, last_ctx)
            
            first_fuse =  rearrange(first_fuse, 'b (t h w) d -> b t h w d', b=b, h=h, w=w)
            last_fuse =  rearrange(last_fuse, 'b (t h w) d -> b t h w d', b=b, h=h, w=w)

            tokens[:,:1] = first_fuse
            tokens[:,1:] = last_fuse
            
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        # attn_bias shape : torch.Size([16, 64, 64])
        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        # tokens shape : torch.Size([2, 64, 1024])
        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        # tokens shape : torch.Size([1, 2, 8, 8, 1024])
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)
       
        

        # tokens shape : torch.Size([64, 2, 1024])
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        # tokens shape : torch.Size([64, 2, 1024])
        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)
        # tokens shape : torch.Size([1, 2, 8, 8, 1024])
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        # shape: [1, 1, 8, 8, 1024]
        first_tokens = tokens[:, :1]
        # shape: [1, 1, 8, 8, 1024]
        last_tokens = tokens[:, 1:]
        
        return first_tokens, last_tokens

        

    def decode(
        self,
        tokens,
        actions,
        
    ):
        # batchsize
        b = tokens.shape[0]
        # h w
        h, w = self.patch_height_width


        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        # tokens b t h w d
        # video_shape = (b, t, h, w)
        video_shape = tuple(tokens.shape[:-1])


        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        actions = rearrange(actions, 'b t h w d -> (b t) (h w) d')      # [B, code_seq_len, dim]

        # 构建空间相对位置偏置（用于 attention）
        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.dec_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape, context=actions)
        

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w) #[B, T, H, W, D]

        rest_frames_tokens = tokens

        recon_video = self.to_pixels_first_frame(rest_frames_tokens)

        # ——— 2. state 关节角重建 —— #
        state_feat = actions.mean(dim=1)                                       # [B, dim]
        state_flat  = self.state_decoder(state_feat)                           # [B, chunksize*state_dim]
        recon_state = state_flat.view(-1, self.chunksize, self.state_dim)      # [B, chunksize, state_dim]



        # # ——— 2. state fk 重建 —— #
        state_fk_flat = self.state_fk_decoder(state_feat)   
        recon_fk_state = state_fk_flat.view(-1, self.chunksize, 14)            # # [B, chunksize, 14]

        return recon_video, recon_state, recon_fk_state
    

    def compute_state_deltas(self, target_states: torch.Tensor) -> torch.Tensor:
        """
        Computes deltas along the chunk_size (time) dimension.
        
        Args:
            target_states (torch.Tensor): Tensor of shape [B, chunk_size, state_dim]

        Returns:
            torch.Tensor: Delta tensor of same shape [B, chunk_size, state_dim]
        """
        B, T, D = target_states.shape

        # 初始化第一帧为0
        zeros = torch.zeros((B, 1, D), device=target_states.device, dtype=target_states.dtype)

        # 差分（从第2帧开始），沿着 chunk_size 维度
        deltas = target_states[:, 1:, :] - target_states[:, :-1, :]

        # 拼接，使输出 shape 保持一致 [B, T, D]
        delta_states = torch.cat([zeros, deltas], dim=1)

        return delta_states
    

    def forward(
        self,
        video,
        action_state = None,
        fk_result_state = None,
        step = 0,
        mask = None,
        return_recons_only = False,
        return_only_codebook_ids = False,
    ):
        
        """
        video shape: [B, C, 2, 256, 256]
        video.ndim == 5 现在是带有时间维度的
        """
        cond_token_proj = None
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4
        

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)
        
        """
        重新从 video 的 tensor 中读取维度信息
            b = 1                        # batch size
            c = 3                        # channel
            f = 2                        # frame count
            image_dims = [256, 256]      # list, 存储 height 和 width
            device = video.device
        """
        B, f, C,  H, W, device = *video.shape, video.device
        # print(f"b : {B}")
        if (H, W) != self.image_size:
            # 1) 把 (B, C, f, H, W) -> (B*f, C, H, W)
            frames = video.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

            frames = frames.float()  # 映射到 [0, 1]
            # 2) 用 bilinear 插值 resize
            frames = F.interpolate(
                frames,
                size=self.image_size,       # (new_H, new_W)
                mode='bilinear',
                align_corners=False
            )
            # 3) 再 reshape 回 (B, C, f, new_H, new_W)
            video = frames.reshape(B, f, C, *self.image_size).permute(0, 2, 1, 3, 4)
       
        # 确保图片维度一致
        assert (video.shape[3],video.shape[4]) == self.image_size
        # 这里 mask 是 none
        assert not exists(mask) or mask.shape[-1] == f

        """
        first_frame: 第 1 帧         (shape: [b, c, 1, H, W])
        rest_frames: 后面的所有帧     (shape: [b, c, f-1, H, W])
        """
        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]


        # TODO: check dim 
        if self.is_use_action_state and action_state is not None:
            # 取第一步的 state
            first_state = action_state[:, 0, :]                  # [B, state_dim]
            last_state = action_state[:,-1,:]                       # [B, state_dim]
            first_state_token = self.action_state_proj(first_state)  # [B, dim]
            last_state_token = self.action_state_proj(last_state)    # [B, dim]
            state_token  = torch.concat((first_state_token, last_state_token),dim=1)  # [B, 2 * dim]
        

        """
        image_size = (256, 256)
        patch_size = (32, 32)
        ->
        num_patches_per_frame = (256 // 32, 256 // 32) = (8, 8)
        """

        # 第一张图转为 patch token embedding                                             shape: torch.Size([1, 1, 8, 8, 1024])
        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)


        # 剩下图转为 patch token embedding                                               shape: torch.Size([1, 1, 8, 8, 1024])
        rest_frames_tokens = self.to_patch_emb_first_frame(rest_frames)
        # 将第 1 帧和剩余帧的 patch token 沿着 token 维度拼接起来，形成一个统一的 token 序列    shape: torch.Size([1, 2, 8, 8, 1024])
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)
        

        shape = tokens.shape
        # h : 8, w : 8
        *_, h, w, _ = shape

        """
        first_tokens shape : torch.Size([b, 1, 8, 8, 1024])
        last_tokens shape  : torch.Size([b, 1, 8, 8, 1024])
        """
        if self.is_use_action_state == True and action_state is not None:
            first_tokens, last_tokens = self.encode(tokens, state_token)
        else:
            first_tokens, last_tokens = self.encode(tokens)

        """
        pack
            pack 的模式 'b * d'
                'b * d' 表示：

                维度的第一个是 b

                最后一个是 d

                中间的维度全部“摊平”成一个维度 *
            运行后:
                first_tokens shape : torch.Size([b, 64, 1024])
                last_tokens shape  : torch.Size([b, 64, 1024])
        """
        first_tokens, first_packed_fhw_shape = pack([first_tokens], 'b * d')
        last_tokens, last_packed_fhw_shape = pack([last_tokens], 'b * d')

        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)
        self.lookup_free_quantization = False
        vq_kwargs = dict(mask = vq_mask) if not self.lookup_free_quantization else dict()

        # tokens shape : torch.Size([1, 4, 1024]) 这里的 token 代表隐变量
        tokens, perplexity, codebook_usage, indices = self.vq(first_tokens, last_tokens, codebook_training_only = False)
        
        # num_unique_indices : 3
        num_unique_indices = indices.unique().size(0)
      

        
        if ((step % 10 == 0 and step < 100)  or (step % 100 == 0 and step < 1000) or (step % 500 == 0 and step < 5000)) and step != 0:
            print(f"update codebook {step}")
            self.vq.replace_unused_codebooks(tokens.shape[0])

        if return_only_codebook_ids:
            return indices
        
        if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            ## error
            print("code_seq_len should be square number or defined as 2")
            return

        # tokens shape : torch.Size([1, 1, 2, 2, 1024])
        # self.code_seq_len : 4
        # action_h : 2, action_w : 2
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w)
        # concat_tokens shape : torch.Size([1, 1, 8, 8, 1024])
        concat_tokens = first_frame_tokens.detach() # + tokens
        # recon_video shape : torch.Size([1, 3, 1, 256, 256])   将编码 tokens 转换回视频帧的像素空间
        recon_video, recon_state, recon_fk_state = self.decode(concat_tokens, tokens)

        # returned_recon shape : torch.Size([1, 3, 256, 256])
        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w')

        # video shape : torch.Size([1, 3, 1, 256, 256])
        video = rest_frames
        
        if return_recons_only:
            return returned_recon, recon_state
        

        """
        这里暂时没有 mask
        """
        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.mse_loss(video, recon_video, reduction = 'none')
            recon_loss = recon_loss[repeat(mask, 'b t -> b c t', c = C)]
            recon_img_loss = recon_loss.mean()
        else:
            recon_img_loss = F.mse_loss(video, recon_video)

        if self.is_use_action_state:
            target_states = action_state[:, 1:1 + self.chunksize, :]
            if self.is_use_delta_action_state:
                target_states = self.compute_state_deltas(target_states)
            print(f"target_states shape : {target_states.shape}")
            recon_state_loss = F.mse_loss(recon_state, target_states)
        else:
            recon_state_loss = torch.tensor(0.0, device=video.device)


        if self.is_use_fk_state and fk_result_state is not None:
            target_fk = fk_result_state[:, 1:1 + self.chunksize, :]
            recon_fk_state_loss = F.mse_loss(recon_fk_state, target_fk)
        else:
            recon_fk_loss = torch.tensor(0.0, device=video.device)
        

        # recon_loss = recon_img_loss + 0.5 * recon_state_loss + 0.08 * recon_fk_state_loss
        recon_loss = recon_img_loss


        return recon_img_loss, recon_state_loss, recon_fk_state_loss, recon_loss, num_unique_indices
        

    def inference(
        self,
        video,
        step = 0,
        mask = None,
        return_only_codebook_ids=False,
        user_action_token_num=None
    ):
        
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb_first_frame(rest_frames)
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)


        shape = tokens.shape
        *_, h, w, _ = shape

        first_tokens, last_tokens = self.encode(tokens)

        # quantize
        first_tokens, first_packed_fhw_shape = pack([first_tokens], 'b * d')
        last_tokens, last_packed_fhw_shape = pack([last_tokens], 'b * d')

        if user_action_token_num is not None:
            tokens, indices = self.vq.inference(first_tokens, last_tokens, user_action_token_num=user_action_token_num)
        else:
            tokens, indices = self.vq.inference(first_tokens, last_tokens)

        
    
        if return_only_codebook_ids:
            return indices

        if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            print("code_seq_len should be square number or defined as 2")
            return
        

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w)
        concat_tokens = first_frame_tokens #.detach() #+ tokens
        recon_video = self.decode(concat_tokens, actions=tokens)
        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w')
        video = rest_frames 
        
        return returned_recon

