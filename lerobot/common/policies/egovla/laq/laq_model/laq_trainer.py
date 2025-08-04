from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
import wandb

from beartype import beartype

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torch.nn.functional as F
from lerobot.common.policies.egovla.laq.laq_model.optimizer import get_optimizer
from lerobot.common.policies.egovla.configuration_egovla import EgoVLAConfig

from ema_pytorch import EMA
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.policies.egovla.laq.laq_model.data import ParquetTrajectoryDataset

from lerobot.configs.default import DatasetConfig
from accelerate import Accelerator, DistributedDataParallelKwargs

from einops import rearrange
from dataclasses import dataclass, field
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
import numpy as np
from lerobot.common.policies.egovla.laq.laq_model.piper_arm import PiperArm
from scipy.spatial.transform import Rotation as R

def exists(val):
    return val is not None
import torch.nn.functional as F

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log



@beartype
class LAQTrainer(nn.Module):
    def __init__(
        self,
        vae,                               # 需要训练的 VAE 模型（带 image_size 属性）
        *,
        num_train_steps,                   # 总训练步数
        batch_size,                        # 每次训练的批次大小
        folder,                            # 数据所在文件夹路径
        traj_info=None,                    # 可选：轨迹元信息（未使用）
        train_on_images=False,             # 标记是否按图像方式训练（未使用）
        lr=3e-4,                           # 学习率
        grad_accum_every=1,                # 梯度累积步数（实际 batch_size = batch_size * grad_accum_every）
        wd=0.,                             # 权重衰减
        max_grad_norm=0.5,                 # 梯度裁剪阈值（用于防止 exploding gradients）
        discr_max_grad_norm=None,          # 判别器的梯度裁剪阈值
        save_results_every=50,             # 保存可视化结果频率
        save_model_every=9998,             # 保存模型频率
        input_mode="top_cam_only",         # 训练输入的模式
        results_folder='/home/jiziheng/Music/robot/gs_scene/gs_hs/lerobot/common/policies/egovla/laq/results/0721',        # 保存结果的目录
        use_ema=True,                      # 是否使用 EMA（指数滑动平均）VAE 进行推理
        ema_update_after_step=0,           # EMA 开始更新的步数
        ema_update_every=1,                # EMA 更新频率
        accelerate_kwargs: dict = dict(),  # Accelerator 配置参数
        weights=None,                      # 可选预训练模型权重
        offsets=None,                      # 数据加载时图像帧的偏移量（temporal offset）
    ):
        super().__init__()
        image_size = vae.image_size

        # Accelerator 加速器设置（用于分布式、混合精度训练）
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
        self.accelerator = Accelerator(**accelerate_kwargs, kwargs_handlers=[ddp_kwargs])
        # 📌 记录模型和训练参数
        self.vae = vae
        self.results_folder_str = results_folder
        self.lr = lr
        # 📈 EMA 模型（可选）
        self.use_ema = use_ema
        if self.is_main and use_ema:
            self.ema_vae = EMA(vae, update_after_step = ema_update_after_step, update_every = ema_update_every)
        # 🧮 步数追踪器
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        self.vae.discr = None # this seems to be missing

        self.input_mode = input_mode
        # —— TensorBoard writer —— 
        self.writer = SummaryWriter(log_dir=str(Path(self.results_folder_str) / 'tensorboard'))

        if exists(self.vae.discr):
            all_parameters = set(vae.parameters())
            discr_parameters = set(vae.discr.parameters())
            vae_parameters = all_parameters - discr_parameters

            self.vae_parameters = vae_parameters

            self.optim = get_optimizer(vae_parameters, lr = lr, wd = wd)
            self.discr_optim = get_optimizer(discr_parameters, lr = lr, wd = wd)
        else:
            self.vae_parameters  = set(vae.parameters())
            self.optim = get_optimizer(self.vae_parameters, lr = lr, wd = wd)

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # create dataset
        self.train_on_images = train_on_images
        
        
        # lerobot training
        # self.ds = ParquetTrajectoryDataset(folder, image_size, offset=offsets, input_mode=input_mode)


        folder = "/home/cfy/data/icra_data/data/fold_cloth"
        ds_cfg = DatasetConfig(
            repo_id="kelo234/folding",
            root=folder,
            revision=None,
            episodes=None,
            use_imagenet_stats=False,
            video_backend="cv2",
        )

        # 3. 构造一个带上你那两个 property 的 policy config
        policy_cfg = EgoVLAConfig(
            # n_obs_steps=1 已经是默认，表示只要当前帧
            chunk_size=50,              # state 序列长度
            laq_supervision=True,       # 默认 True，才会加上未来 offset 帧
            laq_temporal_offset=50,     # 表示还要加载第 50 帧
        )

        # 验算一下两个属性：
        print("obs deltas:", policy_cfg.observation_delta_indices)  
        # => [0, 50]

        print("action deltas:", policy_cfg.action_delta_indices[:5], "...", len(policy_cfg.action_delta_indices))
        # => [0, 1, 2, 3, 4] ... 50

        # 4. pretrained config 只要空壳
        # pretrained_cfg = PreTrainedConfig()

        # 5. 打包成 pipeline config
        pipeline_cfg = TrainPipelineConfig(
            dataset=ds_cfg,
            policy=policy_cfg,
        )

        # 6. 调用 make_dataset，直接得到你想要的 dataset
        self.ds = make_dataset(pipeline_cfg)
        sampler = EpisodeAwareSampler(
            self.ds.episode_data_index,
            drop_n_last_frames=True,
            shuffle=True,
        )
        self.dl = DataLoader(
            self.ds,
            num_workers=8,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            pin_memory="cuda",
            drop_last=False,
        )
        
            
        # ⚠️ 这表示训练集和验证集使用完全相同数据
        self.valid_ds = self.ds

        """
        ✅ 用于训练集的 DataLoader
            参数说明：
                shuffle=True              :                   打乱数据，有助于泛化
                num_workers=1             :                   仅开 1 个进程加载数据
                pin_memory=True           :                   加速将数据传输到 GPU
                prefetch_factor=2         :                   每个 worker 预取两个 batch
        """

        """
        ✅ 用于验证集的 DataLoader
            参数说明：
                shuffle=False             :                   验证数据保持顺序
                num_workers=4             :                   验证集通常可以开得更高，因为不用反向传播
        """
        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            num_workers = 16,
            drop_last=True,
            shuffle=False,
            sampler=sampler
        )

        # ⚡️ 使用 HuggingFace Accelerate 进行分布式准备
        if exists(self.vae.discr):
            (
                self.vae,
                self.optim,
                self.discr_optim,
                self.dl
            ) = self.accelerator.prepare(
                self.vae,
                self.optim,
                self.discr_optim,
                self.dl
            )
        else:
            (
                self.vae,
                self.optim,
                self.dl
            ) = self.accelerator.prepare(
                self.vae,
                self.optim,
                self.dl
            )

        # 🔁 构造 DataLoader 的无限循环迭代器
        self.dl_iter = cycle(self.dl)
        # self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every


        self.results_folder = Path(results_folder)


        self.results_folder.mkdir(parents = True, exist_ok = True)


        # 关节角限制幅度
        self.moving_limits = np.array([
            (0.0, 1.0),
            (0.0, 1.0),
            (-2.618, 2.618),
            (0, 3.14),
            (-2.697, 0),
            (-1.832, 1.832),
            (-1.22, 1.22),
            (-3.14, 3.14),
            (0, 1),
            (-2.618, 2.618),
            (0, 3.14),
            (-2.697, 0),
            (-1.832, 1.832),
            (-1.22, 1.22),
            (-3.14, 3.14),
            (0, 1)
        ])

        self.piper_arm = PiperArm()

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        if exists(self.vae.discr):
            pkg = dict(
                model = self.accelerator.get_state_dict(self.vae),
                optim = self.optim.state_dict(),
                discr_optim = self.discr_optim.state_dict(),
                steps = self.steps.item()
            )
        else:
            pkg = dict(
                model=self.accelerator.get_state_dict(self.vae),
                optim=self.optim.state_dict(),
                steps=self.steps.item()
            )

        # Save DataLoader state
        pkg['dl_iter_state'] = self.get_dl_state(self.dl_iter)

        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        vae = self.accelerator.unwrap_model(self.vae)
        vae.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])
        if exists(self.vae.discr):
            self.discr_optim.load_state_dict(pkg['discr_optim'])


    def cal_fk_result(self, action_state):
        """
        action_state: [B, T, 16]
        output: [B, T, 14] - [left_xyz + left_wxyz + right_xyz + right_wxyz]
        """

        B, T, _ = action_state.shape
        action_np = action_state.cpu().numpy()  # 转 numpy，方便逐个处理

        result = []

        for b in range(B):
            traj = []
            for t in range(T):
                # 提取左臂关节角（弧度）
                left_joints = action_np[b, t, 2:8]  # [6]
                # 提取右臂关节角（弧度）
                right_joints = action_np[b, t, 9:15]  # [6]

                # FK 左臂
                T_left = self.piper_arm.forward_kinematics(left_joints)  # [4, 4]
                pos_left = T_left[:3, 3]  # [x, y, z]
                rot_left = R.from_matrix(T_left[:3, :3]).as_quat()  # [x, y, z, w]

                # FK 右臂
                T_right = self.piper_arm.forward_kinematics(right_joints)
                pos_right = T_right[:3, 3]
                rot_right = R.from_matrix(T_right[:3, :3]).as_quat()

                # 将四元数变为 wxyz 顺序
                quat_left = np.concatenate(([rot_left[3]], rot_left[:3]))   # [w, x, y, z]
                quat_right = np.concatenate(([rot_right[3]], rot_right[:3]))

                # 拼接左右臂的末端位姿
                frame_pose = np.concatenate([pos_left, quat_left, pos_right, quat_right])  # [14]
                traj.append(frame_pose)

            result.append(traj)

        result = torch.tensor(result, dtype=action_state.dtype, device=action_state.device)  # [B, T, 14]
        return result




    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        # 模型/数据将被移动到指定设备
        device = self.device
        # 从 Tensor 中取出当前训练步数
        steps = int(self.steps.item())
        # 将 VAE 置于训练模式（启用 dropout 等）
        self.vae.train()

        # logs

        logs = {}
        # print(f"train_step !!!")

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            # 训练数据迭代器
            if self.input_mode == "top_cam_only":
                img = next(self.dl_iter)
                img = img.to(device)
                # 前向传播，计算重建损失和码本中用到的唯一 token 数
                recon_img_loss, recon_state_loss, loss, num_unique_indices = self.vae(
                    img,
                    step=steps,
                )
            elif self.input_mode == "top_cam_state":
                # action_state : [B, 51, 16]
                batch = next(self.dl_iter)
                # print("...")
                # img, action_state = next(self.dl_iter)
                img = batch["observation.images.top"]
                action_state = batch["action"]

                fk_result_state = self.cal_fk_result(action_state)       # [B, chunk_size, 14]

                fk_min_vals = fk_result_state.amin(dim=(0, 1), keepdim=True)  # shape [1, 1, 14]
                fk_max_vals = fk_result_state.amax(dim=(0, 1), keepdim=True)  # shape [1, 1, 14]

                # 避免除以 0
                fk_range_vals = fk_max_vals - fk_min_vals
                fk_range_vals[fk_range_vals == 0] = 1e-6

                fk_normalized = (fk_result_state - fk_min_vals) / fk_range_vals  # shape [B, T, 14]


                # 假设 action_state shape: [batch, chunk_size, 16]
                # 假设 self.moving_limits 是 numpy 数组 shape: [16, 2]
                min_vals = torch.tensor(self.moving_limits[:, 0], dtype=action_state.dtype, device=action_state.device)
                max_vals = torch.tensor(self.moving_limits[:, 1], dtype=action_state.dtype, device=action_state.device)

                # Reshape 为 [1, 1, 16] 以便广播到整个 batch 和 chunk
                min_vals = min_vals.view(1, 1, -1)
                max_vals = max_vals.view(1, 1, -1)

                # 避免除以 0 的情况（如果 max==min）
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1e-6  # 避免除以零
                action_state = (action_state - min_vals) / range_vals
                img = img.to(device)
                action_state.to(device)
                # 前向传播，计算重建损失和码本中用到的唯一 token 数
                recon_img_loss, recon_state_loss, recon_fk_state_loss, loss, num_unique_indices = self.vae(
                    img,
                    action_state=action_state,
                    fk_result_state=fk_normalized,
                    step=steps,
                )
            else:
                raise ValueError("you must provided a correct input mode. ")
            
            
            # 反向传播梯度，考虑梯度累积
            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss':             loss.item() / self.grad_accum_every})
            accum_log(logs, {'recon_img_loss':   recon_img_loss.item() / self.grad_accum_every})
            accum_log(logs, {'recon_state_loss': recon_state_loss.item() / self.grad_accum_every})
            accum_log(logs, {'recon_fk_state_loss': recon_fk_state_loss.item() / self.grad_accum_every})
            accum_log(logs, {'num_unique_indices': num_unique_indices})

              # 显式写入各项 metric 到 TensorBoard
            
        # 梯度裁剪 & 优化器更新
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        step = int(self.steps.item())
        self.writer.add_scalar('loss',               logs['loss'],               step)
        self.writer.add_scalar('recon_img_loss',     logs['recon_img_loss'],     step)
        self.writer.add_scalar('recon_state_loss',   logs['recon_state_loss'],   step)
        self.writer.add_scalar('recon_fk_state_loss',   logs['recon_fk_state_loss'],   step)
        self.writer.add_scalar('num_unique_indices', logs['num_unique_indices'], step)


        # if self.is_main:  # Ensure only the main process logs in a distributed setting
        #     wandb.log(logs)

        if self.is_main and self.use_ema:
            self.ema_vae.update()

        if self.is_main and not (steps % self.save_results_every):
            unwrapped_vae = self.accelerator.unwrap_model(self.vae)
            vaes_to_evaluate = ((unwrapped_vae, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = ((self.ema_vae.ema_model, f'{steps}.ema'),) + vaes_to_evaluate

            for model, filename in vaes_to_evaluate:
                model.eval()

                if self.input_mode == "top_cam_only":
                    valid_data = next(self.valid_dl_iter)
                elif self.input_mode == "top_cam_state":
                    val_batch = next(self.valid_dl_iter)
                    valid_data = val_batch["observation.images.top"]
                    valid_action_data = val_batch["action"]
                else:
                    raise ValueError("you must provided a correct input mode. ")


                valid_data = valid_data.to(device)
                valid_action_data = valid_action_data.to(device)

                recons_img, recons_state = model(valid_data, valid_action_data, return_recons_only = True)


                if self.train_on_images:
                    imgs_and_recons = torch.stack((valid_data, recons_img), dim = 0)
                    # imgs_and_recons = torch.stack((valid_data, recons), dim = 0)
                    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                    imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                    grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

                    logs['reconstructions'] = grid
                    save_image(grid, str(self.results_folder / f'{filename}.png'))
                else:
                    # imgs_and_recons = torch.stack((valid_data[:,:,0],valid_data[:,:,-1], recons, recons+valid_data[:,:,0]), dim = 0)
                    frame0 = valid_data.permute(0,2,1,3,4)[:, :, 0]      # [B, C, H, W]
                    frameN = valid_data.permute(0,2,1,3,4)[:, :, -1]     # [B, C, H, W]
                    _, _, H_rec, W_rec = recons_img.shape
                    # 如果原始帧尺寸 != 重构帧尺寸，就插值 resize
                    if frame0.shape[-2:] != (H_rec, W_rec):
                        frame0 = F.interpolate(frame0, size=(H_rec, W_rec),
                                            mode='bilinear', align_corners=False)
                        frameN = F.interpolate(frameN, size=(H_rec, W_rec),
                                            mode='bilinear', align_corners=False)

                    imgs_and_recons = torch.stack((frame0,frameN, recons_img), dim = 0)
                    # imgs_and_recons = torch.stack((valid_data, recons), dim = 0)
                    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                    imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                    grid = make_grid(imgs_and_recons, nrow = 3, normalize = False, value_range = (0, 1))

                    pil = TF.to_pil_image(grid)  
                    draw = ImageDraw.Draw(pil)
                    font = ImageFont.load_default()

                    B = valid_action_data.shape[0]
                    for i in range(B):
                        gt = valid_action_data[i].cpu().numpy().tolist()
                        rec = recons_state[i].cpu().detach().numpy().tolist()
                        # 每个样本占用 3 张图，图宽 = W_rec，图高 = H_rec
                        W_rec, H_rec = pil.size
                        per_w = W_rec // 3
                        # 在第 i 个样本行的最上方、每张子图左上角写
                        text = f"GT: {gt}\nREC: {rec}"
                        x = (i*3)*per_w + 5  # 第 i 样本第 1 子图左边起点
                        y = 5
                        draw.text((x, y), text, font=font, fill=(255,255,255))


                    # —— 3. 转回 tensor，写入 logs —— 
                    grid_annot = TF.to_tensor(pil).clamp(0,1)
                    logs['reconstructions'] = grid_annot

                    save_image(grid_annot, str(self.results_folder / f'{filename}.png'))

            self.print(f'{steps}: saving to {str(self.results_folder)}')

        if self.is_main and not (steps % self.save_model_every):
            state_dict = self.vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            torch.save(state_dict, model_path)

            if self.use_ema:
                ema_state_dict = self.ema_vae.state_dict()
                model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
                torch.save(ema_state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):
        device = next(self.vae.parameters()).device
        # if self.accelerator.is_main_process:
        #     wandb.init(project='phenaki_cnn',name=self.results_folder_str.split('/')[-1], config={
        #         "learning_rate": self.lr,
        #         "batch_size": self.batch_size,
        #         "num_train_steps": self.num_train_steps,
        #     })

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
        if self.accelerator.is_main_process:
            self.writer.close()
        #     wandb.finish()  
