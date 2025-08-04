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
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from lerobot.common.datasets.factory import make_dataset, make_dataset_local, resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.MixLerobotDataset import MixLeRobotDataset, MultiMixLeRobotDataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import MixedDatasetConfig
from lerobot.scripts.eval import eval_policy
from pathlib import Path
from omegaconf import OmegaConf, OmegaConf, OmegaConf
from torch.utils.tensorboard import SummaryWriter

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

def evaluate_validation_loss(policy, val_loader, device, use_amp=False, max_batches=10):
    logging.info(colored("Validating.", "yellow", attrs=["bold"]))
    policy.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad(), (torch.autocast(device_type=device.type) if use_amp else nullcontext()):
         for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            loss, _ = policy.forward(batch)
            total_loss += loss.item()
            num_batches += 1
    policy.train()
    logging.info(colored("Validating done.", "yellow", attrs=["bold"]))
    return total_loss / max(num_batches, 1)



@parser.wrap()
def train(cfg: TrainPipelineConfig):
   
    cfg.validate()
    experiment_name = cfg.output_dir.name

    writer = SummaryWriter(log_dir=str(cfg.output_dir / f"tensorboard/{experiment_name}"))

    logging.info(pformat(cfg.to_dict()))
    if cfg.dataset_yaml:
        logging.info(colored(f"Load Mixed dataset config from YAML: {cfg.dataset_yaml}.", "yellow", attrs=["bold"]))
        ds_cfg_dict = OmegaConf.load(cfg.dataset_yaml)
        cfg.dataset = MixedDatasetConfig(**OmegaConf.to_object(ds_cfg_dict))
        logging.info(colored("dataset config: MixedDatasetConfig.", "yellow", attrs=["bold"]))
        
    else:
        cfg.dataset.is_mix = None

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    # offline_dataset = make_dataset(cfg)
    # dataset = make_dataset_local(cfg)

    from lerobot.common.datasets.transforms import ImageTransforms
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    if cfg.dataset.is_mix:

        ds_cfgs = cfg.dataset.datasets 
        repo_ids = [d.repo_id for d in ds_cfgs]
        roots    = [Path(d.lerobot_root) for d in ds_cfgs]
        name = [d.name for d in ds_cfgs]

        ds_meta = LeRobotDatasetMetadata(
            repo_ids[0], roots[0], revision=cfg.dataset.revision
        )

        offline_dataset = MultiMixLeRobotDataset(
            repo_ids=repo_ids,
            root=roots,
            name= name,
            delta_timestamps=resolve_delta_timestamps(cfg.policy, ds_meta),
            image_transforms=image_transforms,
            scene_pt_list=[d.scene_embedding_pt for d in ds_cfgs],
            subtask_pt_list=[d.subtask_embedding_pt for d in ds_cfgs],
            video_backend=cfg.dataset.video_backend,
        )

        for i, ds_cfg in enumerate(ds_cfgs):
            name = ds_cfg.name
            scene_path   = offline_dataset.scene_pt_list[i]
            subtask_path = offline_dataset.subtask_pt_list[i]

            logging.info(
                f"[Dataset '{name}']\n"
                f"  scene_pt:   {scene_path!r} \n"
                f"  subtask_pt: {subtask_path!r}"
            )
    else:
        # ds_cfg = cfg.datasets.datasets[0]
        # resolve_delta_timestamps(cfg)
        # offline_dataset = MixLeRobotDataset(
        #     repo_id=ds_cfg.repo_id,
        #     root=ds_cfg.lerobot_dataset_path,
        #     name= ds_cfg.name,
        #     local_files_only = ds_cfg.local_files_only,
        #     delta_timestamps = cfg.training.get("delta_timestamps"),
        #     scene_pt=ds_cfg.scene_embedding_pt,    
        #     subtask_pt=ds_cfg.subtask_embedding_pt,
        #     video_backend=cfg.video_backend,
        # )
        # logging.info(
        #         f"[Dataset '{ds_cfg.name,}']\n"
        #         f"  scene_pt:   {ds_cfg.scene_embedding_pt}\n"
        #         f"  subtask_pt: {ds_cfg.subtask_embedding_pt}"
        #     )
        offline_dataset = make_dataset(cfg)
    if isinstance(offline_dataset, MultiLeRobotDataset):
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(offline_dataset.repo_id_to_index , indent=2)}"
        )


    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=offline_dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)
    best_ckpt_dir = cfg.output_dir / "best_model"
    best_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{offline_dataset.num_frames=} ({format_big_number(offline_dataset.num_frames)})")
    logging.info(f"{offline_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None


    val_ratio = 0.05
    val_size = int(len(offline_dataset) * val_ratio)
    train_size = len(offline_dataset) - val_size
    train_dataset, val_dataset = random_split(offline_dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(train_dataloader)

    val_loader = DataLoader(
        val_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, offline_dataset.num_frames, offline_dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")


  
    
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        ## ------图片可视化验证 -----------
        # imgs = batch["observation.images.top"].cpu().numpy()

        # sample_idx = 0
        # frames = imgs[sample_idx]  # shape [T, C, H, W]

        # # 在一张图里左右并排显示所有帧
        # fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 5))
        # for i, frame in enumerate(frames):
        #     img = np.transpose(frame, (1, 2, 0))  # C,H,W -> H,W,C
        #     axes[i].imshow(img)
        #     axes[i].axis('off')
        #     axes[i].set_title(f"Frame {i}")
        # plt.tight_layout()
        # plt.show()
        ## ------图片可视化验证 -----------


        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)

            writer.add_scalar("train/loss", train_tracker.loss.avg, step)
            writer.add_scalar("train/lr", train_tracker.lr.avg, step)
            writer.add_scalar("train/grad_norm", train_tracker.grad_norm.avg, step)

            val_loss = evaluate_validation_loss(policy, val_loader, device, cfg.policy.use_amp, max_batches=10)
            writer.add_scalar("val/loss", val_loss, step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.info(colored(f"New best val_loss: {val_loss:.4f}, saving model to {best_ckpt_dir}", "green"))
                save_checkpoint(best_ckpt_dir, step, cfg, policy, optimizer, lr_scheduler)

            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, offline_dataset.num_frames, offline_dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")
    writer.close()


if __name__ == "__main__":
    init_logging()
    train()
