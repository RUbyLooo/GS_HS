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
from pprint import pformat

import torch

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig


IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.images") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
        if key.startswith("observation.state") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def make_dataset_local(cfg, split: str = "train") -> LeRobotDataset | MultiLeRobotDataset:
    """
    Args:
        cfg: A Hydra config as per the LeRobot config scheme.
        split: Select the data subset used to create an instance of LeRobotDataset.
            All datasets hosted on [lerobot](https://huggingface.co/lerobot) contain only one subset: "train".
            Thus, by default, `split="train"` selects all the available data. `split` aims to work like the
            slicer in the hugging face datasets:
            https://huggingface.co/docs/datasets/v2.19.0/loading#slice-splits
            As of now, it only supports `split="train[:n]"` to load the first n frames of the dataset or
            `split="train[n:]"` to load the last n frames. For instance `split="train[:1000]"`.
    Returns:
        The LeRobotDataset.
    """
    print(cfg.root_path)
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    # A soft check to warn if the environment matches the dataset. Don't check if we are using a real world env (dora).
    if cfg.env.name != "dora":
        if isinstance(cfg.dataset_repo_id, str):
            dataset_repo_ids = [cfg.dataset_repo_id]  # single dataset
        else:
            dataset_repo_ids = cfg.dataset_repo_id  # multiple datasets

        for dataset_repo_id in dataset_repo_ids:
            if cfg.env.name not in dataset_repo_id:
                logging.warning(
                    f"There might be a mismatch between your training dataset ({dataset_repo_id=}) and your "
                    f"environment ({cfg.env.name=})."
                )

    resolve_delta_timestamps(cfg)

    # image_transforms = None
    # if cfg.training.image_transforms.enable:
    #     cfg_tf = cfg.training.image_transforms
    #     image_transforms = get_image_transforms(
    #         brightness_weight=cfg_tf.brightness.weight,
    #         brightness_min_max=cfg_tf.brightness.min_max,
    #         contrast_weight=cfg_tf.contrast.weight,
    #         contrast_min_max=cfg_tf.contrast.min_max,
    #         saturation_weight=cfg_tf.saturation.weight,
    #         saturation_min_max=cfg_tf.saturation.min_max,
    #         hue_weight=cfg_tf.hue.weight,
    #         hue_min_max=cfg_tf.hue.min_max,
    #         sharpness_weight=cfg_tf.sharpness.weight,
    #         sharpness_min_max=cfg_tf.sharpness.min_max,
    #         max_num_transforms=cfg_tf.max_num_transforms,
    #         random_order=cfg_tf.random_order,
    #     )

    # if isinstance(cfg.dataset_repo_id, str):
    if isinstance(cfg.root_path, str):
        print("single_dataset")
        # TODO (aliberts): add 'episodes' arg from config after removing hydra
        dataset = LeRobotDataset(
            cfg.dataset_repo_id,
            delta_timestamps=cfg.training.get("delta_timestamps"),
            image_transforms=image_transforms,
            video_backend=cfg.video_backend,
            root=cfg.root_path,
            
            
        )
    else:
        print("multi_dataset")
        dataset = MultiLeRobotDataset(
            cfg.dataset_repo_id,
            delta_timestamps=cfg.training.get("delta_timestamps"),
            image_transforms=image_transforms,
            video_backend=cfg.video_backend,
            root=cfg.root_path,
          
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
