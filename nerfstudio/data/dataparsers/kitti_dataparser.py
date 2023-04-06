# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class KittiDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: Kitti)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    calib: Path = Path("/data/datasets/KITTI-360/calibration/perspective.txt")
    """Camera Intrinsics"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class Kitti(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: KittiDataParserConfig

    def __init__(self, config: KittiDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.calib: Path = config.calib
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        poses = []
        for frame in meta:
            fname = frame['image_path']
            image_filenames.append(fname)
            poses.append(np.array(frame["c2w"]).reshape(4, 4))
        poses = np.array(poses).astype(np.float32)

        bbox = np.array(meta[0]['car_bbox']['vertices'])
        R = np.array(meta[0]['car_bbox']['R'])
        T = np.array(meta[0]['car_bbox']['T'])
        box_norm = np.linalg.norm(R, axis=1)
        bbox = bbox * box_norm
        # data_transform = np.concatenate((R, T[None, :].T), axis=1)
        bbox_min = (np.min(bbox, axis=0) * 1.2).tolist()
        bbox_max = (np.max(bbox, axis=0) * 1.2).tolist()

        Intrinsic = None
        with open(self.calib, 'r') as rf:
            for line in rf:
                if line.startswith('P_rect_00'):
                    Intrinsic = list(map(float, line.strip().split(' ')[1:]))
                    Intrinsic = np.array(Intrinsic).reshape(3, 4)

        focal_length_x = Intrinsic[0][0]
        focal_length_y = Intrinsic[1][1]

        cx = Intrinsic[0][2]
        cy = Intrinsic[1][2]
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([bbox_min, bbox_max], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length_x,
            fy=focal_length_y,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs
