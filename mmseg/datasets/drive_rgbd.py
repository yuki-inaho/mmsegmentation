# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union
from pathlib import Path

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose
from mmseg.datasets.basesegdataset import BaseSegDataset

from mmseg.registry import DATASETS


@DATASETS.register_module()
class BaseRGBDSegDataset(BaseSegDataset):
    METAINFO: dict = dict()

    def __init__(
        self,
        ann_file: str = "",
        img_suffix=".jpg",
        seg_map_suffix=".png",
        depth_map_suffix=".png",
        metainfo: Optional[dict] = None,
        data_root: Optional[str] = None,
        data_prefix: dict = dict(img_path="", seg_map_path="", depth_map_path=""),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
        ignore_index: int = 255,
        reduce_zero_label: bool = False,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.depth_map_suffix = depth_map_suffix
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get("classes", None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(dict(label_map=self.label_map, reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert (
                self._metainfo.get("classes") is not None
            ), "dataset metainfo `classes` should be specified when testing"

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get("img_path", None)
        depth_dir = self.data_prefix.get("depth_map_path", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)
        if not osp.isdir(self.ann_file) and self.ann_file:
            raise NotImplementedError("Only support directory format!")
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                dir_path=img_dir, list_dir=False, suffix=self.img_suffix, recursive=True, backend_args=self.backend_args
            ):
                data_info = dict(img_path=osp.join(img_dir, img))
                img_name_stem = img[:-_suffix_len]
                # @TODO: implement existance check?
                if (ann_dir is not None) and (depth_dir is not None):
                    seg_map = img_name_stem + self.seg_map_suffix
                    data_info["seg_map_path"] = osp.join(ann_dir, seg_map)
                    depth_map = img_name_stem + self.depth_map_suffix
                    data_info["depth_map_path"] = osp.join(depth_dir, depth_map)
                data_info["label_map"] = self.label_map
                data_info["reduce_zero_label"] = self.reduce_zero_label
                data_info["seg_fields"] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x["img_path"])
        return data_list


@DATASETS.register_module()
class DRIVECustomRGBDDataset(BaseRGBDSegDataset):
    """Custom DRIVE dataset for RGB-D data."""

    METAINFO = dict(classes=("background", "stem"), palette=[[120, 120, 120], [255, 255, 46]])

    def __init__(
        self,
        img_suffix="_rgb.jpg",
        depth_map_suffix="_depth.png",
        seg_map_suffix="_ann.png",
        reduce_zero_label=False,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            depth_map_suffix=depth_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )
        assert fileio.exists(self.data_prefix["img_path"], backend_args=self.backend_args)


@DATASETS.register_module()
class DRIVECustomMulticatRGBDDataset(BaseRGBDSegDataset):
    """Custom DRIVE dataset for RGB-D data."""

    METAINFO = dict(
        classes=("background", "stem", "greenfruits"), palette=[[120, 120, 120], [255, 255, 46], [46, 255, 46]]
    )

    def __init__(
        self,
        img_suffix="_rgb.jpg",
        depth_map_suffix="_depth.png",
        seg_map_suffix="_ann.png",
        reduce_zero_label=False,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            depth_map_suffix=depth_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )
        assert fileio.exists(self.data_prefix["img_path"], backend_args=self.backend_args)
