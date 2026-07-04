# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DRIVECustomDataset(BaseSegDataset):
    """Custom DRIVE dataset."""

    METAINFO = dict(classes=("background", "stem"), palette=[[120, 120, 120], [255, 255, 46]])

    def __init__(self, img_suffix=".png", seg_map_suffix=".png", reduce_zero_label=False, **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label, **kwargs
        )
        assert fileio.exists(self.data_prefix["img_path"], backend_args=self.backend_args)


@DATASETS.register_module()
class DRIVECustomDataset3Cat(BaseSegDataset):
    """Custom DRIVE dataset."""

    METAINFO = dict(classes=("background", "stem_foreground", "stem_background"), palette=[[120, 120, 120], [255, 255, 46], [0, 255, 255]])

    def __init__(self, img_suffix=".png", seg_map_suffix=".png", reduce_zero_label=False, **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label, **kwargs
        )
        assert fileio.exists(self.data_prefix["img_path"], backend_args=self.backend_args)
