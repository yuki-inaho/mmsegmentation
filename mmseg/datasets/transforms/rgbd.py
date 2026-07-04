import warnings
import numpy as np
import mmcv
import torch
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from mmcv.transforms.base import BaseTransform
from mmcv.transforms import to_tensor
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
import mmengine.fileio as fileio
from mmcv.transforms.processing import Resize
from mmcv.image.geometric import imrescale
from mmseg.datasets.transforms import ResizeToMultiple


@TRANSFORMS.register_module()
class LoadDepthImageFromFile(BaseTransform):
    """Load a depth image from file.

    Required Keys:

    - depth_map_path

    Modified Keys:

    - depth_map
    - depth_map_shape
    - depth_map_ori_shape

    Args are same as LoadImageFromFile.
    """

    def __init__(
        self,
        imdecode_backend: str = "cv2",
        ignore_empty: bool = False,
        threshold_max_depth_int: int = 2000,
        depth_scale: float = 0.001,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.ignore_empty = ignore_empty
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        self.threshold_max_depth_int = threshold_max_depth_int
        self.depth_scale = depth_scale
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load depth image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: The dict contains loaded depth image and meta information.
        """

        filename = results["depth_map_path"]
        try:
            img_bytes = fileio.get(filename, backend_args=self.backend_args)
            depth_img = mmcv.imfrombytes(img_bytes, flag="unchanged", backend=self.imdecode_backend).astype(np.uint16)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        assert depth_img is not None, f"failed to load image: {filename}"

        # Depth thresholding
        depth_img[depth_img > self.threshold_max_depth_int] = 0

        # Convert to float32 and rescale
        depth_img_f = depth_img.astype(np.float32) * self.depth_scale

        results["depth_map"] = depth_img_f
        results["depth_map_shape"] = depth_img.shape[:2]
        results["depth_map_ori_shape"] = depth_img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"ignore_empty={self.ignore_empty}, "
            f"imdecode_backend='{self.imdecode_backend}', "
        )
        repr_str += f"backend_args={self.backend_args})"

        return repr_str


@TRANSFORMS.register_module()
class PackRGBDSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation with RGBD data.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image
        - ``ori_shape``: original shape of the image as a tuple (h, w, c)
        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - ``pad_shape``: shape of padded images
        - ``scale_factor``: a float indicating the preprocessing scale
        - ``flip``: a boolean indicating if image flip transform was used
        - ``flip_direction``: the flipping direction
    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``( 'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction' )``
    """

    def __init__(
        self,
        meta_keys=(
            "img_path",
            "depth_map_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "reduce_zero_label",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if ("img" in results) and ("depth_map" in results):
            img = results["img"]
            depth_map = results["depth_map"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = img.astype(np.float32)
            depth_map = depth_map.astype(np.float32)[None, ...]
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                depth_map = to_tensor(np.ascontiguousarray(depth_map))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
                depth_map = to_tensor(depth_map).contiguous()

            # concat the depth map to the image
            input_tensor = torch.cat((img, depth_map), dim=0)
            packed_results["inputs"] = input_tensor

        data_sample = SegDataSample()
        if "gt_seg_map" in results:
            if len(results["gt_seg_map"].shape) == 2:
                data = to_tensor(results["gt_seg_map"][None, ...].astype(np.int64))
            else:
                warnings.warn(
                    "Please pay attention your ground truth "
                    "segmentation map, usually the segmentation "
                    "map is 2D, but got "
                    f'{results["gt_seg_map"].shape}'
                )
                data = to_tensor(results["gt_seg_map"].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if "gt_edge_map" in results:
            gt_edge_data = dict(data=to_tensor(results["gt_edge_map"][None, ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str


@TRANSFORMS.register_module()
class ResizeRGBD(Resize):
    def __init__(
        self,
        scale: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        keep_ratio: bool = False,
        clip_object_border: bool = True,
        backend: str = "cv2",
        interpolation="bilinear",
    ):
        super().__init__(scale, scale_factor, keep_ratio, clip_object_border, backend, interpolation)

    def _resize_depth(self, results: dict) -> None:
        """Resize depth map with ``results['scale']``."""
        if results.get("depth_map", None) is not None:
            if self.keep_ratio:
                depth_map = mmcv.imrescale(
                    results["depth_map"], results["scale"], interpolation="nearest", backend=self.backend
                )
            else:
                depth_map = mmcv.imresize(
                    results["depth_map"], results["scale"], interpolation="nearest", backend=self.backend
                )
            results["depth_map"] = depth_map

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results["scale"] = self.scale
        else:
            img_shape = results["img"].shape[:2]
            results["scale"] = _scale_size(img_shape[::-1], self.scale_factor)  # type: ignore
        self._resize_depth(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        return results


@TRANSFORMS.register_module()
class ResizeToMultipleRGBD(ResizeToMultiple):
    def __init__(self, size_divisor=32, interpolation=None):
        super().__init__(size_divisor=32, interpolation=None)

    def transform(self, results: dict) -> dict:
        """Call function to resize images, semantic segmentation map to
        multiple of size divisor.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape' keys are updated.
        """
        # Align image to multiple of size divisor (RGB).
        img = results["img"]
        img = mmcv.imresize_to_multiple(
            img,
            self.size_divisor,
            scale_factor=1,
            interpolation=self.interpolation if self.interpolation else "bilinear",
        )
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["pad_shape"] = img.shape[:2]

        # Align image to multiple of size divisor (RGB).
        depth_map = results["depth_map"]
        depth_map = mmcv.imresize_to_multiple(depth_map, self.size_divisor, scale_factor=1, interpolation="nearest")
        results["depth_map"] = depth_map
        assert depth_map.shape[:2] == img.shape[:2]

        # Align segmentation map to multiple of size divisor.
        for key in results.get("seg_fields", []):
            gt_seg = results[key]
            gt_seg = mmcv.imresize_to_multiple(gt_seg, self.size_divisor, scale_factor=1, interpolation="nearest")
            results[key] = gt_seg

        return results


@TRANSFORMS.register_module()
class GenerateNullDepthImage(BaseTransform):
    """Generate a pseudo depth image input for model conversion.
    Modified Keys:

    - depth_map_path
    - depth_map
    - depth_map_shape
    - depth_map_ori_shape

    Args are same as LoadImageFromFile.
    """

    def __init__(
        self,
        scale: Tuple[int, int] = (640, 480),  # (w, h)
    ) -> None:
        self.scale = scale

    def transform(self, results: dict) -> Optional[dict]:
        results["depth_map_path"] = ""
        # Depth thresholding
        depth_img_f = np.zeros(self.scale[::-1], dtype=np.float32)

        results["depth_map"] = depth_img_f
        results["depth_map_shape"] = depth_img_f.shape[:2]
        results["depth_map_ori_shape"] = depth_img_f.shape[:2]
        return results
