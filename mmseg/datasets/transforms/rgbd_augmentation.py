#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RGB-D Augmentation Transforms for MMSegmentation

This module contains RGB-D specific augmentation transforms adapted from MMRotate
for use with MMSegmentation. The transforms are designed to work with semantic
segmentation tasks that use RGB-D input data.

Adapted transforms:
- RandomSizedCropRGBD: Multi-scale cropping for RGB-D semantic segmentation
- CoarseDropoutDepth: Simulates depth sensor dropout/occlusions
- RandomDepthOffset: Adds random offset to depth values for robustness
"""

import copy
import random
from typing import Dict, Optional, Tuple, Union
import numpy as np
import mmcv
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class RandomSizedCropRGBD(BaseTransform):
    """Randomly crop a sub-image of a certain scale, then resize it back for RGB-D segmentation.

    This transform first samples a random scale from the given range. It then
    performs a random crop of the input image, depth map, and segmentation mask
    at that scale, maintaining the original aspect ratio. Finally, it resizes
    the cropped region back to the target size.

    This is useful for learning representations that are robust to objects
    appearing at different scales in semantic segmentation tasks.

    Required Keys in results:
        - img (np.ndarray): The RGB image.
        - depth_map (np.ndarray, optional): The depth map.
        - gt_seg_map (np.ndarray): The segmentation mask.
        - img_shape (tuple): The shape of the image.

    Modified Keys in results:
        - img
        - depth_map
        - gt_seg_map
        - img_shape

    Args:
        min_max_height (tuple[int, int]): The range of heights to sample for
            the crop, in pixels. The width will be scaled proportionally to
            maintain the aspect ratio.
        height (int): The final height to resize the cropped image to. This
            should match the model's expected input height.
        width (int): The final width to resize the cropped image to. This
            should match the model's expected input width.
        prob (float): The probability of performing this transform.
            Defaults to 1.0.
        max_resample_attempts (int): The maximum number of times to resample
            the crop if it contains only background pixels. Defaults to 10.
        interpolation (str): Interpolation method for resizing the RGB image.
            Defaults to 'bilinear'.
        depth_interpolation (str): Interpolation method for resizing the
            depth map. Defaults to 'nearest'.
        seg_interpolation (str): Interpolation method for resizing the
            segmentation mask. Defaults to 'nearest'.
    """

    def __init__(
        self,
        min_max_height: Tuple[int, int],
        height: int,
        width: int,
        prob: float = 1.0,
        max_resample_attempts: int = 10,
        interpolation: str = "bilinear",
        depth_interpolation: str = "nearest",
        seg_interpolation: str = "nearest",
    ) -> None:
        assert isinstance(min_max_height, tuple) and len(min_max_height) == 2
        assert 0 < min_max_height[0] <= min_max_height[1]
        assert height > 0 and width > 0
        assert 0.0 <= prob <= 1.0
        assert max_resample_attempts >= 0

        self.min_max_height = min_max_height
        self.height = height
        self.width = width
        self.prob = prob
        self.max_resample_attempts = max_resample_attempts
        self.interpolation = interpolation
        self.depth_interpolation = depth_interpolation
        self.seg_interpolation = seg_interpolation

    def _get_random_crop_params(self, ori_h: int, ori_w: int) -> Tuple[int, int, int, int]:
        """Determine random crop height, width, and offset."""
        # 1. Determine crop size based on random height
        crop_h = random.randint(self.min_max_height[0], self.min_max_height[1])
        aspect_ratio = ori_w / ori_h
        crop_w = int(round(crop_h * aspect_ratio))

        # Ensure crop size is not larger than original size
        crop_h = min(crop_h, ori_h)
        crop_w = min(crop_w, ori_w)

        # 2. Determine crop offset
        margin_h = max(ori_h - crop_h, 0)
        margin_w = max(ori_w - crop_w, 0)
        offset_h = random.randint(0, margin_h)
        offset_w = random.randint(0, margin_w)

        return crop_h, crop_w, offset_h, offset_w

    def _has_valid_segmentation(self, seg_mask: np.ndarray) -> bool:
        """Check if the segmentation mask contains non-background pixels."""
        # Assuming background class is 0
        return np.any(seg_mask != 0)

    def transform(self, results: Dict) -> Dict:
        """Apply the random sized crop transformation."""
        if random.random() > self.prob:
            return results

        ori_h, ori_w = results["img_shape"]
        original_seg_map = results.get("gt_seg_map")

        for i in range(self.max_resample_attempts + 1):
            crop_h, crop_w, offset_h, offset_w = self._get_random_crop_params(ori_h, ori_w)

            # Check if crop contains valid segmentation (non-background pixels)
            if original_seg_map is not None:
                cropped_seg = original_seg_map[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]
                if not self._has_valid_segmentation(cropped_seg):
                    # If this is the last attempt, proceed anyway
                    if i < self.max_resample_attempts:
                        continue

            # Apply the crop and resize
            break

        # --- Crop and resize RGB image ---
        img = results["img"][offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]
        target_size = (self.width, self.height)
        img_resized = mmcv.imresize(img, target_size, interpolation=self.interpolation, backend="cv2")
        results["img"] = img_resized

        # --- Crop and resize depth map ---
        if "depth_map" in results and results.get("depth_map") is not None:
            depth_map = results["depth_map"][offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]
            depth_map_resized = mmcv.imresize(
                depth_map, target_size, interpolation=self.depth_interpolation, backend="cv2"
            )
            results["depth_map"] = depth_map_resized

        # --- Crop and resize segmentation mask ---
        if "gt_seg_map" in results and results.get("gt_seg_map") is not None:
            gt_seg_map = results["gt_seg_map"][offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]
            gt_seg_map_resized = mmcv.imresize(
                gt_seg_map, target_size, interpolation=self.seg_interpolation, backend="cv2"
            )
            results["gt_seg_map"] = gt_seg_map_resized

        # --- Update metadata ---
        results["img_shape"] = (self.height, self.width)

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"min_max_height={self.min_max_height}, "
            f"height={self.height}, width={self.width}, "
            f"prob={self.prob}, "
            f"max_resample_attempts={self.max_resample_attempts}, "
            f"interpolation='{self.interpolation}', "
            f"depth_interpolation='{self.depth_interpolation}', "
            f"seg_interpolation='{self.seg_interpolation}')"
        )


@TRANSFORMS.register_module(force=True)
class CoarseDropoutDepth(BaseTransform):
    """Apply CoarseDropout to the depth map only.

    This transform simulates sensor dropouts or occlusions in the depth channel
    by creating rectangular holes. The RGB image and segmentation masks remain unchanged.

    Required Keys:
        - depth_map (np.ndarray)

    Modified Keys:
        - depth_map

    Args:
        max_holes (int): Maximum number of holes to draw. Defaults to 8.
        max_height (int): Maximum height of a hole. Defaults to 8.
        max_width (int): Maximum width of a hole. Defaults to 8.
        min_holes (int, optional): Minimum number of holes. Defaults to `max_holes`.
        min_height (int, optional): Minimum height of a hole. Defaults to `max_height`.
        min_width (int, optional): Minimum width of a hole. Defaults to `max_width`.
        fill_value (int or float): Value for the dropped pixels (holes). This
            should typically be the value representing invalid depth (e.g., 0).
            Defaults to 0.
        prob (float): The probability of applying this transform. Defaults to 0.5.
    """

    def __init__(
        self,
        max_holes: int = 8,
        max_height: int = 8,
        max_width: int = 8,
        min_holes: Optional[int] = None,
        min_height: Optional[int] = None,
        min_width: Optional[int] = None,
        fill_value: Union[int, float] = 0,
        prob: float = 0.5,
    ):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.prob = prob

        assert 0 < self.min_holes <= self.max_holes
        assert 0 < self.min_height <= self.max_height
        assert 0 < self.min_width <= self.max_width
        assert 0.0 <= self.prob <= 1.0

    def transform(self, results: Dict) -> Dict:
        """Apply the CoarseDropout augmentation to the depth map."""
        if random.random() > self.prob:
            return results

        if "depth_map" not in results or results["depth_map"] is None:
            return results

        h, w = results["depth_map"].shape[:2]
        depth_map = results["depth_map"].copy()

        num_holes = random.randint(self.min_holes, self.max_holes)

        for _ in range(num_holes):
            hole_h = random.randint(self.min_height, self.max_height)
            hole_w = random.randint(self.min_width, self.max_width)

            if h - hole_h > 0:
                y1 = random.randint(0, h - hole_h)
            else:
                y1 = 0
            if w - hole_w > 0:
                x1 = random.randint(0, w - hole_w)
            else:
                x1 = 0
            y2 = y1 + hole_h
            x2 = x1 + hole_w

            depth_map[y1:y2, x1:x2] = self.fill_value

        results["depth_map"] = depth_map
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_holes={self.max_holes}, max_height={self.max_height}, "
            f"max_width={self.max_width}, min_holes={self.min_holes}, "
            f"min_height={self.min_height}, min_width={self.min_width}, "
            f"fill_value={self.fill_value}, prob={self.prob})"
        )


@TRANSFORMS.register_module(force=True)
class RandomDepthOffset(BaseTransform):
    """Add a random offset to the entire depth map.

    This transform simulates a global change in depth perception, for example,
    due to sensor calibration drift or changes in lighting conditions that
    affect the entire scene's depth reading.

    Required Keys:
        - depth_map (np.ndarray)

    Modified Keys:
        - depth_map

    Args:
        offset_range (tuple[float, float]): The range from which to sample the
            random offset value (in the same unit as the depth map, e.g., meters).
            The offset will be sampled uniformly from [min_offset, max_offset].
            For example, (-0.1, 0.1) means an offset between -10cm and +10cm
            will be added.
        prob (float): The probability of applying this transform. Defaults to 0.5.
        invalid_depth_value (float): The value in the depth map that represents
            invalid or no-return pixels (e.g., 0.0). These pixels will not
            be modified by the offset. Defaults to 0.0.
        clamp_to_zero (bool): Whether to clamp negative depth values to zero
            after applying the offset. Defaults to True.
    """

    def __init__(
        self,
        offset_range: Tuple[float, float],
        prob: float = 0.5,
        invalid_depth_value: float = 0.0,
        clamp_to_zero: bool = True,
    ):
        assert isinstance(offset_range, tuple) and len(offset_range) == 2, \
            "offset_range must be a tuple of two floats."
        assert offset_range[0] <= offset_range[1], \
            "The first value in offset_range must be less than or equal to the second."
        assert 0.0 <= prob <= 1.0, "prob must be between 0.0 and 1.0."

        self.offset_range = offset_range
        self.prob = prob
        self.invalid_depth_value = invalid_depth_value
        self.clamp_to_zero = clamp_to_zero

    def transform(self, results: Dict) -> Dict:
        """Apply the random depth offset augmentation."""
        if random.random() > self.prob:
            return results

        if "depth_map" not in results or results["depth_map"] is None:
            return results

        depth_map = results["depth_map"].copy()

        # Generate a random offset from the specified range
        random_offset = random.uniform(self.offset_range[0], self.offset_range[1])

        # Create a mask for valid depth pixels (i.e., not the invalid_depth_value)
        valid_mask = (depth_map != self.invalid_depth_value)

        # Add the offset only to the valid pixels
        depth_map[valid_mask] += random_offset

        # Ensure that the depth values do not become negative after adding the offset.
        if self.clamp_to_zero:
            depth_map[depth_map < 0] = 0

        results["depth_map"] = depth_map
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"offset_range={self.offset_range}, "
            f"prob={self.prob}, "
            f"invalid_depth_value={self.invalid_depth_value}, "
            f"clamp_to_zero={self.clamp_to_zero})"
        )
