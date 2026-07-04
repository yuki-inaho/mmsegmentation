# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import is_str


def cococustom_classes():
    """CocoCustom class names for external use."""
    return ["stem"]


def cococustom_palette():
    """CocoCustom palette for external use."""
    return [[255, 255, 43]]


dataset_aliases = {"cococustom": ["cococustom"]}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + "_classes()")
        else:
            raise ValueError(f"Unrecognized dataset: {dataset}")
    else:
        raise TypeError(f"dataset must a str, but got {type(dataset)}")
    return labels


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + "_palette()")
        else:
            raise ValueError(f"Unrecognized dataset: {dataset}")
    else:
        raise TypeError(f"dataset must a str, but got {type(dataset)}")
    return labels
