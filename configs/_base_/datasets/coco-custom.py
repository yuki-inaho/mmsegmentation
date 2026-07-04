# dataset settings
dataset_type = "COCOCustomDataset"

data_root = "/home/inaho-omen/data/annotation_tomato_stem/20230612_TVA_tomato-03-720p-sub15/for_mmseg"
#crop_size = (512, 512)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="Resize", scale=(512, 288)),
    #dict(type="Pad", pad_to_square=True, pad_val=dict(img=(0.0, 0.0, 0.0))),
    # dict(type='RandomResize', scale=(1280, 720), ratio_range=(0.8, 1.5)),
    #dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Pad", size_divisor=32),  # (640 x 360) -> (640 x 384)
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    #dict(type="Resize", scale=(640, 360)),
    dict(type="Resize", scale=(512, 288)),
    dict(type="Pad", size_divisor=32),  # (640 x 360) -> (640 x 384)
    #dict(type="Pad", pad_to_square=True, pad_val=dict(img=(0.0, 0.0, 0.0))),
    #dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    #dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(img_path="images", seg_map_path="train/images"),
        ann_file="train/annotations/train.txt",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(img_path="images", seg_map_path="valid/images"),
        ann_file="valid/annotations/valid.txt",
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
