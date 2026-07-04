# dataset settings
#dataset_type = "DRIVECustomRGBDDataset"
dataset_type = 'DRIVECustomMulticatRGBDDataset'
#data_root = "/home/inaho-omen/data/tomato_drive/tomato_stem_EF_0501-0514_base_KD_05_31_13-14_17-18_add150_val48_2024_ds77c_stem_only_trainval/drive"
#data_root = "/home/inaho-omen/data/tomato_drive/tomato_drive_stem_EF_0501-0514_base_KD_0531_and_0606_add246_val48_2024_ds77c_wi_red_trainval"
#data_root = "/workspace/data/tomato_pipe_bed_semseg/tomato_pipe_bed_segmentation_train84_val"
#data_root = "/workspace/data/tomato_stem_semseg/tomato_stem_EF_0501-0514_base_KD_05_31_13-14_17-18_add150_val48_2024_ds77c_stem_only_trainval_drive_mmseg"
#data_root = "/workspace/data/tomato_stem_semseg/tomato_drive_stem_3-category_EF_0501-0514_base_KD_0531_and_0606_add246_val48_2024_ds77c_wi_red_trainval"
data_root = "/workspace/data/tomato_stem_semseg/tomato_drive_stem_3-category_rgbd_Jun28"
#data_root = "/workspace/data/tomato_stem_semseg/2025_07_09_16_05_37_IR_stem_single"
#data_root = "/workspace/data/tomato_stem_semseg/2025_06_19_15_22_25_RIGHT_30_cropped_drive"
#data_root = "/workspace/data/tomato_stem_semseg/tomato_drive_stem_3-category_rgb_hq"


train_image_dir_relative_path = "train_images_rgb"
train_depth_image_dir_relative_path = "train_images_depth"
train_ann_image_relative_path = "train_masks"
val_image_dir_relative_path = "valid_images_rgb"
val_depth_image_dir_relative_path = "valid_images_depth"
val_ann_image_relative_path = "valid_masks"

IMG_SCALE = (736, 512)
#IMG_SCALE = (640, 480)
#TRAIN_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 56
#TRAIN_BATCH_SIZE = 96



albu_train_weak_transforms = [
    dict(
        type="OneOf",
        transforms=[
            dict(type="RGBShift", r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=1.0),
            dict(type="RandomBrightnessContrast", brightness_limit=[-0.1, 0.1], contrast_limit=[-0.1, 0.1], p=1.0),
        ],
        p=0.5,
    ),
]

albu_train_strong_transforms = [
    dict(type="CLAHE", p=0.25),
    dict(
        type="OneOf",
        transforms=[
            dict(type="RGBShift", r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
            dict(type="HueSaturationValue", hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            dict(type="RandomBrightnessContrast", brightness_limit=[-0.5, 0.5], contrast_limit=[-0.5, 0.5], p=1.0),
            dict(type="IAASharpen", alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ],
        p=0.8,
    ),
    dict(type="JpegCompression", quality_lower=70, quality_upper=95, p=0.2),
    dict(type="ChannelShuffle", p=0.2),
    dict(
        type="OneOf",
        transforms=[
            dict(type="RandomSunFlare", src_radius=100, p=1.0),
            dict(type="RandomShadow", p=1.0),
        ],
        p=0.3,
    ),
    dict(type="Cutout", num_holes=3, max_h_size=30, max_w_size=30, p=0.25),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadDepthImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(type="Resize", scale=crop_size, keep_ratio=False),
    # dict(type="ResizeToMultiple", size_divisor=32),
    dict(type="ResizeRGBD", scale=IMG_SCALE, keep_ratio=False),
    # dict(type="ResizeToMultipleRGBD", size_divisor=32),
    dict(type="Albu", transforms=albu_train_strong_transforms),
    #dict(type="Albu", transforms=albu_train_weak_transforms),
    dict(type="RandomDepthOffset", offset_range=(-1.0,1.0), prob=0.25), 
    #dict(type="Normalize", mean=mean_pix_norm, std=std_pix_norm, to_rgb=bgr_to_rgb),
    #dict(type, CoarseDropoutDepth, RandomDepthOffset="PackSegInputs"),
    dict(type='GenerateEdge', edge_width=4),
    dict(type="PackRGBDSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadDepthImageFromFile"),
    # dict(type="Resize", scale=crop_size, keep_ratio=False),
    # dict(type="ResizeToMultiple", size_divisor=32),
    dict(type="ResizeRGBD", scale=IMG_SCALE, keep_ratio=False),
    # dict(type="ResizeToMultipleRGBD", size_divisor=32),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform (pred_sem_seg and seg_logits will be resized in the same size as the GT image)
    # https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/coco-stuff10k.py
    dict(type="LoadAnnotations"),
    dict(type="PackRGBDSegInputs"),
]

train_dataloader = dict(
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path=train_image_dir_relative_path,
                depth_map_path=train_depth_image_dir_relative_path,
                seg_map_path=train_ann_image_relative_path,
            ),
            pipeline=train_pipeline,
        ),
    ),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=val_image_dir_relative_path,
            depth_map_path=val_depth_image_dir_relative_path,
            seg_map_path=val_ann_image_relative_path,
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader
val_evaluator = dict(type="IoUMetric", iou_metrics=["mDice"])
test_evaluator = val_evaluator
