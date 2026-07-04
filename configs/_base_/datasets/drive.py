# dataset settings
# dataset_type = "DRIVEDataset"
# data_root = "data/DRIVE_MY"

dataset_type = "DRIVECustomDataset"
data_root = "/home/inaho-omen/data/annotation_tomato_stem/20230612_TVA_tomato-03-720p-sub15/for_mmseg_drive"
# data_root = "/home/kasm-user/Desktop/workspace/data/tomato_stem_seg/KD_0530_TVA_0609_0612_sum35"
#dataset_type = "DRIVECustomRGBDDataset"  # @DEBUG
#data_root = "/home/inaho-omen/Project/tomato_stem_mmsegmentation/data/test_rgbd"  # @DEBUG

TRAIN_BATCH_SIZE = 2
img_scale = (1280, 720)
# random_choice_scales = [(448, 256), (512, 288), (672, 384), (896, 512), (1056, 608), (1280, 736)]  # (W, H)
# random_choice_scales = [(448, 256), (512, 288), (672, 384), (896, 512)]  # (W, H)
random_choice_scales = [(448, 256), (512, 288), (672, 384)]  # (W, H)
img_scale_list_tta = [(448, 256), (512, 288)]  # (W, H)
# random_choice_scales = [(448, 256), (512, 288)]  # (W, H)

# crop_size = (512, 288)  # @LOCAL
# crop_size = (640, 360)
crop_size = (672, 384)

albu_train_transforms = [
    # dict(type="RandomResizedCrop", width=crop_size[0], height=crop_size[1], scale=(0.25, 0.5)),
    # dict(type="HorizontalFlip", p=0.5),
    # dict(type="VerticalFlip", p=0.5),
    dict(
        type="OneOf",
        transforms=[
            dict(type="RGBShift", r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
            dict(type="HueSaturationValue", hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ],
        p=0.3,
    ),
    dict(type="JpegCompression", quality_lower=90, quality_upper=95, p=0.2),
    dict(type="ChannelShuffle", p=0.5),
    # dict(type="RandomBrightnessContrast", brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=1.0),
    # dict(
    #    type="OneOf",
    #    transforms=[
    #        dict(type="RandomBrightnessContrast", brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=1.0),
    #        #dict(type="RGBShift", p=1.0),
    #        #dict(type="RandomGamma", p=1.0),
    #    ],
    #    p=0.3,
    # )
    # dict(type="CenterCrop", height=crop_size[0], width=crop_size[1], always_apply=True),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    #dict(type="LoadDepthImageFromFile"),  # @DEBUG
    dict(type="LoadAnnotations"),
    #dict(type="Resize", scale=crop_size, keep_ratio=False),
    #dict(type="ResizeToMultiple", size_divisor=32),
    #dict(type="ResizeRGBD", scale=crop_size, keep_ratio=False),  # @DEBUG
    #dict(type="ResizeToMultipleRGBD", size_divisor=32),  # @DEBUG
    # dict(type="RandomResizedCrop", width=crop_size[0], height=crop_size[1], scale=(0.25, 0.5)),
    dict(type="RandomResize", scale=img_scale, ratio_range=(0.25, 0.75), keep_ratio=True),
    dict(type="ResizeToMultiple", size_divisor=32),
    # dict(type="RandomChoiceResize", scales=random_choice_scales),
    dict(type="Albu", transforms=albu_train_transforms),
    #dict(type="Normalize", mean=mean_pix_norm, std=std_pix_norm, to_rgb=bgr_to_rgb),
    dict(type="PackSegInputs"),
    # dict(type="PackRGBDSegInputs"),  # @DEBUG
    # dict(type="PackSegInputsPseudo4ch"),  # @DEBUG
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    #dict(type="LoadDepthImageFromFile"),  # @DEBUG
    dict(type="Resize", scale=crop_size, keep_ratio=False),
    dict(type="ResizeToMultiple", size_divisor=32),
    #dict(type="ResizeRGBD", scale=crop_size, keep_ratio=False),  # @DEBUG
    #dict(type="ResizeToMultipleRGBD", size_divisor=32),  # @DEBUG
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform (pred_sem_seg and seg_logits will be resized in the same size as the GT image)
    # https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/coco-stuff10k.py
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
    #dict(type="PackRGBDSegInputs"),  # @DEBUG
    # dict(type="PackSegInputsPseudo4ch"),  # @DEBUG
]

tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadDepthImageFromFile"),  # @DEBUG
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale=crop_size, keep_ratio=False)],
            [dict(type="ResizeToMultiple", size_divisor=32)],
            #[dict(type="ResizeRGBD", scale=crop_size, keep_ratio=False)],  # @DEBUG
            #[dict(type="ResizeToMultipleRGBD", size_divisor=32)],  # @DEBUG
            [dict(type="LoadAnnotations")],
            [dict(type="PackSegInputs")],
            # [dict(type="PackRGBDSegInputs")],  # @DEBUG
            # [dict(type="PackSegInputsPseudo4ch")],  # @DEBUG
        ],
    ),
]
train_dataloader = dict(
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=TRAIN_BATCH_SIZE,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path="images/training", seg_map_path="annotations/training"),
            #data_prefix=dict(img_path="images_rgb", depth_map_path="images_depth", seg_map_path="annotations"),  # @DEBUG
            pipeline=train_pipeline,
        ),
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
        data_prefix=dict(img_path="images/validation", seg_map_path="annotations/validation"),
        #data_prefix=dict(img_path="images_rgb", depth_map_path="images_depth", seg_map_path="annotations"),  # @DEBUG
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mDice"])
test_evaluator = val_evaluator
