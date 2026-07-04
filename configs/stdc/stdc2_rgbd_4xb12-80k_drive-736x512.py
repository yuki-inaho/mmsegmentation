_base_ = [
    "../_base_/models/stdc_v2_rgbd.py",
    "../_base_/datasets/drive_rgbd.py",
    "../_base_/default_runtime.py",
]

#NUM_CLASSES = 2
BATCH_SIZE = 64
NUM_CLASSES = 3
BASE_LR = 0.004
DROPOUT_RATIO = 0.0
MAX_ITERS = 1000
SAVE_INTERVAL = 25
VAL_INTERVAL = 25
VIS_INTERVAL = 10

#LOAD_FROM = "models/mmseg/stdc1_in1k-pre_512x1024_80k_cityscapes_20220224_141648-3d4c2981.pth"
LOAD_FROM = "work_dirs/stdc2_rgbd_4xb12-80k_drive-736x512/stdc2-rgbd-3cat_drive-736x512_Apr3-2025_iter_100.pth"
#LOAD_FROM = "work_dirs/stdc2_rgbd_4xb12-80k_drive-736x512/iter_3900.pth"
#LOAD_FROM = "work_dirs/stdc2_rgbd_4xb12-80k_drive-736x512/best_mDice_iter_125.pth"


SEG_LOSS_LIST = [
    dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=1.0, class_weight=[0.5, 1.0, 1.0]),
    #dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=1.0, class_weight=[0.5, 1.0]),
    #dict(type="DiceLoss", loss_name="loss_dice", loss_weight=2.0, use_sigmoid=False),
    #dict(type='LovaszLoss', loss_name='loss_lovasz', loss_weight=1.2, classes=[1,2], reduction="none", loss_type="multi_class"),
    # dict(type="CLDiceLoss", loss_name="loss_cldice", loss_weight=3.0)
    # dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=3.0, alpha=0.5, beta=0.5, class_weight=[0.05, 1.0])
]

STDC_SEG_LOSS_LIST = [
    dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=1.0, class_weight=[0.5, 1.0, 2.0]),
    #dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=1.0, class_weight=[0.5, 1.0]),
    dict(type="DiceLoss", loss_name="loss_dice", loss_weight=2.0, ignore_index=0),
    dict(type='LovaszLoss', loss_name='loss_lovasz', loss_weight=1.2, classes=[1,2], reduction="none", loss_type="multi_class"),
    #dict(type='LovaszLoss', loss_name='loss_lovasz', loss_weight=1.2, classes=[1], reduction="none", loss_type="multi_class"),
    #dict(type='LovaszLoss', loss_name='loss_lovasz', loss_weight=2.5, reduction='none'),
    # dict(type="CLDiceLoss", loss_name="loss_cldice", loss_weight=0.25)
    # dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=3.0, alpha=0.5, beta=0.5, class_weight=[0.05, 1.0])
]

norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            act_cfg=dict(type="ReLU"),
            # act_cfg=dict(type="SiLU"),
        ),
    ),
    decode_head=dict(
        num_classes=NUM_CLASSES,
        dropout_ratio=DROPOUT_RATIO,
        loss_decode=SEG_LOSS_LIST,
    ),
    auxiliary_head=[
        dict(
            type="FCNHead",
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=NUM_CLASSES,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=10000),
            loss_decode=SEG_LOSS_LIST,
        ),
        dict(
            type="FCNHead",
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=NUM_CLASSES,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=10000),
            loss_decode=SEG_LOSS_LIST,
        ),
        dict(
            type="STDCHead",
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=NUM_CLASSES,
            boundary_threshold=0.1,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=STDC_SEG_LOSS_LIST,
        ),
    ],
)

in_dataloader = dict(batch_size=BATCH_SIZE, num_workers=BATCH_SIZE * 2)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=MAX_ITERS, val_interval=VAL_INTERVAL)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


# optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.0001),  # 0.0002 for DeformDETR
# optimizer = dict(type='SGD', lr=BASE_LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
optimizer = dict(type="AdaBelief", lr=BASE_LR, eps=1e-12, betas=(0.9, 0.999), weight_decay=0.005)
# optimizer = dict(type="SAMSGD", lr=BASE_LR, momentum=0.9, weight_decay=0.005, nesterov=True)  # @SAM
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
    # paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# learning policy
param_scheduler = [
    dict(type="LinearLR", by_epoch=False, start_factor=1.0e-5, begin=0, end=MAX_ITERS // 10),
    dict(
        type="CosineAnnealingLR",
        eta_min=BASE_LR * 0.05,
        begin=MAX_ITERS // 2,
        end=MAX_ITERS,
        T_max=MAX_ITERS // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]


# training schedule for 40k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=MAX_ITERS, val_interval=VAL_INTERVAL)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    #    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        save_best="mDice",
        rule="greater",
        by_epoch=False,
        interval=SAVE_INTERVAL,
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=True, interval=VIS_INTERVAL),
)

val_evaluator = dict(type="IoUMetric", iou_metrics=["mDice"])
test_evaluator = dict(type="IoUMetric", iou_metrics=["mDice"])


optimizer_config=dict(type="GradientCumulativeOptimizerHook", cumulative_iters=2)
fp16 = dict(loss_scale="dynamic")
custom_hooks = [
    dict(type="mmdet.EMAHook", ema_type="mmdet.ExpMomentumEMA", momentum=0.0002, update_buffers=True, priority=49)
]

load_from = LOAD_FROM
