_base_ = [
    "../_base_/models/ocrnet_hr18.py",
    "../_base_/datasets/drive.py",
    "../_base_/default_runtime.py",
]

loss_decode_fch = [
    dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=0.4, class_weight=[0.05, 1.0]),
    dict(type="TverskyLoss", loss_name="loss_tversky", loss_weight=1.2, alpha=0.5, beta=0.5, class_weight=[0.05, 1.0]),
]
loss_decode_ocr = [
    dict(type="CrossEntropyLoss", loss_name="loss_ce", loss_weight=1.0, class_weight=[0.05, 1.0]),
    dict(type="TverskyLoss", loss_name="loss_tversky", loss_weight=3.0, alpha=0.5, beta=0.5, class_weight=[0.05, 1.0]),
]


# @TODO: removed duplicated definition
# crop_size = (640, 384)
# data_preprocessor = dict(
#    size=crop_size
# )
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    # data_preprocessor=data_preprocessor,
    decode_head=[
        dict(
            type="FCNHead",
            in_channels=[18, 36, 72, 144],
            channels=sum([18, 36, 72, 144]),
            in_index=(0, 1, 2, 3),
            input_transform="resize_concat",
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            # num_classes=21,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            #loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
            loss_decode=loss_decode_fch,
        ),
        dict(
            type="OCRHead",
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            input_transform="resize_concat",
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            # num_classes=21,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            #loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_decode=loss_decode_ocr,
        ),
    ],
)


base_lr = 1e-2
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.0001),  # 0.0002 for DeformDETR
    clip_grad=dict(max_norm=0.1, norm_type=2)
    # paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)

# learning policy
param_scheduler = [dict(type="PolyLR", eta_min=1e-3, power=0.95, begin=0, end=1000, by_epoch=False)]

# training schedule for 40k

train_cfg = dict(type="IterBasedTrainLoop", max_iters=1000, val_interval=10)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    #    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=True, interval=1),
)

# fp16 settings
# fp16 = dict()
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')

# fp16 placeholder
# fp16 = dict(loss_scale=512.0)
# optimizer_config = dict(type="GradientCumulativeFp16OptimizerHook", cumulative_iters=4)
# optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
