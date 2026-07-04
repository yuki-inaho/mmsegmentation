# custom_imports = dict(imports=["mmseg.engine.optimizers.adabelief"], allow_failed_imports=False)

# _base_ = ["../_base_/models/deeplabv3_unet_s5-d16.py", "../_base_/datasets/coco-custom.py", "../_base_/default_runtime.py"]
# _base_ = ["../_base_/models/deeplabv3_unet_s5-d16.py", "../_base_/datasets/drive.py", "../_base_/default_runtime.py"]
_base_ = ["../_base_/models/deeplabv3_unet_s5-d16.py", "../_base_/datasets/drive.py", "../_base_/default_runtime.py"]

# optimizer
# optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, clip_grad=None)

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
#fp16 = dict(loss_scale=512.0)
#optimizer_config = dict(type="GradientCumulativeFp16OptimizerHook", cumulative_iters=4)
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)

# crop_size = (512, 512)
# data_preprocessor = dict(size=crop_size)
# model = dict(data_preprocessor=data_preprocessor, test_cfg=dict(crop_size=crop_size, stride=(10, 10)))
# load_from = "checkpoints/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201825-6bf0efd7.pth"
