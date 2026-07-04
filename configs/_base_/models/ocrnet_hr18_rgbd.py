# model settings

DROPOUT_RATIO = -1

mean_pix_norm = [123.675, 116.28, 103.53, 0.0]
std_pix_norm = [58.395, 57.12, 57.375, 1.0]
bgr_to_rgb = False

# mean_pix_norm = [123.675, 116.28, 103.53, 0.0]  # @DEBUG
# std_pix_norm = [58.395, 57.12, 57.375, 1.0]  # @DEBUG
# bgr_to_rgb = False  # @DEBUG

norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor", mean=mean_pix_norm, std=std_pix_norm, bgr_to_rgb=bgr_to_rgb, pad_val=0, seg_pad_val=255, size_divisor=32
)  # @CUSTUMED
model = dict(
    type="CascadeEncoderDecoder",
    data_preprocessor=data_preprocessor,
    num_stages=2,
    pretrained="open-mmlab://msra/hrnetv2_w18",
    backbone=dict(
        type="HRNet",
        # in_channels=3,
        in_channels=4,
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1, block="BOTTLENECK", num_blocks=(4,), num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2, block="BASIC", num_blocks=(4, 4), num_channels=(18, 36)),
            stage3=dict(num_modules=4, num_branches=3, block="BASIC", num_blocks=(4, 4, 4), num_channels=(18, 36, 72)),
            stage4=dict(num_modules=3, num_branches=4, block="BASIC", num_blocks=(4, 4, 4, 4), num_channels=(18, 36, 72, 144)),
        ),
    ),
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
            dropout_ratio=DROPOUT_RATIO,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
        ),
        dict(
            type="OCRHead",
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            input_transform="resize_concat",
            channels=512,
            ocr_channels=256,
            dropout_ratio=DROPOUT_RATIO,
            # dropout_ratio=0.01,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        ),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
