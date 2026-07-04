_base_ = './unet-s5-d16_deeplabv3_4xb4-40k_coco-64x64.py'
model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.05, 1.0]),
        #dict(type='DiceLossCustom', loss_name='loss_dice_custom', loss_weight=3.0)
        dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=3.0, alpha=0.5, beta=0.5, class_weight=[0.05, 1.0])
    ]))

#load_from = "checkpoints/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201825-6bf0efd7.pth"
load_from = "checkpoints/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth"
#load_from = "work_dirs/unet-s5-d16_deeplabv3_4xb4-ce-1.0-dice-3.0-40k_coco-64x64/iter_330.pth"
#load_from = "work_dirs/unet-s5-d16_deeplabv3_4xb4-ce-1.0-dice-3.0-40k_coco-64x64/iter_1000.pth"
