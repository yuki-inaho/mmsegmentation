_base_ = './unet-s5-d16_pspnet_4xb4-40k_drive-64x64.py'
model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='DiceLossCustom', loss_name='loss_dice_custom', loss_weight=3.0, naive_dice=True)
    ]))
