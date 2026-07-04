_base_ = "./ocrnet_hr18_4xb4-20k_voc12aug-640x384.py"
model = dict(
    # pretrained='open-mmlab://msra/hrnetv2_w18_small',
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2,)),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)),
        )
    )
)


# https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18s_512x512_20k_voc12aug/ocrnet_hr18s_512x512_20k_voc12aug_20200617_233913-02b04fcb.pth
load_from = "checkpoints/ocrnet_hr18s_512x512_20k_voc12aug_20200617_233913-02b04fcb.pth"
