def get_model_cfg(config):
    if config["model"].lower() == "segformer":
        norm_cfg = dict(type='BN', requires_grad=True)
        model_cfg = dict(
            type='EncoderDecoder',
            pretrained=None,
            backbone=dict(
                type='MixVisionTransformer',
                in_channels=3,
                embed_dims=64,
                num_stages=4,
                num_layers=[3, 8, 27, 3],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.,
                init_cfg = dict(type="Pretrained", checkpoint="/home/pervinco/BKAI-IGH-NeoPolyp/runs/858/weights/best_dice_ckpt_ep_93.pth")), ## "./pretrained_models/mit_b4_mmseg.pth"
            decode_head=dict(
                type='SegformerHead',
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                channels=768, ## 256 
                dropout_ratio=0.3,
                num_classes=config["num_classes"],
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            train_cfg=dict())
        
    elif config["model"].lower() == "swin_transformer_384":
        norm_cfg = dict(type='BN', requires_grad=True)
        backbone_norm_cfg = dict(type='LN', requires_grad=True)
        checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # noqa
        model_cfg = dict(
            type='EncoderDecoder',
            pretrained=None,
            backbone=dict(
                type='SwinTransformer',
                pretrain_img_size=224,
                embed_dims=192,
                patch_size=4,
                mlp_ratio=4,
                window_size=12,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                norm_cfg=backbone_norm_cfg,
                init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
            decode_head=dict(
                type='UPerHead',
                in_channels=[192, 384, 768, 1536],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=config["num_classes"],
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            # auxiliary_head=dict(
            #     type='FCNHead',
            #     in_channels=768,
            #     in_index=2,
            #     channels=256,
            #     num_convs=1,
            #     concat_input=False,
            #     dropout_ratio=0.1,
            #     num_classes=config["num_classes"],
            #     norm_cfg=norm_cfg,
            #     align_corners=False,
            #     loss_decode=dict(
            #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(mode='whole'))


    elif config["model"].lower() == "swin_transformer_512":
        norm_cfg = dict(type='BN', requires_grad=True)
        backbone_norm_cfg = dict(type='LN', requires_grad=True)
        checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'  # noqa
        model_cfg = dict(
            type='EncoderDecoder',
            pretrained=None,
            backbone=dict(
                type='SwinTransformer',
                pretrain_img_size=224,
                embed_dims=192,
                patch_size=4,
                mlp_ratio=4,
                window_size=12,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                norm_cfg=backbone_norm_cfg,
                init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
            decode_head=dict(
                type='UPerHead',
                in_channels=[192, 384, 768, 1536],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=config["num_classes"],
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            # auxiliary_head=dict(
            #     type='FCNHead',
            #     in_channels=768,
            #     in_index=2,
            #     channels=256,
            #     num_convs=1,
            #     concat_input=False,
            #     dropout_ratio=0.1,
            #     num_classes=config["num_classes"],
            #     norm_cfg=norm_cfg,
            #     align_corners=False,
            #     loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(mode='whole'))
        
    elif config["model"].lower() == "metaformer":
        norm_cfg = dict(type='BN', requires_grad=True)
        model_cfg = dict(
            type='EncoderDecoder',
            backbone=dict(
                type='mmpretrain.PoolFormer',
                arch='m48',
                init_cfg=dict(type='Pretrained', checkpoint="./pretrained_models/fpn_poolformer_m48_8x4_512x512_40k_ade20k_20220504_003923-64168d3b.pth", prefix='backbone.'),
                in_patch_size=7,
                in_stride=4,
                in_pad=2,
                down_patch_size=3,
                down_stride=2,
                down_pad=1,
                drop_rate=0.,
                drop_path_rate=0.,
                out_indices=(0, 2, 4, 6),
                frozen_stages=0)
                ,
            neck=dict(
                type='FPN',
                in_channels=[96, 192, 384, 768],
                out_channels=256,
                num_outs=4),

            decode_head=dict(
                type='FPNHead',
                in_channels=[256, 256, 256, 256],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.,
                num_classes=config["num_classes"],
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(mode='whole'))


        
    return model_cfg
