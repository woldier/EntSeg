# -*- coding:utf-8 -*-
_base_ = [
    '../_base_/models/daformer_sepaspp_mit-b5.py',
    '../_base_/datasets/uda_potsdamIRRG_2_vaihingenIRRG_512x512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/poly_schedule_40k.py',
    '../_base_/schedules/adamw.py',
]
data = dict(
    samples_per_gpu=6, workers_per_gpu=3,
    train=dict(
        type='UDARCSDataset',
        data_strategy='source',
        rcs_cfg=dict()
    )
)

# learning policy
lr_config = dict(
    warmup='linear', warmup_iters=1500,
    warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)
optimizer_config = None
optimizer = dict(
    type='AdamW', lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0), )
    )
)
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='mit_b5'),
    decode_head=dict(num_classes=6),
)
uda = dict(
    type='EntSeg',
    alpha=0.99,
    # teacher model cfg
    pseudo_threshold=0.98,  # 0.968
    pseudo_weight_ignore_top=0, pseudo_weight_ignore_bottom=0,
)

work_dir = r'./result/{{dateSplitDir}}/{{timePrefix}}_{{fileBasenameNoExtension}}'
# work_dir = r'./result/{{dateSplitDir}}/debug_{{fileBasenameNoExtension}}'
use_ddp_wrapper = True

checkpoint_config = dict(by_epoch=False, interval=200)
evaluation = dict(interval=200, metric='mIoU', pre_eval=True)
