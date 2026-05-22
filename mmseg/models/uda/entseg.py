# -*- coding:utf-8 -*-
"""
 @FileName   : label_denoise.py
 @Time       : 12/15/24 2:41 PM
 @Author     : Woldier Wong
 @Description: TODO
"""
import random, torch, os, torch.nn.functional as F, numpy as np
from mmseg.models import UDA
from .uda_with_teacher import UDAWithTeacher

from mmcv.runner import master_only
from mmseg.models.losses.cross_entropy_loss import cross_entropy

from mmseg.models.utils.masking_transforms import build_mask_generator

from mmseg.ops import resize


def _unwrap_kwarg(gt_semantic_seg, target_img, target_img_metas, rcs_img, rcs_img_gt_seg, rcs_img_metas, rcs_img_cla,
                  **kwargs):
    return gt_semantic_seg, target_img, target_img_metas, rcs_img, rcs_img_gt_seg, rcs_img_metas, rcs_img_cla,


def _unwrap_feat(feat: list, need_total=False):
    if not need_total:
        return feat[-1]
    return torch.cat(feat, dim=0)


def _compute_uncertainty(logits):
    # logits: [B, C, H, W] -> softmax probabilities
    probs = F.softmax(logits, dim=1)
    # Compute pixel-wise entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)  # [B, H, W]
    return entropy


def _mse(pre, gt):
    return F.mse_loss(pre, gt)


def negative_loss(prob, width=1, num_classes=6):
    # width = 1  # change to 1
    k = num_classes // 2 + random.randint(-width, width)
    _, labels_neg = torch.topk(prob, k, dim=1, sorted=True)
    s_neg = torch.log(torch.clamp(1. - prob, min=1e-5, max=1.))
    labels_neg = labels_neg[:, -1].squeeze().detach()
    loss_neg = F.nll_loss(s_neg, labels_neg)
    return loss_neg


@UDA.register_module()
class EntSeg(UDAWithTeacher):
    def __init__(self,
                 **cfg):
        """
        :param cfg:

        blur (bool): 在class mix 时是否使用 blur
        color_jitter_strength: color_jitter 的 参数, 支持float或者dict
            当s为float时
            brightness=s, contrast=s, saturation=s, hue=s
            当为dict时
            brightness=s["brightness"], contrast=s["contrast"], saturation=s["saturation"], hue=s["saturation"]
        color_jitter_probability: Thresholding with color_jitter. 采用 color_jitter 的阈值
        """
        super().__init__(**cfg)
        # ================ param init ==================================
        self._init_cla_mix_param(**cfg)

        self.mask_trans = build_mask_generator(dict(type='block', mask_ratio=0.5, mask_block_size=32))

    def _init_cla_mix_param(
            self,
            blur=None, color_jitter_strength=.25, color_jitter_probability=.2,
            debug_img_interval=None,
            **kwargs
    ):
        self.blur = blur
        self.color_jitter_s = color_jitter_strength
        self.color_jitter_p = color_jitter_probability
        self.debug_img_interval = debug_img_interval

    def _forward_setup(self, img, img_metas):
        """
        初始化/update Teacher model 以及定义一些 训练所需的变量
        :param img:
        :param img_metas:
        :return:
        """
        batch_size, dev, log_vars = super()._forward_setup(img)
        means, stds = self._get_mean_std(img_metas, dev)
        return batch_size, dev, log_vars, means, stds

    def forward_train(self, img, img_metas, **kwargs):
        batch_size, dev, log_vars, means, stds = self._forward_setup(img, img_metas)
        gt_semantic_seg, target_img, target_img_metas, \
            rcs_img, rcs_img_gt_seg, rcs_img_metas, rcs_img_cla = _unwrap_kwarg(**kwargs)
        #  ============================= EMA predict ============================================
        ema_tgt_logics, ema_tgt_feat = self.get_ema_model().logics(target_img, target_img_metas, return_feature=True)
        pseudo_label, weight = self.get_ema_model().get_pseudo_label_and_weight(ema_tgt_logics)
        pseudo_label = pseudo_label.unsqueeze(1)
        ema_tgt_uncertainty = _compute_uncertainty(ema_tgt_logics) / torch.log(
            torch.as_tensor(self.num_classes, device=dev))
        weight = 1 - ema_tgt_uncertainty
        # ================================ Source Train =========================================
        src_feature, src_logics, src_pre = self._forward_feat_logic_pre(img, img_metas)
        loss_sce = cross_entropy(src_logics, gt_semantic_seg.squeeze(1), ignore_index=255)
        self._process_loss(log_vars, loss_sce=loss_sce)

        # ================================ Target Train =========================================

        ## 1. Cla mix
        strong_parameters = self._prepare_strong_transform_param(means, stds)
        # 用原始的src img
        mix_masks, mixed_img, mixed_lbl, mixed_seg_weight = \
            self._img_mix(rcs_img, rcs_img_gt_seg,
                          target_img, pseudo_label, weight,
                          batch_size, strong_parameters)  # , rcs_cla=rcs_img_cla)
        mixed_feature, mixed_logics, mixed_pre = self._forward_feat_logic_pre(mixed_img, target_img_metas)
        loss_mix = cross_entropy(mixed_logics, mixed_lbl.squeeze(1), weight=mixed_seg_weight, ignore_index=255)
        self._process_loss(log_vars, loss_mix=loss_mix)

        ## 3. Mask Image Modeling
        b, h, w = ema_tgt_uncertainty.shape
        mask_block_size = 16  # TODO 试试 32 64
        mshape = b, 1, round(h / mask_block_size), round(w / mask_block_size)
        weight_low_reso = resize(weight.unsqueeze(1), size=mshape[2:], mode='bilinear',
                                 align_corners=self.get_model().align_corners)
        weight_flattened = weight_low_reso.view(b, -1)
        sorted_indices = torch.argsort(weight_flattened)
        num_pixels = mshape[-1] * mshape[-2]
        num_to_mask = int(num_pixels * 0.5)  # TODO 试试 修改比率
        mask_indices = sorted_indices[:, :num_to_mask]
        mask = torch.ones_like(weight_flattened, dtype=torch.bool)
        mask = mask.scatter_(dim=1, index=mask_indices, value=False)
        mask = mask.view(*mshape)
        mask = resize(mask.float(), size=(h, w))
        mask_img = mask * target_img
        mask_feat, mask_logics, mask_pre = self._forward_feat_logic_pre(mask_img, target_img_metas)
        loss_mask = cross_entropy(mask_logics, pseudo_label.squeeze(1), weight=weight, ignore_index=255)
        self._process_loss(log_vars, loss_mask=0.5 * loss_mask)
        self._show_img(
            img=img, gt_semantic_seg=gt_semantic_seg,
            target_img=target_img, pseudo_label=pseudo_label, weight=weight,  # tgt_pre,
            rcs_img=rcs_img, rcs_img_gt_seg=rcs_img_gt_seg,
            mixed_img=mixed_img, mixed_lbl=mixed_lbl, mixed_seg_weight=mixed_seg_weight.unsqueeze(1),
            mixed_masks=torch.cat(mix_masks), mixed_pre=mixed_pre,
            mask_img=mask_img, mask_pre=mask_pre,
            batch_size=batch_size, means=means, stds=stds)
        return log_vars

    @staticmethod
    def _img_mix(img, gt_semantic_seg,
                 target_img, pseudo_label, pseudo_weight,
                 batch_size, strong_parameters, rcs_cla=None):
        from ..utils.dacs_transform import get_class_masks, strong_transform
        gt_pixel_weight = torch.ones_like(pseudo_weight)  # 对于source, 其损失权重总是为1
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mixed_seg_weight = pseudo_weight.clone()
        mix_masks = get_class_masks(gt_semantic_seg, rcs_cla=rcs_cla)
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i][0])))
            _, mixed_seg_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], mixed_seg_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        return mix_masks, mixed_img, mixed_lbl, mixed_seg_weight

    def _prepare_strong_transform_param(self, means, stds):
        """
        img_metas: 图片的元数据
        dev: 设备id
        """

        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        return strong_parameters

    @master_only
    def _show_img(self,
                  img, gt_semantic_seg,
                  target_img, pseudo_label, weight,
                  rcs_img, rcs_img_gt_seg,
                  mixed_img, mixed_lbl, mixed_seg_weight, mixed_masks, mixed_pre,
                  mask_img, mask_pre,
                  batch_size, means, stds,
                  ):
        if self.iter % 200 == 0:
            from mmseg.models.utils.visualization import show_img, VisDataContainer as VDC, DataType
            out_dir = os.path.join(self.work_dir, 'debug_img')
            out_file = os.path.join(out_dir, f'iter_{int(self.iter):08d}.png')
            data_list = [  # 使用的方式是一个样本一张图
                # 第一行的数据
                [
                    VDC('Source Image', img, DataType.IMG),
                    VDC('Source Seg GT', gt_semantic_seg, DataType.SEG),
                    VDC('', None, DataType.NONE),
                    VDC('RCS Image', rcs_img, DataType.IMG),
                    VDC('RCS Seg GT', rcs_img_gt_seg, DataType.SEG),
                ],
                [
                    VDC('Target Image', target_img, DataType.IMG),
                    VDC('Target Pseudo', pseudo_label, DataType.SEG),
                    VDC('Target Seg. Weight', weight, DataType.WEIGHT),
                    VDC('Mask Image', mask_img, DataType.IMG),
                    VDC('Mask Image Pre.', mask_pre, DataType.SEG),

                ],
                [
                    VDC('Mixed Image', mixed_img, DataType.IMG),
                    VDC('Mixed Image Pre.', mixed_pre, DataType.SEG),
                    VDC('Mixed Image GT', mixed_lbl, DataType.SEG),
                    VDC('Mixed Image Seg. Weight', mixed_seg_weight, DataType.WEIGHT),
                    VDC('Mixed Image Mask', mixed_masks, DataType.WEIGHT),
                ],

            ]
            show_img(data_list, batch_size, means, stds, self.cmap, out_file)
