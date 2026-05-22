# -*- coding:utf-8 -*-
"""
 @FileName   : teacher_model.py
 @Time       : 9/24/24 3:25 PM
 @Author     : Woldier Wong
 @Description: TODO
"""
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy

import numpy as np
import torch
from timm.models.layers import DropPath
from torch.nn import Module
from torch.nn.modules.dropout import _DropoutNd
from mmseg.ops import resize
from mmseg.models import build_segmentor
from mmseg.models.uda.uda_decorator import get_module


class EMATeacher(Module):

    def __init__(
            self,
            # use_mask_params,
            model=None,
            **cfg):
        super(EMATeacher, self).__init__()
        # 构建模型
        ema_cfg = deepcopy(model)
        self.ema_model = build_segmentor(ema_cfg)
        self._init_param(**cfg)
        # self.debug = False
        # self.debug_output = {}

        for p in self.get_ema_model().parameters():
            p.requires_grad = False
        # eval mod
        self.get_ema_model().eval()

    def _init_param(
            self,
            alpha=None,
            pseudo_threshold=None, pseudo_weight_ignore_top=None, pseudo_weight_ignore_bottom=None, **kwargs):
        self.alpha = alpha
        self.pseudo_threshold = pseudo_threshold
        self.psweight_ignore_top = pseudo_weight_ignore_top
        self.psweight_ignore_bottom = pseudo_weight_ignore_bottom

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self, model):
        """
        初始化ema的权重
        """
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        if self.pseudo_threshold is not None:
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=logits.device)
        else:
            pseudo_weight = torch.ones(pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if iter > 0:
            self._update_ema(model, iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

    def __call__(self, target_img, target_img_metas, valid_pseudo_mask):
        # self.update_debug_state()

        # Generate pseudo-label
        ema_logits = self.pre(target_img, target_img_metas)

        pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
            ema_logits)
        del ema_logits

        pseudo_weight = self.filter_valid_pseudo_region(
            pseudo_weight, valid_pseudo_mask)

        # self.debug_output = self.ema_model.debug_output

        return pseudo_label, pseudo_weight

    def pre(self, target_img, target_img_metas, return_feature=False):
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        # ema_logits = self.get_ema_model().generate_pseudo_label(
        #     target_img, target_img_metas)
        # ema_logits = self.get_ema_model().encode_decode(
        #     target_img, target_img_metas)
        x = self.get_ema_model().extract_feat(target_img)
        out = self.get_ema_model()._decode_head_forward_test(x, target_img_metas)
        ema_logics = resize(
            input=out,
            size=target_img.shape[2:],
            mode='bilinear',
            align_corners=self.get_ema_model().align_corners)
        if return_feature:
            return ema_logics, x
        return ema_logics

    def logics(self, target_img, target_img_metas, return_feature=False):
        return self.pre(target_img, target_img_metas, return_feature=return_feature)


class EMATeacherWithUncertainty(EMATeacher):

    def __init__(self, model=None, **cfg):
        super().__init__(model, **cfg)

    def uncertainty_estimate(self, img, img_metas, shot=5):
        self.get_ema_model().train()  # 适合有  MC Dropout 的方法
        ema_logics_list, ema_feat_list = [], []
        for _ in range(shot):
            _ema_logics, _ema_feat = self.logics(img, img_metas, return_feature=True)
            ema_logics_list.append(_ema_logics.unsqueeze(1))  # [b, 1, c, h, w]
            ema_feat_list.append(_ema_feat.unsqueeze(1)) if not  isinstance(_ema_feat, list) \
                else ema_feat_list.append(_ema_feat[-1].unsqueeze(1))
        ema_logics = torch.cat(ema_logics_list, dim=1)  # [b, shot, c, h, w]
        ema_feat = torch.cat(ema_feat_list, dim=1)  # [b, shot, f, h, w]
        uncertainty = torch.var(ema_logics, dim=1, keepdim=False)  # [b, c, h, w]
        ema_logics = torch.mean(ema_logics, dim=1, keepdim=False)
        self.get_ema_model().eval()
        return ema_logics, uncertainty
