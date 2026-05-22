# -*- coding:utf-8 -*-
"""
 @FileName   : entropy_loss.py
 @Time       : 11/29/24 6:29 PM
 @Author     : Woldier Wong
 @Description: TODO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class EntropyLoss(nn.Module):
    def __init__(
            self,
            reduction='mean',
            loss_weight=1.0,
            loss_name='loss_ent'
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        loss = entropy_loss(cls_score).mean()
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


def entropy_loss(cls_score):
    p = F.softmax(cls_score, dim=1)
    log_p = F.log_softmax(p, dim=1)
    loss = -torch.sum(p * log_p, dim=1)
    return loss
