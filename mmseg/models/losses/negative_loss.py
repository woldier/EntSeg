# -*- coding:utf-8 -*-
"""
 @FileName   : negative_loss.py
 @Time       : 12/2/24 1:49 PM
 @Author     : Woldier Wong
 @Description: TODO
"""
import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class NegativeLoss(nn.NLLLoss):
    def __init__(
            self,
            weight: Optional[Tensor] = None,
            size_average=None,
            ignore_index: int = -100,
            reduce=None, reduction: str = 'mean',
            loss_weight=1.0,
            loss_name='loss_neg'
    ):
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                # weight=None,
                # avg_factor=None,
                # reduction_override=None,
                **kwargs):
        loss = F.nll_loss(cls_score, label, weight=self.weight, ignore_index=self.ignore_index,
                          reduction=self.reduction)
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
