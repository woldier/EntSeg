# -*- coding:utf-8 -*-
"""
 @FileName   : uda_with_teacher.py
 @Time       : 10/24/24 8:54 PM
 @Author     : Woldier Wong
 @Description: TODO
"""
import torch
from mmseg.models.uda.uda_decorator import UDADecorator
from mmseg.models.uda.teacher_model import EMATeacher
from mmseg.utils.img_utils import get_mean_std


class UDAWithTeacher(UDADecorator):
    def __init__(self, **cfg):
        """
        UDA With Teacher
        Parameters:
            model (dict): some with UDADecorator.

            EMATeacher:
            alpha (float): EMA alpha. 移动指数平滑中的alpha.
            pseudo_threshold (float): pseudo label confidence threshold
            psweight_ignore_top (int): Ignore the upper part of the pixel interval [0, psweight_ignore_top], default 0.
                忽略上部分的像素区间[0, psweight_ignore_top], 默认为0.
            psweight_ignore_bottom (int): Ignore the lower portion of the pixel interval [0, psweight_ignore_bottom], default 0.
                忽略下部分的像素区间[0, psweight_ignore_bottom], 默认为0.

        """
        super(UDAWithTeacher, self).__init__(**cfg)
        self.register_buffer('local_iter', torch.tensor(0, dtype=torch.long))
        # build ema teacher
        self._init_ema_teacher(cfg)

    def _init_ema_teacher(self, cfg):
        self.ema_model = EMATeacher(**cfg)

    def get_ema_model(self) -> EMATeacher:
        return self.ema_model

    def update_weights(self, iter: int):
        self.get_ema_model().update_weights(self.get_model(), iter)

    def forward_train(self, img, img_metas, **kwargs):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # batch_size, dev, log_vars = self._forward_setup(img, img_metas)
        # TODO
        raise NotImplementedError

    def train_step(self, data_batch, optimizer, **kwargs):
        """
        在每一次迭代之后  自动进行 iter++
        :param data_batch:
        :param optimizer:
        :param kwargs:
        :return:
        """
        out = super().train_step(data_batch, optimizer, **kwargs)
        self.local_iter += 1
        return out

    def _forward_setup(self, img):
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        # Init/update ema model
        self.update_weights(self.local_iter)
        return batch_size, dev, log_vars

    @staticmethod
    def _get_mean_std(img_metas, dev):
        return get_mean_std(img_metas, dev)

    @staticmethod
    def denorm(img, mean, std):
        return img.mul(std).add(mean) / 255.0

    @property
    def iter(self):
        return self.local_iter
