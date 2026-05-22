# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# https://github.com/lhoyer/DAFormer/blob/master/mmseg/models/uda/uda_decorator.py
# ---------------------------------------------------------------
import torch
from copy import deepcopy
from mmcv.parallel import MMDistributedDataParallel
from mmseg.ops import resize
from mmseg.models import BaseSegmentor, build_segmentor, EncoderDecoder


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


class UDADecorator(BaseSegmentor):

    def __init__(self, work_dir=None, cmap='isprs', **cfg):
        """"
        默认会使用cfg.model 的配置作为 model 的配置文件
        并且本UDADecorator 默认会使用cfg.model 的train_cfg 和test_cfg 文件

        Parameters:
            model (dict): model 的配置
                train_cfg (dict): model 的train 配置
                test_cfg (dict): model 的train 配置
                decode_head (dict):
                    num_classes (int): 分类数目
            work_dir(str|None): 当前工作目录用于需要保存临时文件的需求，如果没设置在init UDA model 时会设置与  cfg.work_dir 相同
            cmap(str): 方法中最常见的保存临时文件的需求是可视化训练过程的图片，而cmap定义了mask 的color.
                        支持的cmap详见mmseg/models/utils/visualization.py
        """

        super(BaseSegmentor, self).__init__()

        self.model = build_segmentor(deepcopy(cfg['model']))
        self.train_cfg = cfg['model']['train_cfg']
        self.test_cfg = cfg['model']['test_cfg']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.work_dir = work_dir
        self.cmap = cmap

    def get_model(self) -> EncoderDecoder:
        return get_module(self.model)

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(self,
                      img, img_metas,
                      **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # losses = self.model.forward_train(
        #     img, img_metas, gt_semantic_seg, return_feat=return_feat)
        # return losses
        raise NotImplementedError()

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        return self.get_model().aug_test(imgs, img_metas, rescale)

    def _forward_feat_logic_pre(self, img, img_metas):
        """
        对 model 进行 forward 并且返回 中间层的feature，head预测的logic值，以及logic值经过softmax和argmax后得到的预测类别
        :param img:
        :param img_metas:
        :return:
        """
        feature = self.get_model().extract_feat(img)
        pred = self.get_model()._decode_head_forward_test(feature, img_metas)
        pred = resize(input=pred, size=img.shape[2:], mode='bilinear',
                      align_corners=self.get_model().align_corners)
        label = torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=True)
        return feature, pred, label

    def _process_loss(self, log_vars, retain_graph=False, **kwargs):
        """
            用于处理loss
            Examples:
            >>> log_vars = {}
            >>> loss_ce = # 计算 损失
            >>> self._process_loss(log_vars, loss_ce=loss_ce)  # loss 多节点同步&反向传播&保留item用于log
            >>> ###
            >>> # log_vars: {loss_ce:1.99}
        """
        loss_clean, loss_var = self._parse_losses(kwargs)
        log_vars.update(loss_var)
        loss_clean.backward(retain_graph=retain_graph)
