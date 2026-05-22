# -*- coding:utf-8 -*-
"""
 @FileName   : visualization.py.py
 @Time       : 9/24/24 5:17 PM
 @Author     : Woldier Wong
 @Description: https://github.com/lhoyer/DAFormer/blob/master/mmseg/models/utils/visualization.py
"""
import numpy as np, os
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.jit.frontend import NotSupportedError

from mmseg.datasets import ISPRSDataset, LoveDADataset
from enum import Enum
from mmseg.utils.img_utils import denorm

_ISPRS_PALETTE = [color for sublist in ISPRSDataset.PALETTE for color in sublist]  # 转为1维数组 L* 3
_LoveDA_PALETTE = [color for sublist in LoveDADataset.PALETTE for color in sublist]
_PALETTE_DICT = {
    "isprs": _ISPRS_PALETTE,
    "loveda": _LoveDA_PALETTE
}


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def _colorize(img, cmap, mask_zero=False):
    vmin = np.min(img)
    vmax = np.max(img)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image


def subplotimg(ax,
               img,
               title,
               range_in_title=False,
               **kwargs):
    if img is None:
        return
    with torch.no_grad():
        if torch.is_tensor(img):
            img = img.cpu()
        if len(img.shape) == 2:
            if torch.is_tensor(img):
                img = img.numpy()
        elif img.shape[0] == 1:
            if torch.is_tensor(img):
                img = img.numpy()
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            img = img.permute(1, 2, 0)
            if not torch.is_tensor(img):
                img = img.numpy()
        # 排除 cmap="grey" 的情况 这种情况要将 其传给ax.imshow(img, **kwargs)
        if kwargs.get("cmap", "") in _PALETTE_DICT.keys():
            cmap = kwargs.pop('cmap')
            # assert str.lower(cmap) in _PALETTE_DICT.keys(), "support " + "|".join(_PALETTE_DICT.keys()) + f"got {cmap}"
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize_mask(img, _PALETTE_DICT[cmap])

    if range_in_title:
        vmin = np.min(img)
        vmax = np.max(img)
        title += f' {vmin:.3f}-{vmax:.3f}'

    ax.imshow(img, **kwargs)
    ax.set_title(title)


class DataType(Enum):
    IMG = 'img'
    SEG = 'seg'
    WEIGHT = 'weight'
    NONE = 'none'


class VisDataContainer(object):
    def __init__(self, name: str, data, data_type: DataType):
        self.name = name
        self.data = data
        assert isinstance(data_type, DataType), "data_type must be an instance of DataType Enum"
        self.data_type = data_type

    def __repr__(self):
        return f"VisDataContainer(name={self.name}, data_type={self.data_type})"


def show_img(
        data_list,
        batch_size, means, stds,
        cmap, out_file
):
    """
    :param data_list: 支持两种方式 [VisDataContainer, ..., VisDataContainer] 或者 [ [VDC, VDC], [VDC,]]
    :param batch_size:
    :param means:
    :param stds:
    :param cmap:
    :param out_file:
    :return:
     DataType 用于设置不同data 的处理方式
         DataType.IMG： 对于img 在 可视化时要将其de norm
         DataType.SEG： 对于label 索引在可视化是根据cmap进行着色
         DataType.WEIGHT 对于一些中间的权重，直接展示， 不用特殊处理
     Examples:
     所有batch 放在一个图中
     >>> from mmseg.models.utils.visualization import show_img, VisDataContainer as VDC, DataType
     >>>        out_dir = os.path.join(self.work_dir, 'debug_img')
     >>>        out_file = os.path.join(out_dir, f'iter_{int(self.iter):08d}.png')
     >>>        data_list = [  # 使用的方式是一个样本一张图
     >>>                VDC('Source Image', img, DataType.IMG),
     >>>                VDC('Source Image Tgt. Style', src_img_in_tgt_sty, DataType.IMG),
     >>>                VDC('Source EMA pre.', ema_src_pre, DataType.SEG),
     >>>                VDC('Source pre.', src_pre, DataType.SEG),
     >>>                VDC('Source Seg GT', gt_semantic_seg, DataType.SEG),
     >>>                VDC('Target Image', target_img, DataType.IMG),
     >>>                VDC('Target Pseudo', pseudo_label, DataType.SEG),
     >>>        ]
     >>>        show_img(data_list, batch_size, means, stds, self.cmap, out_file)

     一个样本一张图
     >>> from mmseg.models.utils.visualization import show_img, VisDataContainer as VDC, DataType
     >>>        out_dir = os.path.join(self.work_dir, 'debug_img')
     >>>        out_file = os.path.join(out_dir, f'iter_{int(self.iter):08d}.png')
     >>>        data_list = [  # 使用的方式是一个样本一张图
     >>>            # 第一行的数据
     >>>            [
     >>>                VDC('Source Image', img, DataType.IMG),
     >>>                VDC('Source Image Tgt. Style', src_img_in_tgt_sty, DataType.IMG),
     >>>                VDC('Source EMA pre.', ema_src_pre, DataType.SEG),
     >>>                VDC('Source pre.', src_pre, DataType.SEG),
     >>>                VDC('Source Seg GT', gt_semantic_seg, DataType.SEG),
     >>>            ],
     >>>            # 第二行的数据
     >>>            [
     >>>                VDC('Target Image', target_img, DataType.IMG),
     >>>                VDC('Target Image Src. Style', tgt_img_in_src_sty, DataType.IMG),
     >>>                VDC('Target Pseudo', pseudo_label, DataType.SEG),
     >>>            ],
     >>>        ]
     >>>        show_img(data_list, batch_size, means, stds, self.cmap, out_file)

    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)  # 创建dir
    gridspec_kw = {'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0.05, 'right': 1, 'left': 0}
    # 数据预处理 img denorm
    _denorm_img(data_list, means, stds)
    # 如果 data_list 是 一维数组， 那么只有一张图片且 batch size 就是图片的行数， 列数为 len(data_list)
    if isinstance(data_list[0], VisDataContainer):
        rows, cols = batch_size, len(data_list)
        fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), gridspec_kw=gridspec_kw)
        for row in range(batch_size):
            for col, item in enumerate(data_list):
                _subplotimg(item, row, col, axs, cmap, row)
        for ax in axs.flat:
            ax.axis('off')
        plt.savefig(out_file)
        plt.close()
    # 如果 data_list 是 二维数组， 那么有batch size 张图片， 每张图片有len(data_list[0])行
    elif isinstance(data_list[0], list) and isinstance(data_list[0][0], VisDataContainer):
        rows, cols = len(data_list), max([len(item) for item in data_list])  # 得到行数和列数
        for img_idx in range(batch_size):
            fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), gridspec_kw=gridspec_kw)
            for row, row_item in enumerate(data_list):  # 遍历行
                for col, item in enumerate(row_item):  # 遍历列
                    _subplotimg(item, row, col, axs, cmap, img_idx)  # plot

            for ax in axs.flat:
                ax.axis('off')
            out_file_split = os.path.splitext(out_file)
            new_path = ''.join(out_file_split[0]) + f"_batch_{img_idx:03}_" + ''.join(out_file_split[-1])  # 拼接新名字
            plt.savefig(new_path)
            plt.close()
    else:
        raise NotSupportedError("data_list supports only one or two dimensional arrays.")


def _subplotimg(item, row, col, axs, cmap, idx):
    """"
        row(int): axs 的 行索引
        col(int): axs 的 列索引
        cmap(str): 对于SEG 需要着色
        idx(int): 选择的data索引， item 中存的数据是一个batch 的 [b, c, h, w],我们通过idx 确定取的索引
    """
    if item.data_type == DataType.NONE or item.data is None: return
    if item.data_type == DataType.SEG:
        subplotimg(axs[row][col], item.data[idx], item.name, cmap=cmap)
    else:
        subplotimg(axs[row][col], item.data[idx], item.name)


def _denorm_img(data_list, means, stds):
    for item in data_list:
        if isinstance(item, list):
            _denorm_img(item, means, stds)  # 递归
        elif isinstance(item, VisDataContainer):
            if item.data_type == DataType.IMG and item.data is not None:
                item.data = torch.clamp(denorm(item.data, means, stds), 0, 1)
        else:
            raise NotSupportedError(f"not support type {type(data_list)}")
