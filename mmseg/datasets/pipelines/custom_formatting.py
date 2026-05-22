# -*- coding:utf-8 -*-
"""
 @FileName   : custom_formatting.py.py
 @Time       : 12/4/24 3:34 PM
 @Author     : Woldier Wong
 @Description: 自定义 formatting

"""
import numpy as np
from .formatting import to_tensor, DC
from ..builder import PIPELINES


@PIPELINES.register_module()
class FormatBundleByKey(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    增强 DefaultFormattingBundle, 原有的Bundle只支持固定的filed："img"和"gt_semantic_seg"。
    但是， 当我们生成多个img_aug 时， DefaultFormattingBundle不再适用。
    因此我们通过key 来对filed 进行处理。
    对于图像 的命名， 其中必须含有 'img' 字符串，这样的fields 会被当作 图像处理
    对于标签 的命名， 其中必须含有 'seg' 字符串，这样的fields 会被当作 标签处理

    """
    _DEFAULT_KEYS = ['img', 'gt_semantic_seg']

    def __init__(self, keys=None):
        # 判断keys种是否有_DEFAULT_KEYS的所有元素, 没有就添加进去
        if keys is None:
            keys = self._DEFAULT_KEYS
        keys = list(keys)
        # 添加缺失的元素
        for key in self._DEFAULT_KEYS:
            assert 'img' in key or 'seg' in key
            if key not in keys:
                keys.append(key)  # 或者更新集合：self.keys.add(key)
        self.keys = tuple(keys)

    def __call__(self, results):
        """
        Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        for key in self.keys:
            if 'img' in key:
                img = results[key]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                results[key] = DC(to_tensor(img), stack=True)
            if 'seg' in key:
                # convert to long
                results[key] = DC(
                    to_tensor(results[key][None, ...].astype(np.int64)),
                    stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__
