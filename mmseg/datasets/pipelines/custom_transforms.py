from ..builder import PIPELINES
from .transforms import PhotoMetricDistortion, Pad, Normalize, mmcv


@PIPELINES.register_module()
class NormalizeByKey(Normalize):
    _DEFAULT_KEYS = ['img']

    def __init__(self, mean, std, to_rgb=True, keys=None):
        super().__init__(mean, std, to_rgb)
        # 判断keys种是否有_DEFAULT_KEYS的所有元素, 没有就添加进去
        if keys is None:
            keys = self._DEFAULT_KEYS
        keys = list(keys)
        # 添加缺失的元素
        for key in self._DEFAULT_KEYS:
            if key not in keys:
                keys.append(key)  # 或者更新集合：self.keys.add(key)
        self.keys = tuple(keys)

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in self.keys:
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class PhotoMetricDistortionByKey(PhotoMetricDistortion):
    def __init__(
            self,
            brightness_delta=32, contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5), hue_delta=18,
            keys: tuple = ('img',)
    ):
        """
        :param keys: 需要进行PhotoMetricDistortion的key
        """
        super().__init__(brightness_delta, contrast_range, saturation_range, hue_delta)
        self.keys = keys

    def __call__(self, results):
        img = results["img"]  # 原始的img
        for key in self.keys:  # 多次增强
            aug_img = self._inner_call(img)
            results[key] = aug_img
        return results


@PIPELINES.register_module()
class PadByKey(Pad):
    def __init__(
            self, size=None, size_divisor=None,
            pad_val=0, seg_pad_val=255,
            keys: tuple = ('img',)
    ):
        super().__init__(size, size_divisor, pad_val, seg_pad_val)
        self.keys = keys

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""

        # padded_img = self._inner_pad_img(img)
        # results['img'] = padded_img
        assert "img" in self.keys  # 所有的图片都要pad
        for key in self.keys:
            img = results[key]
            padded_img = self._inner_pad_img(img)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
