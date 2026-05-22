# Copyright (c) OpenMMLab. All rights reserved.
import mmcv.utils
from mmcv.runner import build_optimizer, OptimizerHook
from mmseg.core.utils.misc import is_dict_of
import torch


def build_optimizers(model, cfg, distributed):
    # # for DA model or other model they want to use more than one optimizer
    if is_dict_of(cfg.optimizer.to_dict(), dict):
        optimizer = _build_optimizers(model, cfg.optimizer)
        if cfg.get('optimizer_cfg', None) is None:
            cfg.optimizer_config = None
        # default to use OptimizerHook
        elif distributed and 'type' not in cfg.optimizer_config:
            cfg.optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            cfg.optimizer_config = cfg.optimizer_config
    # for general segmentation model
    else:
        optimizer = build_optimizer(model, cfg.optimizer)
    return optimizer


def _build_optimizers(model, cfgs):
    """Build multiple optimizers from configs.

    If `cfgs` contains several dicts for optimizers, then a dict for each
    constructed optimizers will be returned.
    If `cfgs` only contains one optimizer config, the constructed optimizer
    itself will be returned.

    For example,

    1) Multiple optimizer configs:

    .. code-block:: python

        optimizer_cfg = dict(
            model1=dict(type='SGD', lr=lr),
            model2=dict(type='SGD', lr=lr))

    The return dict is
    ``dict('model1': torch.optim.Optimizer, 'model2': torch.optim.Optimizer)``

    2) Single optimizer config:

    .. code-block:: python

        optimizer_cfg = dict(type='SGD', lr=lr)

    The return is ``torch.optim.Optimizer``.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        cfgs (dict): The config dict of the optimizer.

    Returns:
        dict[:obj:`torch.optim.Optimizer`] | :obj:`torch.optim.Optimizer`:
            The initialized optimizers.
    """
    optimizers = {}
    if hasattr(model, 'module'):
        model = model.module
    # determine whether 'cfgs' has several dicts for optimizers
    is_dict_of_dict = True
    for key, cfg in cfgs.items():
        if not isinstance(cfg, dict):
            is_dict_of_dict = False
    assert is_dict_of_dict
    for key, cfg in cfgs.items():
        cfg_ = cfg.copy()
        module = getattr(model, key)
        if is_dict_of(cfg_.to_dict(), dict) and isinstance(_unwrap_model(module), torch.nn.ModuleDict):
            # 对于 discriminator 可能是一个dict， 并且此时如果 cfg_也是各 dict， 因此对每个都disc应该设置独立的optimizer，
            # 其他情况下， 如cfg_ 不是dict of dict， 尽管module 是torch.nn.ModuleDict， 也共享一个optimizer
            for k, m in _unwrap_model(module).items():
                assert k in cfg_, f'disc key={k} not in cfg_: {cfg_}'
                optimizers[f"{key}.{k}"] = build_optimizer(m, cfg_[k])
        else:
            optimizers[key] = build_optimizer(module, cfg_)
    return optimizers


def _unwrap_model(model):
    return getattr(model, 'module', model)
