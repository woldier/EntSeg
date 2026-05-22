# -*- coding:utf-8 -*-
"""
 @FileName   : dist_utils.py
 @Time       : 12/15/24 2:56 PM
 @Author     : Woldier Wong
 @Description: TODO
"""
import torch
from mmcv.runner import master_only, get_dist_info


def is_distributed():
    if torch.cuda.is_available():
        return torch.distributed.is_initialized()
    return False


@master_only
def print_on_main_thread(message):
    print(message)
