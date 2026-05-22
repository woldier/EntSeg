# Copyright (c) OpenMMLab. All rights reserved.
def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def is_dict_of(dict_cfg, expected_type):
    """
    判断当前的dict 中的元素 是否都是  expected_type
    :param dict_cfg:
    :param expected_type:
    :return:
    """
    exp_seq_type = dict
    if not isinstance(dict_cfg, exp_seq_type):
        return False
    for _, item in dict_cfg.items():
        if not isinstance(item, expected_type):
            return False
    return True
