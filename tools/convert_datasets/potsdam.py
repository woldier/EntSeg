# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import mmcv
import numpy as np

color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                      [255, 255, 0], [0, 255, 0], [0, 255, 255],
                      [0, 0, 255]])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='potsdam folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    # Original image of Potsdam dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.
    image = mmcv.imread(image_path)

    h, w, c = image.shape
    clip_size = args.clip_size
    stride_size = args.stride_size

    if image_path.split('/')[-1] == 'top_potsdam_4_12_label_noBoundary.tif':  # 这张图片的标注信息不对 导致label 基本全是 255
        # 获取 img dir
        base_path = os.path.dirname(image_path)
        sub_tmp_dir = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,d))][0]
        ori_img_path = glob.glob(os.path.join(base_path,sub_tmp_dir, '*4_12_*.tif'))[0]
        ori_img = mmcv.imread(ori_img_path)

        recovered_label = recover_label(image, ori_img, 0.17, color_map)
        color_image = color_map[recovered_label]
        image = color_image

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size
    ymin = y * clip_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ],
                     axis=1)

    if to_label:
        flatten_v = np.matmul(
            image.reshape(-1, c),
            np.array([2, 3, 4]).reshape(3, 1))
        out = np.zeros_like(flatten_v)
        for idx, class_color in enumerate(color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        idx_i, idx_j = osp.basename(image_path).split('_')[2:4]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(
                clip_save_dir,
                f'{idx_i}_{idx_j}_{start_x}_{start_y}_{end_x}_{end_y}.png'))


def main():
    args = parse_args()
    splits = {
        'train': [
            '2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11',
            '4_12', '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7',
            '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8', '7_9'
        ],
        'val': [
            '5_15', '6_15', '6_13', '3_13', '4_14', '6_14', '5_14', '2_13',
            '4_15', '2_14', '5_13', '4_13', '3_14', '7_13'
        ]
    }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'potsdam')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    zipp_list = sorted(glob.glob(os.path.join(dataset_path, '*.zip')))  # 保证 先解压 img 再解压label
    print('Find the data', zipp_list)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for zipp in zipp_list:
            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)
            src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
            if not len(src_path_list):
                sub_tmp_dir = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])
                src_path_list = glob.glob(os.path.join(sub_tmp_dir, '*.tif'))
            prog_bar = mmcv.ProgressBar(len(src_path_list))
            for i, src_path in enumerate(src_path_list):
                idx_i, idx_j = osp.basename(src_path).split('_')[2:4]
                data_type = 'train' if f'{idx_i}_{idx_j}' in splits[
                    'train'] else 'val'
                if 'label' in src_path:
                    dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                    clip_big_image(src_path, dst_dir, args, to_label=True)
                else:
                    dst_dir = osp.join(out_dir, 'img_dir', data_type)
                    clip_big_image(src_path, dst_dir, args, to_label=False)
                prog_bar.update()

    print('\nRemoving the temporary files...')

    print('Done!')


def recover_label(mixed_image, image, lam, color_map):
    """
    根据混合图像和原始图像反推 label。

    :param mixed_image: 混合结果 (H, W, 3)
    :param image: 原始图像 (H, W, 3)
    :param lam: 混合权重 λ
    :param color_map: 颜色映射表 (N, 3)
    :return: 恢复的 label 图 (H, W)
    """
    # 计算 label 的颜色
    label_color = (mixed_image - lam * image) / (1 - lam)

    # 初始化 label 矩阵
    h, w, _ = mixed_image.shape
    # recovered_label = np.zeros((h, w), dtype=np.int32)

    # 逐像素匹配最近的颜色 dist = np.linalg.norm(color_map - label_color[:,:,None,...], axis=-1); np.argmin(dist, axis=-1)
    # for i in range(h):
    #     for j in range(w):
    #         # 当前像素的颜色
    #         color = label_color[i, j]
    #         # 计算与每种颜色的欧氏距离
    #         distances = np.linalg.norm(color_map - color, axis=1)
    #         # 找到距离最近的颜色对应的标签
    #         recovered_label[i, j] = np.argmin(distances)
    dist = np.linalg.norm(color_map - label_color[:, :, None, ...], axis=-1)

    return np.argmin(dist, axis=-1)


if __name__ == '__main__':
    main()
