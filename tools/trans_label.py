# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from PIL import Image
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='VOC2012', choices=['VOC2012', 'KITTI'], help="where the dataset is stored")
parser.add_argument("--label_path", type=str, default="", help="Where to dump the data")
parser.add_argument("--trans_path", type=int, default="", help="number of threads to use")
args = parser.parse_args()


def convert_to_single_channel_label(original_label_path, color_mapping):
    original_label = cv2.imread(original_label_path)[:,:,::-1]
    single_channel_label = np.zeros(original_label.shape[:2], dtype=np.uint8)

    for i in range(original_label.shape[0]):
        for j in range(original_label.shape[1]):
            rgb = tuple(original_label[i][j])
            class_index = color_mapping[rgb]
            single_channel_label[i, j] = class_index

    return single_channel_label

# VOC2012
color_mapping = {
    (0, 0, 0): 0,         # 背景类别
    (128, 0, 0): 1,       # 飞机类别
    (0, 128, 0): 2,       # 自行车类别
    (128, 128, 0): 3,     # 鸟类别
    (0, 0, 128): 4,       # 船类别
    (128, 0, 128): 5,     # 瓶子类别
    (0, 128, 128): 6,     # 公交车类别
    (128, 128, 128): 7,   # 车类别
    (64, 0, 0): 8,        # 猫类别
    (192, 0, 0): 9,       # 椅子类别
    (64, 128, 0): 10,     # 奶牛类别
    (192, 128, 0): 11,    # 餐桌类别
    (64, 0, 128): 12,     # 狗类别
    (192, 0, 128): 13,    # 马类别
    (64, 128, 128): 14,   # 摩托车类别
    (192, 128, 128): 15,  # 人类别
    (0, 64, 0): 16,       # 盆栽植物类别
    (128, 64, 0): 17,     # 羊类别
    (0, 192, 0): 18,      # 沙发类别
    (128, 192, 0): 19,    # 火车类别
    (0, 64, 128): 20,     # 电视类别
    (224, 224, 192): 0,
}


def main():
    files = os.listdir(args.label_path)
    for f in files:
        if args.dataset == 'VOC2012':
            img = convert_to_single_channel_label(os.path.join(args.label_path,f), color_mapping)
        else:
            img = cv2.imread(os.path.join(args.label_path,f),0)
            
        if args.dataset == 'KITTI':
            img = cv2.resize(img, (1242,375),interpolation=cv2.INTER_NEAREST)
    
        output_path = os.path.join(args.trans_path, f.split(".")[0] + '.csv')
        np.savetxt(output_path, img.astype(int), fmt='%i', delimiter=",")
        
        
if __name__ == '__main__':
    main()