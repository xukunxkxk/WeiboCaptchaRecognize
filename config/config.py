# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn
# define the paths to the images directory
# images path of the captcha
IMAGES_PATH = "..."

# Captcha list#
# a-z A-Z 0-9
CLASSES_LIST = [chr(ord('a')+x) for x in range(26)] + \
               [str(x) for x in range(10)]
# a > 0 b > 1 ... 9 > 35
CLASSES_DICT = {v: index for index, v in enumerate(CLASSES_LIST)}
# 0 > a 1 > b ... 35 > 9
REVERSE_CLASSES_DICT = {index: v for index, v in enumerate(CLASSES_LIST)}

NUM_CLASSES = len(CLASSES_LIST)

# total chars in the captcha images
CHAR_NUMBERS = 5

# model saving path
MODEL_PATH = "..."

# loss and accuracy saving path
OUTPUT_PATH = "..."

# images height, width and depth
HEIGHT = 40
WIDTH = 100
DEPTH = 3
PADDING = 8

# nums of epoch
EPOCH = 60

# initial learning rate
INIT_LR = 0.001

# restart training check point
CHECKPOINT = ''
START_EPOCH = 0
NEW_LR = 0.0001
BATCH_SIZE = 32

# using which kind of network to train
NET = 'AttentionLSTM'
# split the labels or not
SPLIT = False
# using data aug or not
USING_AUG = False

# model to predict
CNN_MODEL = r'...'

# CNN fine-tuned
LAY_NUM = 11


if __name__ == '__main__':
    pass
