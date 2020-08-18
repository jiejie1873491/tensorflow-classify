# -*- coding: utf-8 -*-
import os

TRAIN_IMGPATH = '/media/txtx/f19f1c88-52d1-4fee-9746-dd97d2c44beb/data/cat_vs_dog/kaggle/train_keras'
VAL_IMGPATH = '/media/txtx/f19f1c88-52d1-4fee-9746-dd97d2c44beb/data/cat_vs_dog/kaggle/test_keras'
NUM_CLASS = len(sorted(os.listdir(TRAIN_IMGPATH)))

TRAIN_STEP = 100001
SAVE_STEP = 2000
VAL_STEP = 2000

BASE_lr = 0.001
WEIGHT_DECEY = 0.0005

BATCH_SIZE = 16
IMG_SIZE = 224