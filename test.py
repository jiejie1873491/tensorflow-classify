# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import glob
import time
import shutil
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import configs as cfgs
from mynet import myNet

class_names = sorted(os.listdir(cfgs.TRAIN_IMGPATH))
save_dir = 'models'

net = myNet()
output, logit = net.backbone()

saver= tf.train.Saver()
with tf.Session() as sess:
    checkpoint_path = tf.train.latest_checkpoint(save_dir)
    saver.restore(sess, checkpoint_path)

    print('***'*20)
    print('model params restore from ' + checkpoint_path)
    print('***'*20)
    
    imgpath = '/media/txtx/f19f1c88-52d1-4fee-9746-dd97d2c44beb/data/cat_vs_dog/kaggle/google'
    num=0
    for imgname in glob.glob(os.path.join(imgpath, '*.jp*')) :   
        img = cv2.imread(imgname)
        if img is None:
            continue

        index = os.path.split(imgname)[-1]
        img1 = cv2.resize(img, (cfgs.IMG_SIZE, cfgs.IMG_SIZE))
        img2 = img1.astype(np.float32) / 255.0
        img2 = np.expand_dims(img2, axis=0)
        
        t1 = time.time()
        logits = sess.run(logit,
                          feed_dict = {net.inputs:img2,
                                       net.is_training:False})
        t2 = time.time() - t1
        print(t2)
        label = np.argmax(logits, axis = 1)[0]
        if label == 0:
            num+=1
        cv2.putText(img, '{}: {:.4f}'.format(class_names[label], logits[0][label]), (20, 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 2)
        cv2.imshow('', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    print(num)    
    