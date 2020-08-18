import os
import cv2
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
# from tensorflow.contrib.slim.python.slim.nets import resnet_utils
# resnet_arg_scope = resnet_utils.resnet_arg_scope

import configs as cfgs

class myNet(object):
    def __init__(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, cfgs.IMG_SIZE, cfgs.IMG_SIZE, 3], name='gt_inputs')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, cfgs.NUM_CLASS], name='gt_labels')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    def catergorical_focal_loss(self, y_true, y_pred):
        """
        Formula:
            loss = -alpha*((1-p_t)^gamma)*log(p_t)
        """
        print(y_true.shape)
        gamma = 1.0
        alpha_class = np.array([0.2, 0.3, 0.5], dtype=np.float32)
        
        y_pred = tf.nn.softmax(y_pred)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate cross entropy
        cross_entropy = -tf.cast(y_true, dtype=tf.float32) * tf.math.log(y_pred)

        # Calculate weight that consists of modulating factor and weighting factor
        alpha = tf.gather(alpha_class, tf.argmax(y_true, -1))
        alpha = tf.expand_dims(alpha, axis=-1)
        weight = alpha * tf.pow((1 - y_pred), gamma)  # alpha * K.pow((1-y_pred), gamma) 

        # Calculate focal loss
        loss = weight * cross_entropy

        # # mean the losses in mini_batch
        # focal_loss = tf.reduce_mean(loss)
        
        return loss

    def backbone(self): 
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            with slim.arg_scope([slim.conv2d], trainable=False):
                # output, end_points = resnet_v1.resnet_v1_50(self.inputs, num_classes=cfgs.NUM_CLASS, is_training=self.is_training)
                output, end_points = resnet_v1.resnet_v1_101(self.inputs, num_classes=None, is_training=self.is_training, global_pool=False)

        output = slim.conv2d(output,
                             cfgs.NUM_CLASS, [1, 1],
                             activation_fn=None,
                             normalizer_fn=None,
                             scope='logits')

        output = tf.reduce_mean(output, [1, 2], name='global_pool')
        logits = tf.nn.softmax(output)
            
        return output, logits

    def compute_loss(self, labels, logits):
        loss = self.catergorical_focal_loss(labels, logits)
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_mean(loss)
            
    def compute_acc(self, labels, logits):
        # acc = tf.cast(tf.equal(tf.argmax(net, axis=1), tf.to_int64(labels)), tf.float32)
        acc = tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), tf.float32)
        acc = tf.reduce_mean(acc)
        
        return acc