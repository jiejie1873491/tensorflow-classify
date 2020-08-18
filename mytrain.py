import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

from mydataset import myDataset
from mynet import myNet
import configs as cfgs 

# datasets
train_datasets = myDataset(cfgs.TRAIN_IMGPATH)
val_datasets = myDataset(cfgs.VAL_IMGPATH)

# network, loss, acc
net = myNet()
output, logits = net.backbone()
loss = net.compute_loss(net.labels, output)
acc = net.compute_acc(net.labels, logits)

# hi_params
global_step = tf.compat.v1.train.get_or_create_global_step()
lr = tf.compat.v1.train.piecewise_constant(global_step,
                                 boundaries=[200000],
                                 values=[cfgs.BASE_lr, cfgs.BASE_lr / 10.0])

# optimizer, train_op
optimizer = tf.compat.v1.train.AdamOptimizer(lr)
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.compat.v1.control_dependencies(update_ops):
    weight_decay_loss = tf.compat.v1.add_n(tf.compat.v1.losses.get_regularization_losses())
    train_op = optimizer.minimize(loss + weight_decay_loss, global_step=global_step)

# begin to train
saver_dir = 'models'
if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)

saver = tf.compat.v1.train.Saver(max_to_keep=100)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # load pretrained model
    checkpoint_path = tf.compat.v1.train.latest_checkpoint(saver_dir)
    if checkpoint_path is not None:
        print('***'*20)
        print('model params restore from ' + checkpoint_path)
        print('***'*20)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(sess, checkpoint_path)
    else:
        print('***'*20)
        print('model params restore from imagenets')
        print('***'*20)
        checkpoint_path = 'tf_pretrained_models/resnet_v1_101.ckpt'
        model_variables = slim.get_model_variables()
        restore_variables = {}
        for var in model_variables:
            if var.op.name.startswith('resnet_v1_101'):
                if 'logits' in var.op.name:
                    print(var.op.name + 'do not loaded')
                    continue
                restore_variables[var.op.name] = var
            else:
                print(var.op.name + ' not in the pretrained model')
        
        restorer = tf.compat.v1.train.Saver(restore_variables)
        restorer.restore(sess, checkpoint_path)
    
    print('\nStart to training')
    for step in range(1, cfgs.TRAIN_STEP):
        global_stepn = sess.run(global_step)
        if global_stepn >= cfgs.TRAIN_STEP:
            break

        is_training = True
        batch_imgs, batch_labels = train_datasets.gen_data(cfgs.BATCH_SIZE, is_training)
        _, global_stepn, train_loss, train_acc = sess.run([train_op, global_step, loss, acc],
                                                           feed_dict={net.inputs: batch_imgs,
                                                                      net.labels: batch_labels,
                                                                      net.is_training: is_training})
        print('step:{} global_step:{} loss: {:.6f} acc: {:.4f}'.format(step, global_stepn + 1, train_loss, train_acc))
        if global_stepn != 0 and global_stepn % cfgs.SAVE_STEP == 0:
            saver.save(sess, os.path.join(saver_dir, 'classify_'+str(global_stepn)+'.ckpt'))

        # Do valuate
        if  global_stepn != 0 and global_stepn % cfgs.VAL_STEP == 0:
            val_losses = []
            val_acces = []
            is_training_val = False
            for _ in range(len(val_datasets.datasets)//cfgs.BATCH_SIZE):
                val_batch_imgs, val_batch_labels = val_datasets.gen_data(cfgs.BATCH_SIZE, is_training_val)
                val_loss, val_acc = sess.run([loss, acc], feed_dict={net.inputs: val_batch_imgs,
                                                                     net.labels: val_batch_labels,
                                                                     net.is_training: is_training_val})
                val_losses.append(val_loss)
                val_acces.append(val_acc)
            print('mean_val_loss: {:.6f} mean_val_acc: {:.4f}'.format(np.mean(val_losses), np.mean(val_acces)))
            with open('val.txt', 'a') as f:
                f.write('step:{}  global_step:{}  mean_val_loss: {:.6f}  mean_val_acc: {:.4f}\n'.format(step, global_stepn + 1, np.mean(val_losses), np.mean(val_acces)))

