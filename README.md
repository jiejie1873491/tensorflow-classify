# tensorflow-classify

使用TensorFlow进行图像分类训练和测试，适合基础学习。

1、将训练数据和测试数据按类别放入各自文件夹里面，再将这些类别文件夹放到configs.py中指定的TRAIN_IMGPATH和VAL_IMGPATH文件夹下面；

2、mydataset.py为数据生成代码，返回一个批次的图像和对应的标签，实现了部分图像增强操作（随机水平翻转，随机裁剪、随机填充），直接运行mydataset.py可显示增强后的图像；

3、mynet.py中实现了focal loss，并对每个类别设置不同的权重alpha值，可以根据自己需求调用不同的backbone，本项目设置了backbone参数trainable=False，只训练最后一层的参数,如果想训练全部参数，将trainable设为True;
backbone预训练模型参数下载链接：https://github.com/tensorflow/models/tree/master/research/slim

4、mytrain.py为训练代码，打印训练集和测试集的损失和准确率；实现使用ImageNet训练的权重进行迁移训练以及模型中断训练后恢复训练，batch_normalization的使用

5、test.py测试代码
