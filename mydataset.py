import os
import cv2
import glob
import numpy as np
import configs as cfgs

class myDataset(object):
    def __init__(self, imgdir):
        self.imgdir = imgdir
        self.label = sorted(os.listdir(self.imgdir))
        self.index = 0

        datasets = []
        for sublabel, subdir in enumerate(self.label):
            for imgname in os.listdir(os.path.join(self.imgdir, subdir)):
                datasets.append([os.path.join(self.imgdir, subdir, imgname), sublabel])
        self.datasets = datasets

        
    def random_horizon_flip(self, img):
        if np.random.random() > 0.5:
            img = img[:, ::-1, :]
        return img 
    
    def random_crop(self, img):
        if np.random.random() > 0.5:
            h, w = img.shape[:2]
            ratio = 0.1
            x1 = np.random.randint(0, w*ratio)
            y1 = np.random.randint(0, h*ratio)
            x2 = np.random.randint(w*(1-ratio), w-1)
            y2 = np.random.randint(h*(1-ratio), h-1)
            img = img[y1:y2, x1:x2,:]
        return img
    
    def random_fill(self, img):
        if np.random.random() > 0.5:
            h, w = img.shape[:2]
            ratio = 0.2
            black_h = np.zeros(shape=[np.random.randint(0, h*ratio), w, 3], dtype=np.uint8)
            img = np.concatenate([black_h, img, black_h], axis=0)
            black_w = np.zeros(shape=[img.shape[0], np.random.randint(0, w*ratio), 3], dtype=np.uint8)
            img = np.concatenate([black_w, img, black_w], axis=1)
        return img

    def gen_data(self, batch_size, is_training):
        if self.index % len(self.datasets) == 0:
            np.random.shuffle(self.datasets)

        batch_imgs = np.zeros(shape=[batch_size, cfgs.IMG_SIZE, cfgs.IMG_SIZE, 3], dtype=np.float32)
        batch_labels = np.zeros(shape=[batch_size, cfgs.NUM_CLASS], dtype=np.float32)
        for i in range(batch_size):
            data = self.datasets[self.index]
            imgname = data[0]
            img = cv2.imread(imgname)
            # 1 image augmentation;(random_horizon_flip, random_crop, random_fill)
            # TODO: brightness, rotate...
            if is_training:
                img = self.random_horizon_flip(img)
                img = self.random_crop(img)
                img = self.random_fill(img)
            
            # 2 normalization and resize 
            img = img.astype(np.float32) / 255.0
            img = cv2.resize(img, (cfgs.IMG_SIZE, cfgs.IMG_SIZE))
            batch_imgs[i] = img

            label = np.zeros(cfgs.NUM_CLASS)
            label[data[1]] = 1
            batch_labels[i] = label
            
            self.index += 1
            if self.index == len(self.datasets):
                self.index = 0

        return batch_imgs, batch_labels


if __name__ == "__main__":
    dataset = myDataset(cfgs.TRAIN_IMGPATH)
    class_names = sorted(os.listdir(cfgs.TRAIN_IMGPATH))
    for k in range(10):
        print(k)
        imgs, labels = dataset.gen_data(cfgs.BATCH_SIZE, True)
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = img*255.0
            img = img.astype(np.uint8)
            cv2.putText(img, class_names[np.argmax(labels[i])], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(img.shape)
            cv2.imshow('',img)
            cv2.waitKey()
            cv2.destroyAllWindows()
