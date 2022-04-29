from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import albumentations as A

import imageio
import glob
import numpy as np

from models.utils.door_dataset import DoorDataset


class ImageHandler:
    def __init__(self, src_dirs, classes, class_mode='binary', file_ext='png', batch_size=32, pre_func=None):
        self.__src_dirs = src_dirs
        self.__classes = classes
        self.__file_ext = file_ext

        self.__augmentor = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5)
        ])

        self.__images, self.__labels = self.__read_and_label_images()
        print(f'image shape: {self.__images.shape}')
        print(f'label shape: {self.__labels.shape}')

        self.__scale_image()

        self.__one_hot_encoding(pre_func=pre_func)

        (self.train_images, self.train_labels), \
            (self.valid_images, self.valid_labels), \
            (self.test_images, self.test_labels) = self.get_train_valid_test_set()

        self.train_ds, self.valid_ds, self.test_ds = self.get_dataset(batch_size=batch_size)

    def __read_and_label_images(self):
        images = []
        labels = []
        for idx, src_dir in enumerate(self.__src_dirs):
            print(f'[INFO] reading src `{src_dir}`')
            for img_path in glob.glob(src_dir + '/*.' + self.__file_ext):
                img = imageio.imread(img_path)
                images.append(img)
                labels.append(self.__classes[idx])

        print(f'[INFO] finished reading images: len of images {len(images)}, len of labels {len(labels)}')
        return np.array(images, dtype='float32'), np.array(labels, dtype='float32')

    def __scale_image(self):
        print('[INFO] scaling images')
        self.__images = self.__images / 255.0

    def __one_hot_encoding(self, pre_func=None):
        if pre_func is not None:
            self.__images = pre_func(self.__images)
        self.__labels = to_categorical(self.__labels)

    def get_train_valid_test_set(self, test_size=0.25, valid_size=0.2, random_state=42):
        train_images, test_images, train_labels, test_labels = train_test_split(
            self.__images,
            self.__labels,
            test_size=test_size,
            random_state=random_state
        )
        print(f'[INFO] train_images, train_labels, test_images, test_labels splitted')
        print(f'[INFO] each shape: {train_images.shape}, {train_labels.shape}, {test_images.shape}, {test_labels.shape}')
        train_images, valid_images, train_labels, valid_labels = train_test_split(
            train_images,
            train_labels,
            test_size=valid_size,
            random_state=random_state
        )
        return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)

    def get_dataset(self, batch_size=32):
        return DoorDataset(self.train_images, self.train_labels, batch_size=batch_size, augmentor=self.__augmentor, shuffle=True), \
                DoorDataset(self.valid_images, self.valid_labels, batch_size=batch_size), \
                DoorDataset(self.test_images, self.test_labels, batch_size=batch_size)
