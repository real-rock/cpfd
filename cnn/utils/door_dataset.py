from tensorflow.keras.utils import Sequence
import numpy as np
import sklearn

from cnn.exceptions.empty_property import EmptyPropertyException

BATCH_SIZE = 128
IMAGE_SIZE = 224


class DoorDataset(Sequence):
    def __init__(self, images_array, labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=False, pre_func=None):
        """
        파라미터 설명
        Args:
            images_array: 원본 224x224 만큼의 image 배열값.
            labels: 해당 image의 label들
            batch_size: __getitem__(self, index) 호출 시 마다 가져올 데이터 batch 건수
            augmentor: albumentations 객체
            shuffle: 학습 데이터의 경우 epoch 종료시마다 데이터를 섞을지 여부
        """
        if images_array is None:
            raise EmptyPropertyException("[ERROR] images_array can't be None value.")
        self.images_array = images_array
        if labels is None:
            raise EmptyPropertyException("[ERROR] labels can't be None value.")
        self.labels = labels
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.pre_func = pre_func
        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        images_fetch = self.images_array[index * self.batch_size: (index + 1) * self.batch_size]
        label_batch = self.labels[index * self.batch_size: (index + 1) * self.batch_size]

        return images_fetch, label_batch

    def on_epoch_end(self):
        if self.shuffle:
            self.images_array, self.labels = sklearn.utils.shuffle(self.images_array, self.labels)
        else:
            pass
