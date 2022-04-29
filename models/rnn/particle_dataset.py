from tensorflow.keras.utils import Sequence
import numpy as np
import sklearn


class ParticleDS(Sequence):
    def __init__(self, x, y, window_size=12, batch_size=32):
        self.X = x
        self.y = y
        self.window_size = window_size
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        X_batch = self.X[index * self.batch_size: (index + 1) * self.batch_size]
        y_batch = self.y[index * self.batch_size: (index + 1) * self.batch_size]
        return X_batch, y_batch
