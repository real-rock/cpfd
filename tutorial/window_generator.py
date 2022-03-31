import os
import datetime

import IPython
import IPython.display
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


class WindowGenerator:
    _example = None

    def __init__(self, input_width, label_width, shift, train=None, val=None, test=None, label_columns=None):
        # dataset
        self.__train_df = train
        self.__val_df = val
        self.__test_df = test

        self.__label_columns = label_columns
        if label_width is not None:
            self.__label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.__column_indices = {name: i for i, name in enumerate(train.columns)}

        self.__input_width = input_width
        self.__label_width = label_width
        self.__shift = shift

        self.__total_window_size = input_width + shift

        self.__input_slice = slice(0, input_width)
        self.__input_indices = np.arange(self.__total_window_size)[self.__input_slice]

        self.__label_start = self.__total_window_size - self.__label_width
        self.__labels_slice = slice(self.__label_start, None)
        self.__label_indices = np.arange(self.__total_window_size)[self.__labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.__total_window_size}',
            f'Input indices: {self.__input_indices}',
            f'Label indices: {self.__label_indices}',
            f'Label column name(s): {self.__label_columns}',
        ])

    @property
    def train(self):
        return self.make_dataset(self.__train_df)

    @property
    def val(self):
        return self.make_dataset(self.__val_df)

    @property
    def test(self):
        return self.make_dataset(self.__test_df)

    @property
    def example(self):
        """Get and cache an example batch of 'inputs, labels' for plotting"""
        result = getattr(self, '_example', None)
        if result is not None:
            result = next(iter(self.train))
        self._example = result
        return result

    def get_total_window_size(self):
        return self.__total_window_size

    def split_window(self, features):
        inputs = features[:, self.__input_slice, :]
        labels = features[:, self.__labels_slice, :]
        if self.__label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.__column_indices[name]] for name in self.__label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.__input_width, None])
        labels.set_shape([None, self.__label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.__column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.__input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

            if self.__label_columns:
                label_col_index = self.__label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.__label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels',
                        c="#2ca02c", s=64)

            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.__label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k',
                            label='Predictions', c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.get_total_window_size(),
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        ds = ds.map(self.split_window)

        return ds
