import os
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from base.base_data_loader import BaseDataLoader


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sample_rate, duration_time, n_classes,
                 batch_size=32, shuffle=True):
        self._wav_paths = wav_paths
        self._labels = labels
        self._sample_rate = sample_rate
        self._duration_time = duration_time
        self._n_classes = n_classes
        self._batch_size = batch_size
        self._shuffle = shuffle

        # Splitted audio index
        self._indexes = []

        self._on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self._wav_paths) / self._batch_size))

    def __getitem__(self, index):
        indexes = self._indexes[
                  index * self._batch_size:(index+1) * self._batch_size]

        wav_paths = [self._wav_paths[k] for k in indexes]
        labels = [self._labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self._batch_size, 1,
                      int(self._sample_rate * self._duration_time)),
                     dtype=np.int16)
        Y = np.empty((self._batch_size, self._n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i, ] = wav.reshape(1, -1)
            Y[i, ] = to_categorical(label - 1, num_classes=self._n_classes)

        # Returns 1 second audio file and one-hot encoded label
        return X, Y

    def _on_epoch_end(self):
        self._indexes = np.arange(len(self._wav_paths))
        if self._shuffle:
            np.random.shuffle(self._indexes)


class CommonDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(CommonDataLoader, self).__init__(config)
        wav_paths = glob('{}/**'.format(self.config.data.src_root),
                         recursive=True)
        wav_paths = [x.replace(os.sep, '/')
                     for x in wav_paths if '.wav' in x]

        # Get labels
        classes = sorted(os.listdir(self.config.data.src_root))
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)
        labels = [os.path.split(x)[0].split('/')[-1]
                  for x in wav_paths]
        labels = label_encoder.transform(labels)
        self._wav_train, self._wav_val, self._label_train, self._label_val = \
            train_test_split(wav_paths, labels, test_size=0.1, random_state=42)

    def get_train_data(self):
        return DataGenerator(self._wav_train,
                             self._label_train,
                             self.config.data.sample_rate,
                             self.config.data.duration_time,
                             len(set(self._label_train)),
                             batch_size=self.config.data.batch_size)

    def get_test_data(self):
        return DataGenerator(self._wav_val,
                             self._label_val,
                             self.config.data.sample_rate,
                             self.config.data.duration_time,
                             len(set(self._label_val)),
                             batch_size=self.config.data.batch_size)