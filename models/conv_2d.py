import os

from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
import tensorflow as tf

from base.base_model import BaseModel

N_CLASSES = len(os.listdir('wave_files'))


class Conv2D(BaseModel):
    def __init__(self, config):
        super(Conv2D, self).__init__(config)
        self._sample_rate = self.config.data.sample_rate
        self._duration_time = self.config.data.duration_time
        self.build_model()

    def build_model(self):
        inputs = layers.Input(
            shape=(1, int(self._sample_rate * self._duration_time)),
            name='input')
        x = Melspectrogram(n_dft=512, n_hop=160, padding='same',
                           sr=self._sample_rate, n_mels=128,
                           fmin=0.0, fmax=self._sample_rate / 2,
                           power_melgram=1.0,
                           return_decibel_melgram=True, trainable_fb=False,
                           trainable_kernel=False,
                           name='melbands')(inputs)
        x = Normalization2D(str_axis='batch', name='batch_norm')(x)
        x = layers.Conv2D(8, kernel_size=(7, 7), activation='tanh',
                          padding='same', name='conv2d_tanh')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same',
                                name='max_pool_2d_1')(x)
        x = layers.Conv2D(16, kernel_size=(5, 5), activation='relu',
                          padding='same', name='conv2d_relu_1')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same',
                                name='max_pool_2d_2')(x)
        x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu',
                          padding='same', name='conv2d_relu_2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same',
                                name='max_pool_2d_3')(x)
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                          padding='same', name='conv2d_relu_3')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same',
                                name='max_pool_2d_4')(x)
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                          padding='same', name='conv2d_relu_4')(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dropout(rate=0.2, name='dropout')(x)
        x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001),
                         name='dense')(x)
        outputs = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(
            x)

        self.model = Model(inputs=inputs, outputs=outputs,
                           name='2d_convolution')
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
