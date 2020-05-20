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


class Conv1D(BaseModel):
    def __init__(self, config):
        super(Conv1D, self).__init__(config)
        self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=(1, int(self.config.data.sample_rate *
                                            self.config.data.duration_time)),
                              name='input')
        x = Melspectrogram(n_dft=512, n_hop=160, padding='same',
                           sr=self.config.data.sample_rate, n_mels=128,
                           fmin=0.0, fmax=self.config.data.sample_rate / 2,
                           power_melgram=1.0, return_decibel_melgram=True,
                           trainable_fb=False, trainable_kernel=False,
                           name='melbands')(inputs)
        x = Normalization2D(str_axis='batch', name='batch_norm')(x)
        x = layers.Permute((2, 1, 3), name='permute')(x)
        x = TimeDistributed(layers.Conv1D(8, kernel_size=4, activation='tanh'),
                            name='td_conv_1d_tanh')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_2d_1')(x)
        x = TimeDistributed(layers.Conv1D(16, kernel_size=4, activation='relu'),
                            name='td_conv_1d_relu_1')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_2d_2')(x)
        x = TimeDistributed(layers.Conv1D(32, kernel_size=4, activation='relu'),
                            name='td_conv_1d_relu_2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_2d_3')(x)
        x = TimeDistributed(layers.Conv1D(64, kernel_size=4, activation='relu'),
                            name='td_conv_1d_relu_3')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_2d_4')(x)
        x = TimeDistributed(layers.Conv1D(128, kernel_size=4, activation='relu'),
                            name='td_conv_1d_relu_4')(x)
        x = layers.GlobalMaxPooling2D(name='global_max_pooling_2d')(x)
        x = layers.Dropout(rate=0.1, name='dropout')(x)
        x = layers.Dense(64, activation='relu',
                         activity_regularizer=l2(self.config.model.learning_rate),
                         name='dense')(x)
        outputs = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='1d_convolution')
        self.model.compile(optimizer=self.config.model.optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
