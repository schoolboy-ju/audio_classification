import os

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
import tensorflow as tf

from base.base_model import BaseModel


class UrbanConvNet(BaseModel):
    def __init__(self, config):
        super(UrbanConvNet, self).__init__(config)
        self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=int(self.config.data.sample_rate *
                                        self.config.data.duration_time),
                              name='input')
        x = layers.Reshape((1, int(self.config.data.sample_rate *
                                   self.config.data.duration_time)))(inputs)
        x = Melspectrogram(n_dft=self.config.model.n_dft,
                           n_hop=self.config.model.n_hop,
                           padding=self.config.model.padding,
                           sr=self.config.data.sample_rate,
                           n_mels=self.config.model.n_mels,
                           fmin=0.0,
                           fmax=self.config.data.sample_rate / 2,
                           power_melgram=self.config.model.power_melgram,
                           return_decibel_melgram=
                           self.config.model.return_decibel_melgram,
                           trainable_fb=self.config.model.trainable_fb,
                           trainable_kernel=self.config.model.trainable_kernel,
                           name='melbands')(x)
        x = Normalization2D(str_axis='batch', name='batch_norm')(x)
        x = layers.Conv2D(32, (3, 3), padding='same', name='conv1',
                          kernel_initializer='he_normal')(x)
        x = layers.Conv2D(32, (3, 3), activation='elu', padding='same',
                          name='conv1_elu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 4))(x)  # (48, 47)
        x = layers.Dropout(0.2)(x)

        x = layers.ZeroPadding2D(padding=(0, 1))(x)  # (48, 49)

        x = layers.Conv2D(32, (3, 3), padding='same', name='conv2',
                          kernel_initializer='he_normal')(x)
        x = layers.Conv2D(32, (3, 3), activation='elu', padding='same',
                          name='conv2_elu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3))(x)  # (16, 16)
        x = layers.Dropout(rate=0.2)(x)

        x = layers.Conv2D(32, (3, 3), activation='elu', padding='same',
                          name='conv3_elu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)  # (8, 8)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(32, (3, 3), activation='elu', padding='same',
                          name='conv4_elu', kernel_initializer='he_normal')(x)

        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)  # (8, 8)
        x = layers.Dropout(rate=0.2)(x)

        x = layers.Conv2D(32, (3, 3), padding='valid',
                          name='conv5', kernel_initializer='he_normal')(x)
        x = layers.AveragePooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='elu')(x)
        x = layers.Dropout(rate=0.2)(x)
        outputs = layers.Dense(self.config.data.n_classes,
                               activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=outputs,
                           name='urban_convnet')
        adam = optimizers.Adam(lr=0.005)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])
