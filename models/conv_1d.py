from base.base_model import BaseModel

from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
import tensorflow as tf


class Conv1D(BaseModel):
    def __init__(self, config):
        super(Conv1D, self).__init__(config)
        self.build_model()

    def build_model(self):
        # TODO(joohyun): keras model code here
        pass