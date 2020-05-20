import sys
import os
import h5py

from base.base_data_loader import BaseDataLoader


class DataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        self._train = None
        self._validation = None
        self._test = None

    def get_train_data(self):
        self._train = h5py.File(self.config.data.train_hdf_path, 'r')
        return self._train['raw'], self._train['label']

    def get_validation_data(self):
        self._validation = h5py.File(self.config.data.validation_hdf_path, 'r')
        return self._validation['raw'], self._validation['label']

    def get_test_data(self):
        self._test = h5py.File(self.config.data.test_hdf_path, 'r')
        return self._test['raw'], self._test['label']

    def close_train_data_file(self):
        self._train.close()

    def close_validation_data_file(self):
        self._validation.close()

    def close_test_data_file(self):
        self._test.close()
