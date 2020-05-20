from base.base_trainer import BaseTrain
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class CommonTrainer(BaseTrain):
    def __init__(self, model, train_data, val_data, config):
        super(CommonTrainer, self).__init__(model, train_data, val_data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self._init_callbacks()

    def _init_callbacks(self):

        if not os.path.exists(self.config.callbacks.checkpoint_dir):
            os.makedirs(self.config.callbacks.checkpoint_dir)

        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config.callbacks.checkpoint_dir,
                    '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name
                ),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=
                self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )




    def train(self):
        history = self.model.fit(
            self.train_data[0], self.train_data[1],
            validation_data=(self.val_data[0], self.val_data[1]),
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks,
            shuffle=False
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_accuracy'])
