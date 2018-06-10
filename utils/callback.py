# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn

from keras.callbacks import Callback
from keras.callbacks import BaseLogger
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class MonitorCallback(BaseLogger):
    def __init__(self, train_x, valid_x, train_y, valid_y,
                 file_path, char_nums, class_num, split,
                 json_path=None, start_at=0, name=None):
        """
        Monitor accuracy and loss after every epoch
        Plot the curve on local disk
        :param train_x: 4d array (images_num, height, width, channels). Training images.
        :param valid_x: 4d array (images_num, height, width, channels). Valid images.
        :param train_y: 1d array strings. Training labels, each element is latter or num of the captcha.
        :param valid_y: 1d array strings. Valid labels, each element is latter or num of the captcha.
        :param file_path: String. Path to save loss and accuracy images.
        :param char_nums: Integers. Numbers of char of the captcha. For example '2E4r2' char_nums is 5
        :param class_num: Integers. Numbers of corpus of the captcha. if (A-Z a-z 0-9) >> 26 + 26 + 10 = 62
        :param split: Boolean. Split the labels or not .
                For the captcha '1234', corpus is 0-9
                if split is True  the label will be (4, 10)[[1 0 0 ....], [0 1 0 ...], [0 0 1 ...], [0 0 0 1 ...]]
                if split is False the label will be (40, )[1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ...]
        :param json_path: String. Path to save loss and accuracy. When restarting train with given model checkpoint
                tis json file will be user for continuing logging loss and accuracy
        :param start_at: Integer. Restart epoch to continue train
        :param name: String. Net Name e.g. VGG ResNet
        """
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(MonitorCallback, self).__init__()
        self.train_x = train_x
        self.train_y = np.array(train_y)
        self.valid_x = valid_x
        self.valid_y = np.array(valid_y)
        self.file_path = file_path
        self.char_nums = char_nums
        self.class_num = class_num
        self.split = split
        self.json_path = json_path
        self.start_at = start_at
        self.name = name

    def on_train_begin(self, logs=None):
        # init history dic
        self.H = {}
        # if the JSON history path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())
                # check to see if a starting epoch was supplied
                if self.start_at > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]
                else:
                    for k in self.H.keys():
                        self.H[k] = []
            else:
                # set accuracy to empty list
                self.H["train_accuracy"] = []
                self.H["valid_accuracy"] = []
        else:
            # set accuracy to empty list
            self.H["train_accuracy"] = []
            self.H["valid_accuracy"] = []

    def on_epoch_end(self, epoch, logs=None):
        # calculate accuracy rate
        if self.split:
            # split chars training
            if self.name == 'LSTM':
                axis = 1
            else:
                axis = 0
            # calc accuracy
            pred_train = self.model.predict(self.train_x, batch_size=1024)
            pred_train = np.argmax(pred_train, axis=-1)
            ground_truth = np.argmax(self.train_y, axis=-1)
            train_accuracy = np.mean(np.sum(pred_train == ground_truth, axis=axis) // self.char_nums)

            pred_valid = self.model.predict(self.valid_x, batch_size=1024)
            pred_valid = np.argmax(pred_valid, axis=-1)
            ground_truth = np.argmax(self.valid_y, axis=-1)
            valid_accuracy = np.mean(np.sum(pred_valid == ground_truth, axis=axis) // self.char_nums)

            self.H["train_accuracy"].append(train_accuracy)
            self.H["valid_accuracy"].append(valid_accuracy)
        else:
            # calc accuracy
            pred_train = self.model.predict(self.train_x, batch_size=64)
            pred_train = np.asarray([np.argmax(pred_train[:, self.class_num * i:self.class_num * (i + 1)], axis=-1)
                                     for i in range(self.char_nums)])
            ground_truth = np.asarray(
                [np.argmax(self.train_y[:, self.class_num * i:self.class_num * (i + 1)], axis=-1)
                 for i in range(self.char_nums)])
            train_accuracy = np.mean(np.sum(pred_train == ground_truth, axis=0) // self.char_nums)

            pred_valid = self.model.predict(self.valid_x, batch_size=64)
            pred_valid = np.asarray([np.argmax(pred_valid[:, self.class_num * i:self.class_num * (i + 1)], axis=-1)
                                     for i in range(self.char_nums)])
            ground_truth = np.asarray(
                [np.argmax(self.valid_y[:, self.class_num * i:self.class_num * (i + 1)], axis=-1)
                 for i in range(self.char_nums)])
            valid_accuracy = np.mean(np.sum(pred_valid == ground_truth, axis=0) // self.char_nums)
            self.H["train_accuracy"].append(train_accuracy)
            self.H["valid_accuracy"].append(valid_accuracy)
        print('\roverall train accuracy: %s - valid accuracy: %s' % (str(train_accuracy),
                                                                     str(valid_accuracy)),
              end=100 * ' ' + '\n')
        # saving accuracy and loss figure
        for k, v in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        self._save_loss_accuracy()

    def _save_loss_accuracy(self):
        # save loss accuracy to json file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        # plot and save the accuracy and loss
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure(figsize=(20, 10))
            plt.subplot(211)
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.ylabel("Loss")

            plt.subplot(212)
            plt.plot(N, self.H["train_accuracy"], label="train_acc")
            plt.plot(N, self.H["valid_accuracy"], label="val_acc")
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend()
            # save the figure
            plt.savefig(os.path.sep.join([self.file_path, 'loss_accuracy.png']))
            plt.close()


class EpochCheckpointSaver(Callback):
    def __init__(self, name, every_epoch, model_path):
        """
        Save model checkpoint ever given epoch rounds
        :param name: String. Name of network model using
        :param every_epoch: Integer. Save model every_epoch rounds
        :param model_path: String. Path to saving model
        """
        super(EpochCheckpointSaver, self).__init__()
        self.every_epoch = every_epoch
        self.model_path = model_path
        self.name = name
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self._save_check_point(epoch)
        self.epoch += 1

    def on_train_end(self, logs=None):
        file_path = os.path.sep.join(
            [self.model_path, '{name}-epoch-{epoch:03d}.hdf5'.format(name=self.name, epoch=self.epoch)])
        self.model.save(file_path, overwrite=True)

    def _save_check_point(self, epoch):
        if epoch % self.every_epoch == 0:
            file_path = os.path.sep.join(
                [self.model_path, '{name}-epoch-{epoch:03d}.hdf5'.format(name=self.name, epoch=epoch + 1)])
            self.model.save(file_path, overwrite=True)


class CTCMonitor(BaseLogger):
    def __init__(self, train_x, valid_x, train_y, valid_y,
                 file_path, char_nums, class_num, base_model,
                 json_path=None, start_at=0):
        """
        This class for monitor accuracy and loss when using CTC loss
        Plot the curve on local disk
        :param train_x: 4d array (images_num, height, width, channels). Training images.
        :param valid_x: 4d array (images_num, height, width, channels). Valid images.
        :param train_y: 1d array strings. Training labels, each element is latter or num of the captcha.
        :param valid_y: 1d array strings. Valid labels, each element is latter or num of the captcha.
        :param file_path: String. Path to save loss and accuracy images.
        :param char_nums: Integers. Numbers of char of the captcha. For example '2E4r2' char_nums is 5
        :param class_num: Integers. Numbers of corpus of the captcha. if (A-Z a-z 0-9) >> 26 + 26 + 10 = 62
        :param json_path: String. Path to save loss and accuracy. When restarting train with given model checkpoint
                tis json file will be user for continuing logging loss and accuracy
        :param start_at: Integer. Restart epoch to continue train
        """
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(CTCMonitor, self).__init__()
        self.train_x = train_x
        self.train_y = np.array(train_y)
        self.valid_x = valid_x
        self.valid_y = np.array(valid_y)
        self.file_path = file_path
        self.char_nums = char_nums
        self.class_num = class_num
        self.json_path = json_path
        self.start_at = start_at
        self.predict_model = base_model

    def on_train_begin(self, logs=None):
        # init history dic
        self.H = {}
        # if the JSON history path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())
                # check to see if a starting epoch was supplied
                if self.start_at > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]
                else:
                    for k in self.H.keys():
                        self.H[k] = []
            else:
                self.H["train_accuracy"] = []
                self.H["valid_accuracy"] = []
        else:
            self.H["train_accuracy"] = []
            self.H["valid_accuracy"] = []

    def on_epoch_end(self, epoch, logs=None):

        # calculate accuracy rate
        batch_size = 32
        train_accuracy = 0
        pred_train = self.predict_model.predict(self.train_x, batch_size=batch_size)
        shape = pred_train[:, 2:, :].shape
        ctc_decode = K.ctc_decode(pred_train[:, 2:, :], input_length=np.full(shape[0], shape[1]))[0][0]
        out = K.get_value(ctc_decode)[:, :self.char_nums]
        if out.shape[1] == self.char_nums:
            train_accuracy = np.mean(np.sum(self.train_y == out, axis=1) // self.char_nums)

        valid_accuracy = 0
        pred_valid = self.predict_model.predict(self.valid_x, batch_size=batch_size)
        shape = pred_valid[:, 2:, :].shape
        ctc_decode = K.ctc_decode(pred_valid[:, 2:, :], input_length=np.full(shape[0], shape[1]))[0][0]
        out = K.get_value(ctc_decode)[:, :self.char_nums]
        if out.shape[1] == self.char_nums:
            valid_accuracy = np.mean(np.sum(self.valid_y == out, axis=1) // self.char_nums)

        self.H["train_accuracy"].append(train_accuracy)
        self.H["valid_accuracy"].append(valid_accuracy)

        print('\roverall train accuracy: %s - valid accuracy: %s' % (str(train_accuracy),
                                                                     str(valid_accuracy)),
              end=100 * ' ' + '\n')

        # saving accuracy and loss figure
        for k, v in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        self._save_loss_accuracy()

    def _save_loss_accuracy(self):
        # save loss accuracy to json file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure(figsize=(20, 10))
            plt.subplot(211)
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.ylabel("Loss")

            plt.subplot(212)
            plt.plot(N, self.H["train_accuracy"], label="train_acc")
            plt.plot(N, self.H["valid_accuracy"], label="val_acc")
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend()
            # save the figure
            plt.savefig(os.path.sep.join([self.file_path, 'loss_accuracy.png']))
            plt.close()


if __name__ == '__main__':
    pass
