# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn
import numpy as np

class Generator(object):
    def __init__(self, images, labels, batch_size, char_num, sep, aug=None):
        '''
        :param images: 4D array. Images of the captcha
        :param labels: 1D array.
        :param batch_size: Integers. Train batch
        :param char_num: Integers. Char num of the captcha
        :param sep: Boolean, split the labels or not
        :param aug: Boolean. Using data aug or not
        '''
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.images = images
        self.labels = labels
        self.aug = aug
        self.batch_size = batch_size
        self.char_num = char_num
        self.num_images = np.shape(self.images)[0]
        self.sep = sep

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0
        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            for i in np.arange(0, self.num_images, self.batch_size):
                # extract the images and labels from the HDF dataset
                images = self.images[i: i + self.batch_size]
                labels = self.labels[i: i + self.batch_size]

                # if the data augmenator exists, apply it
                if self.aug is not None:
                    images, labels = next(self.aug.flow(images, labels, batch_size=self.batch_size))

                # yield a tuple of images and labels
                if self.sep:
                    labels = [labels[:, i, :] for i in range(self.char_num)]
                yield images, labels

            # increment the total number of epochs
            epochs += 1


class CTCGenerator(object):
    def __init__(self, images, labels, batch_size, features_num, char_num, aug=None):
        '''
        :param images: 4D array. Images of the captcha
        :param labels: 1D array.
        :param batch_size: Integers.
        :param features_num: Integers. cnn output columns number
        :param char_num: Integers. Char num of the captcha
        :param aug: Boolean. Using data aug or not
        '''
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.images = images
        self.labels = labels
        self.aug = aug
        self.batch_size = batch_size
        self.char_num = char_num
        self.num_images = np.shape(self.images)[0]
        self.feature_num = features_num

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0
        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            for i in np.arange(0, self.num_images, self.batch_size):
                # extract the images and labels from the HDF dataset
                images = self.images[i: i + self.batch_size]
                labels = self.labels[i: i + self.batch_size]

                # if the data augmenator exists, apply it
                if self.aug is not None:
                    images, labels = next(self.aug.flow(images, labels, batch_size=self.batch_size))
                # yield a tuple of images / labels / feature nums / char nums /
                size = np.shape(images)[0]
                yield [images, labels,
                       np.full(size, self.feature_num-2),
                       np.full(size, self.char_num)], np.ones(size)

            # increment the total number of epochs
            epochs += 1


if __name__ == '__main__':
    pass