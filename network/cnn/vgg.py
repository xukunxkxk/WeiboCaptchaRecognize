# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn
from keras.models import Model
from keras.layers.core import Dropout
from keras import backend as K
from keras.layers import Input
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D


class VGGNet(object):
    @staticmethod
    def conv_module(x, filters, cha_dim, l2_rate, name=None):
        """
        building CONV module
        :param x: last layer
        :param filters: 1d array. filters nums of each stage
        :param cha_dim: Integer. -1 stands for channel last, 1 stands for channel first
        :param l2_rate: Float. L2 regularizer rate of the conv layer.
        :param name: String. Name of the stage
        """
        # initialize the CONV, BN, and RELU layer names
        conv_name, bn_name = None, None

        # if a layer name was supplied, prepend it
        if name is not None:
            conv_name = name + "_conv"
            bn_name = name + "_bn"

        # define a CONV > BN
        x = Conv2D(filters, (3, 3), kernel_regularizer=l2(l2_rate), activation='relu', padding='same',
                   name=conv_name)(x)
        x = BatchNormalization(axis=cha_dim, name=bn_name)(x)
        return x

    @staticmethod
    def build(width, height, depth, conv_nums, filters, l2_rate):
        """
        Build CNN like vgg
        use GlobalAveragePooling to reduce parameters
        :param width: Integer. The width of the image
        :param height: Integer. The height of the image
        :param depth: Integer. The depth of the image
        :param conv_nums: 1d array. Conv nums  of each stage
        :param filters: 1d array. filters nums of each stage
        :param l2_rate: Float. L2 regularizer rate of the conv layer.
        """
        input_shape = height, width, depth
        chan_dim = -1
        # if using channels first change the order of input shape
        if K.image_data_format() == 'channels_first':
            input_shape = depth, height, width
            chan_dim = 1

        input_image = Input(shape=input_shape)
        layer = input_image
        # define cnn arc like vgg
        for block, (conv_num, filter) in enumerate(zip(conv_nums, filters)):
            for stage in range(conv_num):
                layer = VGGNet.conv_module(layer, filter, chan_dim, l2_rate,
                                           name='block_{0}_stage_{1}'.format(block, stage))
            # add drop out after each block and pooling
            if block == len(conv_nums) - 1:
                layer = GlobalAveragePooling2D(K.image_data_format())(layer)
                layer = Dropout(rate=0.5)(layer)
            else:
                layer = MaxPooling2D(pool_size=(2, 2))(layer)
                layer = Dropout(rate=0.25)(layer)

        # outputs
        model = Model(inputs=input_image, outputs=layer)
        return model

    
if __name__ == '__main__':
    pass
