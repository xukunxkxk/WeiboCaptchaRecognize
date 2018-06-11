# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn
from keras.models import Model
from keras.layers.core import Activation, Dropout
from keras import backend as K
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D


class GoogLeNet(object):
    @staticmethod
    def build(width, height, depth, reg=0.0005):
        """
        Build CNN network like GoogLeNet and use GlobalAveragePooling to reduce parameters
        :param width: Integer. The width of the image
        :param height: Integer. The height of the image
        :param depth: Integer. The depth of the image
        :param l2_rate: Float. L2 regularizer rate of the conv layer.
        """
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        input_shape = height, width, depth
        cha_dim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            cha_dim = 1

        inputs = Input(shape=input_shape)
        x = GoogLeNet.conv_module(inputs, 32, (3, 3), (1, 1), cha_dim, reg=reg, name="block1")
        x = MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)
        x = GoogLeNet.conv_module(x, 32, (1, 1), (1, 1), cha_dim, reg=reg, name="block2")
        x = GoogLeNet.conv_module(x, 32, (3, 3), (1, 1), cha_dim, reg=reg, name="block3")
        x = MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(x)

        # apply two Inception modules followed by a POOL
        x = GoogLeNet.inception_module(x, 64, 32, 64, 16, 32, 64, 64, cha_dim, "3", reg=reg)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="pool3")(x)

        x = GoogLeNet.inception_module(x, 128, 32, 128, 24, 48, 64, 64, cha_dim, "4", reg=reg)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(x)

        x = GoogLeNet.inception_module(x, 128, 32, 128, 24, 48, 64, 64, cha_dim, "5", reg=reg)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="pool5")(x)

        x = GoogLeNet.inception_module(x, 256, 112, 256, 32, 64, 128, 128, cha_dim, "6", reg=reg)
        # apply a POOL layer (average) followed by dropout
        x = GlobalAveragePooling2D(K.image_data_format())(x)
        x = Dropout(0.5, name="do")(x)

        # outputs
        model = Model(inputs=inputs, outputs=x)
        return model

    @staticmethod
    def conv_module(x, filters, kernel_size, stride, cha_dim, padding='same',
                    reg=5e-4, name=None):
        # initialize the CONV, BN, and RELU layer names
        conv_name, bn_name, act_name = None, None, None

        # if a layer name was supplied, prepend it
        if name is not None:
            conv_name = name + "_conv"
            bn_name = name + "_bn"
            act_name = name + "_act"
        # define a CONV > BN > RELU pattern
        x = Conv2D(filters, kernel_size, strides=stride, padding=padding,
                   kernel_regularizer=l2(reg), name=conv_name)(x)
        x = Activation('relu', name=act_name)(x)
        x = BatchNormalization(axis=cha_dim, name=bn_name)(x)
        return x

    @staticmethod
    def inception_module(x, num_1_1x1, num_2_1x1, num_2_3x3,
                         num_3_1x1, num_3_3x3_1, num_3_3x3_2,
                         num_4_1x1, cha_dim, stage, reg=5e-5):
        # define the first branch of the Inception module which
        # consists of 1x1 convolutions
        # learn local features.
        first = GoogLeNet.conv_module(x, num_1_1x1, (1, 1), (1, 1), cha_dim, reg=reg, name=stage + "_first")

        # define the second branch of the Inception module which
        # consists of 1x1 and 3x3 convolutions
        second = GoogLeNet.conv_module(x, num_2_1x1, (1, 1), (1, 1), cha_dim,
                                             reg=reg, name=stage + "_second1")

        second = GoogLeNet.conv_module(second, num_2_3x3, (3, 3), (1, 1), cha_dim,
                                             reg=reg, name=stage + "_second2")

        # define the third branch of the Inception module which
        # are our 1x1 and 5x5 convolutions
        third = GoogLeNet.conv_module(x, num_3_1x1, (1, 1), (1, 1), cha_dim,
                                            reg=reg, name=stage + "_third1")
        third = GoogLeNet.conv_module(third, num_3_3x3_1, (3, 3), (1, 1), cha_dim,
                                            reg=reg, name=stage + "_third2")
        third = GoogLeNet.conv_module(third, num_3_3x3_2, (3, 3), (1, 1), cha_dim,
                                            reg=reg, name=stage + "_third3")

        # define the fourth branch of the Inception module which
        # is the POOL projection
        fourth = MaxPooling2D((2, 2), strides=(1, 1), padding="same", name=stage + "_pool")(x)
        fourth = GoogLeNet.conv_module(fourth, num_4_1x1, (1, 1), (1, 1), cha_dim,
                                             reg=reg, name=stage + "_fourth")

        x = concatenate([first, second, third, fourth], axis=cha_dim, name=stage + "_mixed")

        return x

if __name__ == '__main__':
    pass
