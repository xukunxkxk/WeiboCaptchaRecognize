# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn
from keras.models import Model
from keras.layers.core import Activation
from keras import backend as K
from keras.layers import Input, add
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D


class ResNet(object):
    @staticmethod
    def build(width, height, depth, stages, filters, reg=0.0001, bn_eps=2e-5, bn_mom=0.9):
        '''
        :param width: Integer. The width of the image
        :param height: Integer. The height of the image
        :param depth: Integer. The depth of the image
        :param stages: Integer. The nums of residual models in each stage
        :param filters: Integer. The filters of residual models in conv layer
        :param reg: Integer. The width of the image
        :param bn_eps: Float. Prevent divide by zero error
        :param bn_mom: Float. Prevent divide by zero error
        :return:
        '''
        input_shape = height, width, depth
        cha_dim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            cha_dim = 1

        inputs = Input(shape=input_shape)
        # like a level of normalization
        x = BatchNormalization(axis=cha_dim, epsilon=bn_eps, momentum=bn_mom)(inputs)
        # check if we are utilizing the CIFAR dataset
        # adding some conv and pooling layer before stacked residual module
        # apply a single CONV layer
        x = Conv2D(filters[0], (3, 3), use_bias=False,
                   padding="same", kernel_regularizer=l2(reg))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                                       cha_dim, red=True, bn_eps=bn_eps, bn_mom=bn_mom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1],
                                           (1, 1), cha_dim, bn_eps=bn_eps, bn_mom=bn_mom)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=cha_dim, epsilon=bn_eps, momentum=bn_mom)(x)
        x = Activation("relu")(x)
        x = GlobalAveragePooling2D(K.image_data_format())(x)

        # outputs
        model = Model(inputs=inputs, outputs=x)
        return model

    @staticmethod
    def residual_module(data, K, stride, chan_dim, red=False, reg=1e-4, bn_eps=2e-5, bn_mom=0.9):
        # the shortcut branch of the ResNet module should be initialize as the input (identity) data
        shortcut = data

        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(data)
        act1 = Activation("relu")(bn1)
        # According to He etal., the biases are in the BN layers that immediately follow the convolutions
        # so there is no need to introduce a second bias term.
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        # the third block of the ResNet module is another set of 1x1 CONVs
        bn3 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])
        return x


if __name__ == '__main__':
    pass
