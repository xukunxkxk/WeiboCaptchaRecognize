# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn
from keras.models import Model, load_model
from keras.layers.core import Activation, Dense, Dropout
from keras import backend as K
from keras.layers import Input, TimeDistributed, Bidirectional, RepeatVector, Reshape, Lambda, Permute
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.recurrent import LSTM, GRU
from keras.layers import GlobalAveragePooling2D
from keras.layers.merge import concatenate
from network.rnn.attention import Attention


class LSTMNet(object):
    @staticmethod
    def build(width, height, depth, char_nums, classes, l2_rate):
        """
        Build CNN network like vgg and use GlobalAveragePooling to reduce parameters
        :param width: Integer. The width of the image
        :param height: Integer. The height of the image
        :param depth: Integer. The depth of the image
        :param l2_rate: Float. L2 regularizer rate of the conv layer.
        :param char_nums: Integers. Numbers of char of the captcha. For example '2E4r2' char_nums is 5
        :param classes: Integers. Numbers of corpus of the captcha. if (A-Z a-z 0-9) >> 26 + 26 + 10 = 62
        """
        # define cnn part
        input_shape = height, width, depth
        chan_dim = -1
        # if using channels first change the order of input shape
        if K.image_data_format() == 'channels_first':
            input_shape = depth, height, width
            chan_dim = 1

        inputs = Input(shape=input_shape)
        layer = inputs
        # define cnn arc like vgg
        for i in range(4):
            layer = Conv2D(32 * 2 ** i, (3, 3), padding='same', input_shape=input_shape,
                           kernel_regularizer=l2(l2_rate), activation='relu')(layer)
            layer = BatchNormalization(axis=chan_dim)(layer)
            layer = Conv2D(32 * 2 ** i, (3, 3), padding='same', input_shape=input_shape,
                           kernel_regularizer=l2(l2_rate), activation='relu')(layer)
            layer = BatchNormalization(axis=chan_dim)(layer)
            layer = MaxPooling2D(pool_size=(2, 2))(layer)
            layer = Dropout(rate=0.25)(layer)
        layer = Conv2D(512, (3, 3), padding='same', input_shape=input_shape,
                       kernel_regularizer=l2(l2_rate), activation='relu')(layer)
        layer = BatchNormalization(axis=chan_dim)(layer)
        layer = Conv2D(512, (3, 3), padding='same', input_shape=input_shape,
                       kernel_regularizer=l2(l2_rate), activation='relu')(layer)
        layer = BatchNormalization(axis=chan_dim)(layer)
        layer = GlobalAveragePooling2D(K.image_data_format())(layer)
        layer = Dropout(rate=0.25)(layer)

        # define LSTM part
        layer = RepeatVector(char_nums)(layer)
        layer = Bidirectional(GRU(128, return_sequences=True), merge_mode='sum')(layer)
        layer = Bidirectional(GRU(128, return_sequences=True), merge_mode='concat')(layer)
        layer = Dropout(rate=0.25)(layer)
        layer = TimeDistributed(Dense(classes))(layer)
        layer = Activation('softmax')(layer)
        model = Model(inputs=inputs, outputs=layer)
        return model

    @staticmethod
    def transfer_build(char_nums, classes, model_path, layers_num, l2_rate):
        """
        build the network with pre-trained CNN.
        :param char_nums: Integers. Numbers of char of the captcha. For example '2E4r2' char_nums is 5
        :param classes: Integers. Numbers of corpus of the captcha. if (A-Z a-z 0-9) >> 26 + 26 + 10 = 62
        :param model_path: Model of the pre-trained CNN.
        :param layers_num: layers of the features used
        :param l2_rate: L2 regularizer rate of the conv layer.
        :return:
        """
        model = load_model(model_path)
        for layer in model.layers:
            layer.trainable = False
        inputs = model.inputs
        layer = model.layers[len(model.layers) - layers_num - 1].output

        # define LSTM part
        layer = RepeatVector(char_nums)(layer)
        layer = Bidirectional(GRU(128, return_sequences=True,
                                  kernel_regularizer=l2(l2_rate)), merge_mode='sum')(layer)
        layer = Bidirectional(GRU(128, return_sequences=True,
                                  kernel_regularizer=l2(l2_rate)), merge_mode='concat')(layer)
        layer = Dropout(0.25, name='lstm_drop_out')(layer)
        layer = TimeDistributed(Dense(classes))(layer)
        layer = Activation('softmax', name='activation_lstm')(layer)
        model = Model(inputs=inputs, outputs=layer)
        return model


class AttentionLSTMNet(object):
    @staticmethod
    def cnn(width, height, depth, l2_rate):
        """
        :param width: Integer. The width of the image
        :param height: Integer. The height of the image
        :param depth: Integer. The depth of the image
        :param l2_rate: Float. L2 regularizer rate of the conv layer.
        """
        # define cnn part
        input_shape = height, width, depth
        chan_dim = -1
        # if using channels first change the order of input shape
        if K.image_data_format() == 'channels_first':
            input_shape = depth, height, width
            chan_dim = 1

        inputs = Input(shape=input_shape)

        layer = inputs
        # define cnn arc like vgg
        for i in range(3):
            layer = Conv2D(32 * 2 ** i, (3, 3), padding='same', input_shape=input_shape,
                           kernel_regularizer=l2(l2_rate), activation='relu')(layer)
            layer = BatchNormalization(axis=chan_dim)(layer)
            layer = Conv2D(32 * 2 ** i, (3, 3), padding='same', input_shape=input_shape,
                           kernel_regularizer=l2(l2_rate), activation='relu')(layer)
            layer = BatchNormalization(axis=chan_dim)(layer)
            layer = MaxPooling2D(pool_size=(2, 2))(layer)
            layer = Dropout(rate=0.25)(layer)
        # permute height and width
        layer = Permute((2, 1, 3))(layer)
        conv_shape = layer.get_shape()
        layer = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(layer)
        return inputs, layer, int(conv_shape[1])


    @staticmethod
    def build(width, height, depth, char_nums, classes, l2_rate):
        """
        build the network using attention mechanism
        :param width: Integer. The width of the image
        :param height: Integer. The height of the image
        :param depth: Integer. The depth of the image
        :param char_nums: Integers. Numbers of char of the captcha. For example '2E4r2' char_nums is 5
        :param classes: Integers. Numbers of corpus of the captcha. if (A-Z a-z 0-9) >> 26 + 26 + 10 = 62
        :param l2_rate: L2 regularizer rate of the conv layer.
        :return:
        """
        inputs, layer, shape = AttentionLSTMNet.cnn(width, height, depth, l2_rate)
        layer = Bidirectional(LSTM(128, return_sequences=True))(layer)
        outputs = []
        for i in range(char_nums):
            output = Attention()(layer)
            output = Dense(classes, activation='softmax')(output)
            outputs.append(output)
        layer = concatenate(outputs)
        model = Model(inputs=inputs, outputs=layer)
        return model


class CTCNet(object):
    @staticmethod
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    @staticmethod
    def build(width, height, depth, char_nums, classes, l2_rate):
        """
        build the network using CTC loss
        :param width: Integer. The width of the image
        :param height: Integer. The height of the image
        :param depth: Integer. The depth of the image
        :param char_nums: Integers. Numbers of char of the captcha. For example '2E4r2' char_nums is 5
        :param classes: Integers. Numbers of corpus of the captcha. if (A-Z a-z 0-9) >> 26 + 26 + 10 = 62
        :param l2_rate: L2 regularizer rate of the conv layer.
        :return:
        """
        # define cnn part
        input_shape = height, width, depth
        chan_dim = -1
        # if using channels first change the order of input shape
        if K.image_data_format() == 'channels_first':
            input_shape = depth, height, width
            chan_dim = 1

        inputs = Input(shape=input_shape)

        layer = inputs
        # define cnn arc like vgg
        for i in range(3):
            layer = Conv2D(32 * 2 ** i, (3, 3), padding='same', input_shape=input_shape,
                           kernel_regularizer=l2(l2_rate), activation='relu')(layer)
            layer = BatchNormalization(axis=chan_dim)(layer)
            layer = Conv2D(32 * 2 ** i, (3, 3), padding='same', input_shape=input_shape,
                           kernel_regularizer=l2(l2_rate), activation='relu')(layer)
            layer = BatchNormalization(axis=chan_dim)(layer)
            layer = MaxPooling2D(pool_size=(2, 2))(layer)
            layer = Dropout(rate=0.25)(layer)
        # permute height and width
        layer = Permute((2, 1, 3))(layer)

        conv_shape = layer.get_shape()
        layer = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(layer)
        layer = Dense(128, activation='relu')(layer)

        # define LSTM part
        layer = Bidirectional(LSTM(128, return_sequences=True,
                                   kernel_regularizer=l2(l2_rate)), merge_mode='sum')(layer)
        layer = Bidirectional(LSTM(128, return_sequences=True,
                                   kernel_regularizer=l2(l2_rate)), merge_mode='concat')(layer)
        layer = Dropout(0.25)(layer)
        # when using CTC, add class space
        layer = Dense(classes+1, activation='softmax')(layer)
        predict_model = Model(inputs=inputs, outputs=layer)

        labels = Input(name='labels', shape=[char_nums], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(CTCNet.ctc_lambda_func, output_shape=(1,), name='ctc')\
            ([layer, labels, input_length, label_length])
        train_model = Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
        return predict_model, train_model, conv_shape[1]


if __name__ == '__main__':
    pass