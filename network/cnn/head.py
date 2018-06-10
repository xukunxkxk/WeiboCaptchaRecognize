# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn
from keras.models import Model
from keras.layers.core import Activation, Dense
from keras.layers.merge import concatenate


class FCHead(object):
    @staticmethod
    def build(base_model, char_nums, classes, sep=True):
        """
        Predict layer of CNN
        :param base_model: CNN
        :param char_nums: Integers. Numbers of char of the captcha. For example '2E4r2' char_nums is 5
        :param classes: Integers. Numbers of corpus of the captcha. if (A-Z a-z 0-9) >> 26 + 26 + 10 = 62
        :param sep: Boolean. split the label or not
        """
        # outputs
        outputs = []
        x = base_model.output
        for i in range(char_nums):
            output = Dense(classes)(x)
            output = Activation("softmax", name=str(i))(output)
            outputs.append(output)
        if not sep:
            outputs = concatenate(outputs)
        model = Model(inputs=base_model.input, outputs=outputs)
        return model


if __name__ == '__main__':
    pass
