# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn
import numpy as np
from config.config import *
from sklearn.model_selection import train_test_split
from imutils import paths
from keras.preprocessing.image import img_to_array
import cv2
import os


def read_data(image_path, split=False, vector=True):
    '''
    Read images and labels from disk and random split the dataset to train set and labels set
    :param image_path: String. Path of images on the disk
    :param split: Boolean. Split the label or not
    :param vector: Boolean. Whether to one-hot encoding label. When usign ctc loss this param need set False
    :return: train images, valid images, train labels, valid labels,

    Captcha on the disk should be organized like image_path/'342f3'.jpg(png, ...)  342f3 represent the true the
    ground truth of the captcha
    '''

    class_num = NUM_CLASSES
    label_dict = CLASSES_DICT
    data = []
    labels = []
    # loop over the input images
    for image_path in paths.list_images(image_path):
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(image_path)
        # image = preprocess(image, padding)
        image = img_to_array(image)
        data.append(image)
        # extract the class label from the image path and update the
        # labels list
        label = image_path.split(os.path.sep)[-1].split('.')[0].lower()
        label = [label_dict[x] for x in label]
        # extend each character
        one_hot_labels = []
        for each_label in label:
            if vector:
                # one hot encoding
                one_hot_label = [0] * class_num
                one_hot_label[each_label] = 1
                if split:
                    one_hot_labels.append(one_hot_label)
                else:
                    # extend the labels
                    one_hot_labels.extend(one_hot_label)
            else:
                one_hot_labels.append(each_label)
        labels.append(one_hot_labels)
    labels = np.array(labels)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # gen train, valid
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=10000, train_size=40000,
                                                    random_state=42)

    return trainX, testX, trainY, testY


if __name__ == '__main__':
    pass