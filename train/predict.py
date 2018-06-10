# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn


if __name__ == '__main__':
    from config.config import *
    from keras.models import load_model
    from keras.preprocessing.image import img_to_array
    import cv2
    import numpy as np
    import argparse

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=CNN_MODEL, help="path to output model")
    ap.add_argument("-d", "--dataset", default=IMAGES_PATH, help="path to input dataset")
    ap.add_argument("-p", "--image_path", require=True, help="path to input dataset")
    args = vars(ap.parse_args())

    model = load_model(args['model'])
    image = cv2.imread(args['image_path'])
    image = img_to_array(image) / 255.0
    image = image[np.newaxis, :]
    label = model.predict(image)
    label = np.squeeze(label)
    label = [REVERSE_CLASSES_DICT[np.argmax(label[pos * NUM_CLASSES:(pos + 1) * NUM_CLASSES])]
             for pos in range(CHAR_NUMBERS)]

    print(label)



