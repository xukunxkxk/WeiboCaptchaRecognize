# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn


if __name__ == '__main__':
    from config.config import *
    from network.rnn.lstm import CTCNet
    from utils.read_features import read_data
    from utils.generator import CTCGenerator
    from utils.callback import CTCMonitor
    from keras.utils import plot_model
    from keras.optimizers import SGD
    import argparse
    import os

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default=IMAGES_PATH, help="path to input dataset")
    ap.add_argument("-m", "--model", default=MODEL_PATH, help="path to output model")
    ap.add_argument("-o", "--output", default=OUTPUT_PATH, help="path to accuracy and loss plot")
    ap.add_argument("-c", "--checkpoint", type=str, default=CHECKPOINT, help='path to specific checkpoint '
                                                                             'to restart training')
    ap.add_argument("-s", "--start-epoch", type=int, default=START_EPOCH, help='epoch to restart training at')
    args = vars(ap.parse_args())

    class_num = NUM_CLASSES
    char_nums = CHAR_NUMBERS

    epoch = EPOCH
    net = 'LSTM_with_ctc_loss'
    split = SPLIT
    using_aug = USING_AUG

    # reading data
    trainX, testX, trainY, testY = read_data(image_path=args["dataset"], split=split, vector=False)
    predict_model, train_model, features_shape = CTCNet.build(WIDTH, HEIGHT, DEPTH, char_nums, class_num, 2e-4)

    # plot the model
    plot_model(train_model, to_file=os.path.sep.join([args['output'], net + ".png"]), show_shapes=True)

    json_path = os.path.sep.join([args['model'], '{}_log.json'.format(net)])

    # data generator
    train_gen = CTCGenerator(trainX, trainY, 32, features_shape, char_nums)
    valid_gen = CTCGenerator(testX, testY, 32, features_shape, char_nums)

    # callbacks
    callbacks = [CTCMonitor(trainX, testX, trainY, testY, args['output'], char_nums, class_num,
                            base_model=predict_model, json_path=json_path, start_at=args['start_epoch']),
                 ]

    # compile model
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True, decay=0.01 / epoch)
    train_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

    H = train_model.fit_generator(train_gen.generator(),
                                  steps_per_epoch=train_gen.num_images // 32,
                                  validation_data=valid_gen.generator(),
                                  validation_steps=valid_gen.num_images // 32,
                                  epochs=epoch, verbose=1,
                                  initial_epoch=args['start_epoch'],
                                  callbacks=callbacks)