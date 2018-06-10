# -*- coding: utf-8 -*-
# @Time    : 2018/6/4  21:51
# @Author  : Dyn

if __name__ == '__main__':
    import numpy as np
    np.random.seed(2017)
    from tensorflow import set_random_seed
    set_random_seed(1994)
    import matplotlib
    matplotlib.use('Agg')


    def ploy_decay(epoch):
        max_epochs = EPOCH
        base_lr = INIT_LR
        power = 1.0
        return base_lr * (1 - (epoch / max_epochs)) ** power

    from config.config import *
    from network.cnn.head import FCHead
    from network.cnn.vgg import VGGNet
    from network.cnn.googlenet import GoogLeNet
    from network.cnn.resnet import ResNet
    from network.rnn.lstm import LSTMNet, AttentionLSTMNet
    from utils.read_features import read_data
    from utils.callback import MonitorCallback, EpochCheckpointSaver
    from keras.utils import plot_model
    from keras.optimizers import SGD
    from keras import backend as K
    from keras.models import load_model
    import numpy as np
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
    depth = DEPTH

    epoch = EPOCH
    net = NET
    split = SPLIT
    using_aug = USING_AUG

    # read data
    trainX, testX, trainY, testY = read_data(image_path=args["dataset"], split=split)

    # initialize the model
    if args['start_epoch'] == 0:
        if net == 'VGG':
            model = VGGNet.build(WIDTH, HEIGHT, depth, [2, 2, 2, 2, 2], [32, 64, 238, 256, 512], 2e-4)
            model = FCHead.build(model, char_nums, class_num, split)
        elif net == 'ResNet':
            model = ResNet.build(WIDTH, HEIGHT, depth, (9, 9, 9), (64, 64, 128, 256), 2e-4)
            model = FCHead.build(model, char_nums, class_num, split)
        elif net == 'GoogLeNet':
            model = GoogLeNet.build(WIDTH, HEIGHT, depth, 2e-4)
            model = FCHead.build(model, char_nums, class_num, split)
        elif net == 'LSTM':
            model = LSTMNet.build(WIDTH, HEIGHT, depth, char_nums, class_num, 2e-4)
            # model = LSTMCaptchaNet.transfer_build(char_nums, class_num, CNN_MODEL, LAY_NUM, 2e-4)
        elif net == 'AttentionLSTM':
            model = AttentionLSTMNet.build(WIDTH, HEIGHT, depth, char_nums, class_num, 2e-4)
        else:
            model = None

        opt = SGD(lr=INIT_LR, momentum=0.9, nesterov=True, decay=INIT_LR / epoch)
        # opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss="categorical_crossentropy", optimizer=opt)

        # plot the model
        plot_model(model, to_file=os.path.sep.join([args['output'], net + ".png"]), show_shapes=True)
    else:
        print('[INFO] start training at epoch {0}'.format(args['start_epoch']))
        model = load_model(os.path.sep.join([args['model'], args['checkpoint']]))
        # update the learning rate
        K.set_value(model.optimizer.lr, NEW_LR)
        # set json path of log

    json_path = os.path.sep.join([args['model'], '{}_log.json'.format(net)])

    # accuracy and loss log
    model_name = os.path.sep.join([args["model"], net + "{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.hdf5"])
    callbacks = [MonitorCallback(trainX, testX, trainY, testY, args['output'], char_nums, class_num, split,
                                 json_path=json_path, start_at=args['start_epoch']),
                 EpochCheckpointSaver(net, 5, args['model'])
                 # LearningRateScheduler(ploy_decay),
                 ]

    # train the network
    print("[INFO] training network...")
    H = model.fit(trainX, trainY,
                  validation_data=(testX, testY),
                  batch_size=BATCH_SIZE, epochs=epoch, verbose=1, callbacks=callbacks,
                  initial_epoch=args['start_epoch'])

    print(callbacks[0].H["train_accuracy"])
    print(callbacks[0].H["valid_accuracy"])
    print(np.max(callbacks[0].H["valid_accuracy"]))
    print(np.argmax(callbacks[0].H["valid_accuracy"]))
