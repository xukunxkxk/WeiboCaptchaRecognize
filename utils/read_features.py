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
import json
import os
import glob
import torch
from copy import deepcopy
from tqdm import trange

def load_ckpt(llama_path):
    ckpt_list = sorted(glob.glob(os.path.join(llama_path, "*.pth")))
    model_list = [torch.load(ckpt, map_location="cpu") for ckpt in ckpt_list]
    params = json.load(open(os.path.join(llama_path, "params.json")))
    return model_list, params
    

def merge_and_split(llam_path, tp, pp=1):
    print("Load Model ckpt ing")
    model_list, config = load_ckpt(llam_path)
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    n_hidden = config['dim']
    num_shards = len(model_list)
    n_heads_per_shard = n_heads // num_shards
    dims_per_head = n_hidden // n_heads
    print("We have {} ckpts")

    def get_params(model_key):
        return [model[model_key] for model in model_list]
    
    def merge_params(model_key, dim):
        return torch.cat(get_params(model_key), dim=dim)
    
    def create_pp_dict(pp_idx):
        model_dict = {}
        model_dict['model'] = {}
        model_dict['model']['language_model'] = {}
        if pp_idx == 0:
            model_dict['model']['language_model']['embedding'] = {}
            model_dict['model']['language_model']['embedding']['word_embeddings'] = {}
        if pp_idx == pp - 1:
            model_dict['model']['language_model']['output_layer'] = {}
        model_dict['model']['language_model']['encoder'] = {}
        model_dict['checkpoint_version'] = 3.0
        return model_dict
    
    def permute(w):
        return w.view(n_heads, n_hidden // n_heads // 2, 2, n_hidden).transpose(1, 2).reshape(n_hidden, n_hidden)

    # mp_rank_0tp_idx_00pp_idx
    model_dict = [create_pp_dict(pp_idx) for pp_idx in range(pp)]
    model_dict = [deepcopy(model_dict) for tp_idx in range(tp)]

    # word embeddings
    word_embeddings = merge_params('tok_embeddings.weight', 1)
    for tp_idx, each_word_embeddings in enumerate(torch.chunk(word_embeddings, tp, dim=0)):
        model_dict[tp_idx][0]['model']['language_model']['embedding']['word_embeddings']['weight'] = each_word_embeddings
    
    # output layer
    output_layer = merge_params('output.weight', 0)
    for tp_idx, each_output_layer in enumerate(torch.chunk(output_layer, tp, dim=0)):
        model_dict[tp_idx][-1]['model']['language_model']['output_layer']['weight'] = each_output_layer

    # final layer norm
    megatron_key = "final_layernorm.weight"
    fair_key = "norm.weight"
    for tp_idx in range(tp):
        model_dict[tp_idx][-1]['model']['language_model']['encoder'][megatron_key] = model_list[0][fair_key].float().clone()
    print("Merge Model ckpt ing")
    numer_of_transformer_each_pp = n_layers // pp
    for tp_idx in trange(tp):
        for pp_idx in trange(pp):
            prefix_model_dict = model_dict[tp_idx][pp_idx]['model']['language_model']['encoder']
            start_idx = pp_idx * numer_of_transformer_each_pp
            end_idx = (pp_idx + 1) * numer_of_transformer_each_pp
            for layer_idx in range(start_idx, end_idx):
                pp_layer_idx = layer_idx % numer_of_transformer_each_pp

                # input_layernorm
                megatron_key = f"layers.{pp_layer_idx}.input_layernorm.weight"
                fair_key = f"layers.{layer_idx}.attention_norm.weight"
                prefix_model_dict[megatron_key] = model_list[0][fair_key].float().clone()
                # post attention layernorm
                megatron_key = f"layers.{pp_layer_idx}.post_attention_layernorm.weight"
                fair_key = f"layers.{layer_idx}.ffn_norm.weight"
                prefix_model_dict[megatron_key] = model_list[0][fair_key].float().clone()

                # q k v
                megatron_key = f"layers.{pp_layer_idx}.self_attention.query_key_value.weight"
                query = None

                query = permute(torch.cat(
                    [model_list[i][f"layers.{layer_idx}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, n_hidden)
                     for i in range(num_shards)
                        ],dim=0,
                ).reshape(n_hidden, n_hidden))

                key = permute(torch.cat(
                    [model_list[i][f"layers.{layer_idx}.attention.wk.weight"].view(n_heads_per_shard, dims_per_head, n_hidden)
                     for i in range(num_shards)
                        ],dim=0,
                ).reshape(n_hidden, n_hidden))

                value = torch.cat(
                    [model_list[i][f"layers.{layer_idx}.attention.wv.weight"].view(n_heads_per_shard, dims_per_head, n_hidden)
                     for i in range(num_shards)
                        ],dim=0,
                ).reshape(n_hidden, n_hidden)
                query_key_value = torch.cat([query, key, value], dim=0)
                input_shape = query_key_value.size()
                view_shape = (3, n_heads, n_hidden // n_heads) + input_shape[1:]
                query_key_value = query_key_value.view(*view_shape).transpose(0, 1).contiguous().view(*input_shape)
                prefix_model_dict[megatron_key] = torch.chunk(query_key_value, tp, 0)[tp_idx]

                # dense
                megatron_key = f"layers.{pp_layer_idx}.self_attention.dense.weight"
                fair_key = f"layers.{layer_idx}.attention.wo.weight"
                prefix_model_dict[megatron_key] = torch.chunk(merge_params(fair_key, 1), tp, 1)[tp_idx]
                # dense_4h_to_h
                megatron_key = f"layers.{pp_layer_idx}.mlp.dense_4h_to_h.weight"
                fair_key = f"layers.{layer_idx}.feed_forward.w2.weight"
                prefix_model_dict[megatron_key] = torch.chunk(merge_params(fair_key, 1), tp, 1)[tp_idx]
                # dense_h_to_4h
                megatron_key = f"layers.{pp_layer_idx}.mlp.dense_h_to_4h.weight"
                w1 = torch.chunk(merge_params(f"layers.{layer_idx}.feed_forward.w1.weight", 0), tp, 0)[tp_idx]
                w3 = torch.chunk(merge_params(f"layers.{layer_idx}.feed_forward.w3.weight", 0), tp, 0)[tp_idx]
                prefix_model_dict[megatron_key] = torch.cat([w1, w3], dim=0)
    return model_dict


def save_llama_dict(model_dict, output_path, tp, pp=1):
    base_dir = os.path.join(output_path, "release")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    print("Saving Model ckpt ing")
    for tp_idx in trange(tp):
        for pp_idx in trange(pp):
            sub_dir = os.path.join(base_dir, f"mp_rank_0{tp_idx}_00{pp_idx}")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            model_dir = os.path.join(sub_dir, f"model_optim_rng.pt")
            torch.save(model_dict[tp_idx][pp_idx], model_dir)


llam_path = r"D:\llama"
output_path = llam_path
tp = 2
pp = 4
model_dict = merge_and_split(llam_path, tp, pp)
save_llama_dict(model_dict, output_path, tp, pp)

if __name__ == '__main__':
    pass
