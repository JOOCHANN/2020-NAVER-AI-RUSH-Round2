import argparse

import torch
import numpy as np
import pandas as pd

from utils import read_csv_data, get_onehotencoder
from models import pjcclassifier

import nsml
from nsml.constants import DATASET_PATH

import pickle
import os

def bind_model(model):

    def load(dirname, **kwargs):
        print('Trying to load')
        old_model = pickle.load(open(os.path.join(dirname, 'model.sav'), 'rb'))
        model.get_encoder(old_model.enc, old_model.scaler)
        model.model = old_model.model

    def save(dirname, **kwargs):
        filename = f'{dirname}/model.sav'
        print(f'Trying to save to {filename}')
        pickle.dump(model, open(filename, 'wb'))

    def infer(test_dir, **kwargs):
        print('start inference')
        return model.evaluate(test_dir)

    nsml.bind(load=load, save=save, infer=infer)

def cut_np(train_x, train_y, times=5):
    tmp = np.concatenate((train_x, train_y), axis=1)
    print(tmp.shape)
    result_x = []
    result_y = []

    for i in range(len(tmp)):
        if (tmp[i][-1] == 1) or (i % times == 0):
            result_x.append(tmp[i][:-1])
            result_y.append(tmp[i][-1])
        else :
            continue

    result_x = np.array(result_x)
    result_y = np.array(result_y)

    print(result_x.shape)
    print(result_y.shape)

    return result_x, result_y.reshape(-1, 1)

def make_sequence(train_x, train_pxy):
    timesteps = 3
    features = len(train_pxy[0])
    samples = len(train_pxy) - (timesteps-1)

    result_pxy = []
    result_pxy.append(np.concatenate((train_pxy[0], train_pxy[0], train_pxy[0], train_pxy[0])))
    result_pxy.append(np.concatenate((train_pxy[1], train_pxy[0], train_pxy[0], train_pxy[0])))
    result_pxy.append(np.concatenate((train_pxy[2], train_pxy[1], train_pxy[0], train_pxy[0])))
    for i in range(samples-1):
        result_pxy.append(np.concatenate((train_pxy[i+3], train_pxy[i+2], train_pxy[i+1], train_pxy[i])))
    result_pxy = np.array(result_pxy)

    result_x = []
    result_x.append(np.array([train_x[0,4]]))
    for i in range(samples+1):
        result_x.append(np.array([train_x[i,4]]))
    result_x = np.array(result_x)
    result_x = result_x.reshape(-1, 1)

    # result_x1 = []
    # result_x1.append(np.array([train_x[0,3]]))
    # for i in range(samples+1):
    #     result_x1.append(np.array([train_x[i,3]]))
    # result_x1 = np.array(result_x1)
    # result_x1 = result_x1.reshape(-1, 1)

    result_x = np.concatenate((train_x, result_x), axis=1)

    return result_x, result_pxy

def main():
    parser = argparse.ArgumentParser()
    # NSML args
    parser.add_argument('--is-on-automl', action='store_true')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)

    args = parser.parse_args()

    pjc = pjcclassifier()

    bind_model(pjc)

    if args.pause:
        nsml.paused(scope=locals())

    if args.mode == 'train':
        
        train_x, train_y, train_pxy = read_csv_data(os.path.join(DATASET_PATH, 'train', 'train_data'), os.path.join(DATASET_PATH, 'train', 'train_label'))
        print("before onehotencoder")
        print(train_x.shape)
        print(train_y.shape)
        print(train_pxy.shape)

        train_x, train_pxy = make_sequence(train_x, train_pxy)
        print("after sequence")
        #print(train_x[0:10000])
        print(train_x.shape)
        print(train_pxy.shape)

        # count = 0
        # for i in range(len(train_y)):
        #     if train_y[i] == 1:
        #         print(i, train_x[i], train_pxy[i], train_y[i])
        #         count = count + 1
        # print(count)

        train_x, train_y, train_pxy, enc_x, enc_y, scaler = get_onehotencoder(train_x, train_y, train_pxy)
        print("after onehotencoder")
        print(train_x.shape)
        print(train_y.shape)
        print(train_pxy.shape)

        train_x = np.concatenate((train_x, train_pxy), axis=1)
        print("after concat")
        print(train_x.shape)

        # train_x, train_y = make_sequence(train_x, train_y)

        print("run model")

        pjc.get_encoder(enc_x, scaler)
        pjc.fit(train_x, train_y)

        print('training done')


if __name__ == '__main__':
    main()
