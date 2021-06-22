import torch
import torch.nn as nn

import os

import numpy as np

import nsml

import pandas as pd

from utils import read_csv_data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

ck_point_name = 'best'

class pjcclassifier:
    def __init__(self):
        super().__init__()
        self.enc = None
        self.scaler = None
        self.model = RandomForestClassifier(n_estimators=100, random_state=1, 
        max_features=None, n_jobs=12, min_samples_split=8, min_samples_leaf=8) # auto, log2

    def get_encoder(self, enc, scaler):
        self.enc = enc
        self.scaler = scaler

    def fit(self, train_x, train_y):
        rfc = self.model
        rfc = rfc.fit(train_x, train_y)
        print("model done")

        print("Train Score : ", rfc.score(train_x, train_y))

        # Confusion Matrix
        predict_y = rfc.predict(train_x)
        test_y_class = np.argmax(train_y, axis=1)
        predict_y_class = np.argmax(predict_y, axis=1)

        print(classification_report(test_y_class, predict_y_class))
        print(confusion_matrix(test_y_class, predict_y_class))

        # Feature importance
        feature_importance = rfc.feature_importances_
        print("feature importance", feature_importance)

        print('save model')
        nsml.save(checkpoint=ck_point_name)
    
    def evaluate(self, test_dir: str):
        assert self.enc is not None
        assert self.scaler is not None

        x_path = os.path.join(test_dir, 'test_data')

        csv_files = sorted([fname for fname in os.listdir(x_path) if fname.endswith('csv')])

        df_list = []
        df_pxy = []

        for fname in csv_files:
            df_i = pd.read_csv(os.path.join(x_path, fname))

            df_i['px'] = df_i['px'].replace({'-': '0'}).astype('int64')
            df_i['py'] = df_i['py'].replace({'-': '0'}).astype('int64')
            pxy = df_i[['px', 'py']]

            aux_col = df_i[['px', 'py']].values
            if aux_col.std(axis=0)[0] == 0:
                aux_col = aux_col / [1.0, 1.0]
            else:   
                aux_col = aux_col / aux_col.std(axis=0)
            
            # aux_mean = df_i[['px', 'py']].values
            # if aux_mean.mean(axis=0)[0] == 0:
            #     aux_mean = aux_mean / [1.0, 1.0]
            # else:
            #     aux_mean = aux_mean / aux_mean.mean(axis=0)

            pxy = np.concatenate((pxy, aux_col), axis=1)
            del df_i['px']
            del df_i['py']

            for c in df_i.columns:
                df_i[c] = df_i[c].replace({np.nan: '-'})

            df_list.append(df_i)
            df_pxy.append(pxy)
        
        print('number of csv file', len(csv_files))
        result_dict = dict()

        for fname, df, dfpxy in zip(csv_files, df_list, df_pxy):

            df = df.to_numpy()
            # dfpxy = dfpxy.to_numpy()

            print("filename", fname)
            print('test data shape', df.shape)
            print('test pxy shape', dfpxy.shape)

            df, dfpxy = make_sequence(df, dfpxy)
            print('after make sequence test pxy shape', dfpxy.shape)

            x_test_onehot = self.enc.transform(df).toarray()
            pxy_test_scaler = self.scaler.transform(dfpxy)
            print('test data shape after onehotencoding', x_test_onehot.shape)
            print('test data shape after scaler', pxy_test_scaler.shape)

            x_test_onehot = np.concatenate((x_test_onehot, pxy_test_scaler), axis=1)
            print('test data shape after onehotencoding', x_test_onehot.shape)

            predict_y = self.model.predict(x_test_onehot)

            # tmp = []
            # for k in range(len(predict_y)):
            #     if predict_y[k][1] > 0.1:
            #         tmp.append(1)
            #     else:
            #         tmp.append(0)

            # tmp = np.array(tmp)
            # tmp = tmp.astype(np.uint)
            # predict_y_class = tmp
            # print(predict_y_class.shape)

            predict_y_class = np.argmax(predict_y, axis=1).astype(np.uint)
            result_dict[fname] = pd.Series(predict_y_class.flatten())

        aggrdf = pd.DataFrame(result_dict)

        return aggrdf

def make_sequence(df, dfpxy):
    timesteps = 3
    features = len(dfpxy[0])
    samples = len(dfpxy) - (timesteps-1)

    result_pxy = []
    result_pxy.append(np.concatenate((dfpxy[0], dfpxy[0], dfpxy[0], dfpxy[0])))
    result_pxy.append(np.concatenate((dfpxy[1], dfpxy[0], dfpxy[0], dfpxy[0])))
    result_pxy.append(np.concatenate((dfpxy[2], dfpxy[1], dfpxy[0], dfpxy[0])))
    for i in range(samples-1):
        result_pxy.append(np.concatenate((dfpxy[i+3], dfpxy[i+2], dfpxy[i+1], dfpxy[i])))
    result_pxy = np.array(result_pxy)

    result_x = []
    result_x.append(np.array([df[0,4]]))
    for i in range(samples+1):
        result_x.append(np.array([df[i,4]]))
    result_x = np.array(result_x)
    result_x = result_x.reshape(-1, 1)

    # result_x1 = []
    # result_x1.append(np.array([df[0,3]]))
    # for i in range(samples+1):
    #     result_x1.append(np.array([df[i,3]]))
    # result_x1 = np.array(result_x1)
    # result_x1 = result_x1.reshape(-1, 1)

    result_x = np.concatenate((df, result_x), axis=1)

    return result_x, result_pxy