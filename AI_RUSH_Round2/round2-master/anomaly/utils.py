import pandas as pd

import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def read_csv_data(train_data, train_label):
    """
    read csv and then preprocess
    """

    csv_file = os.listdir(train_data)
    data = []

    for i in range(len(csv_file)):
        data.append(os.path.join(train_data, csv_file[i]))

    label = pd.read_csv(train_label) # dataframe

    for i in range(len(data)):
        df = pd.read_csv(data[i], encoding='utf8')

        df['px'] = df['px'].replace({'-': '0'}).astype('int64')
        df['py'] = df['py'].replace({'-': '0'}).astype('int64')
        pxy = df[['px', 'py']].values
        
        aux_col = df[['px', 'py']].values
        if aux_col.std(axis=0)[0] == 0:
            aux_col = aux_col / [1.0, 1.0]
        else:    
            aux_col = aux_col / aux_col.std(axis=0)

        # aux_mean = df[['px', 'py']].values
        # if aux_mean.(axis=0)[0] == 0:
        #     aux_mean = aux_mean / [1.0, 1.0]
        # else:
        #     aux_mean = aux_mean / aux_mean.mean(axis=0)

        pxy = np.concatenate((pxy, aux_col), axis=1)
        del df['px']
        del df['py']

        for c in df.columns:
            df[c] = df[c].replace({np.nan: '-'})
        
        tmp = label[[csv_file[i]]]
        tmp = tmp.to_numpy()
        tmp = tmp[:len(df)]
        df = df.to_numpy()

        if i == 0:
            x_result = df
            y_result = tmp
            pxy_result = pxy
            continue
        
        x_result = np.concatenate((x_result, df))
        y_result = np.concatenate((y_result, tmp))
        pxy_result = np.concatenate((pxy_result, pxy))

        if i % 100 == 0:
            # break
            print(i, "done...")

    return x_result, y_result, pxy_result


def get_onehotencoder(train_x, train_y, train_pxy):
    enc_x = OneHotEncoder(handle_unknown='ignore')
    enc_x.fit(train_x)
    train_x_onehot = enc_x.transform(train_x).toarray()

    enc_y = OneHotEncoder(handle_unknown='ignore')
    enc_y.fit(train_y)
    train_y_onehot = enc_y.transform(train_y).toarray()

    scaler = StandardScaler()
    scaler.fit(train_pxy)
    train_pxy_scaler = scaler.transform(train_pxy)

    return train_x_onehot, train_y_onehot, train_pxy_scaler, enc_x, enc_y, scaler
