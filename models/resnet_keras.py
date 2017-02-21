#Forked from kernel by n01z3: https://www.kaggle.com/drn01z3/data-science-bowl-2017/resnet50-features-xgboost/run/699911

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import gmean
from sklearn.decomposition import PCA

import keras as k
import keras.layers as l

def get_model(size):
    m = k.models.Sequential()

    m.add(l.Dense(96, input_dim=size))
    m.add(l.Activation('relu'))
    m.add(l.Dense(16))
    m.add(l.Activation('relu'))
    m.add(l.Dense(1))
    m.add(l.Activation('sigmoid'))

    m.compile(loss='binary_crossentropy', optimizer='adam')
    return m



def keras():
    df = pd.read_csv('../input/stage1_labels.csv')
    df_test = pd.read_csv('../input/stage1_sample_submission.csv')

    print("Loading data.")

    x = np.array([np.mean(np.load('../input/stage1/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    x_test = np.array([np.mean(np.load('../input/stage1/%s.npy' % str(id)), axis=0) for id in df_test['id'].tolist()])
    y = df['cancer'].as_matrix()

    print("Start PCA.")
    pca = PCA(n_components=64)
    pca.fit(x)
    x = pca.transform(x)
    x_test = pca.transform(x_test)

    x = x / 4
    x_test = x_test / 4

    skf = StratifiedKFold(n_splits=5, random_state=88, shuffle=True)

    preds = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]

        m = get_model(trn_x.shape[1])
        m.fit(trn_x, trn_y, batch_size=256, nb_epoch=150, validation_data=(val_x, val_y), verbose=2)

        pred = [p[0] for p in m.predict(x_test)]
        preds.append(pred)

    preds = np.array(preds)
    print(preds)
    print(preds.shape)

    preds = preds.mean(axis=0)
    df_test['cancer'] = preds
    df_test.to_csv('../submission/subm_krs.csv', index=False)

if __name__ == '__main__':
    keras()
