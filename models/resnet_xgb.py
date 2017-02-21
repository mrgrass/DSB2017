#Forked from kernel by n01z3: https://www.kaggle.com/drn01z3/data-science-bowl-2017/resnet50-features-xgboost/run/699911

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from scipy.stats import gmean
from sklearn.decomposition import PCA


def xgboost():
    df = pd.read_csv('../input/stage1_labels.csv')
    df_test = pd.read_csv('../input/stage1_sample_submission.csv')

    print("Loading data.")
    x = np.array([np.mean(np.load('../input/stage1/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    x_test = np.array([np.mean(np.load('../input/stage1/%s.npy' % str(id)), axis=0) for id in df_test['id'].tolist()])
    y = df['cancer'].as_matrix()

    print("Start PCA.")
    pca = PCA(n_components=128)
    pca.fit(x)
    x = pca.transform(x)
    x_test = pca.transform(x_test)

    skf = StratifiedKFold(n_splits=5, random_state=88, shuffle=True)

    preds = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]

        clf = xgb.XGBRegressor(max_depth=6,
                               n_estimators=3000,
                               min_child_weight=9,
                               learning_rate=0.001,
                               nthread=8,
                               subsample=0.90,
                               colsample_bytree=0.95,
                               seed=8888)

        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=40)
        preds.append(np.clip(clf.predict(x_test),0.001,1))

    pred = gmean(np.array(preds), axis=0)
    df_test['cancer'] = pred
    df_test.to_csv('../submission/subm_xgb.csv', index=False)

if __name__ == '__main__':
    xgboost()
