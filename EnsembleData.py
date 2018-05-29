from functools import reduce

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import gc
from multiprocessing import Pool

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def prepare_data(samples, groups, processes,  na_value, hot, norm='standard', feature=None):
    # get raw data: train shape is (-1, groups, 300), test shape is (-1, 300), real shape is (-1, 299)
    _, train, test, real = read_sample(samples=samples, groups=groups)
    _groups = _

    # seprate features and labels
    train_fea, train_lab, test_fea, test_lab, real_id, real_fea = sep_fea_lab(train, test, real)

    # fill na values
    # value is None using standard trans
    with Pool(processes) as m_pool:
        args = [(train_fea[i], i, na_value) for i in range(_groups)]
        _ = m_pool.map(fill_na, args)
        for i in range(_groups):
            train_fea[_[i][0]] = _[i][1]
    if na_value == 'mean':
        _na_value = [test_fea.mean(), real_fea.mean()]
    else:
        _na_value = [na_value, na_value]
    test_fea = test_fea.fillna(_na_value[0])
    real_fea = real_fea.fillna(_na_value[1])

    # normalization and feature deal in every model

    # normalization
    # train_fea, test_fea, real_fea = normalization(train_fea, test_fea, real_fea, norm, _groups)
    #
    # if not feature:
    #     if hot:
    #         return _groups, train_fea, one_hot_encoding(train_lab), test_fea, one_hot_encoding(test_lab), real_id, real_fea
    #     else:
    #         return _groups, train_fea, train_lab, test_fea, test_lab, real_id, real_fea
    # elif feature == 'pca':
    #     with Pool(2) as m_pool:
    #         args = [(train_fea[i], test_fea[i], real_fea[i], i) for i in range(_groups)]
    #         results = m_pool.map(pca_trans, args)
    #         for i in range(len(results)):
    #             train_fea[results[i][3]] = results[i][0]
    #             test_fea[results[i][3]] = results[i][1]
    #             real_fea[results[i][3]] = results[i][2]
    # elif feature == 'selecte':
    #     with Pool(2) as m_pool:
    #         args = [(train_fea[i], train_lab[i], test_fea[i], real_fea[i], i) for i in range(_groups)]
    #         results = m_pool.map(sel_trans, args)
    #         for i in range(len(results)):
    #             train_fea[results[i][3]] = results[i][0]
    #             test_fea[results[i][3]] = results[i][1]
    #             real_fea[results[i][3]] = results[i][2]
    if hot:
        train_lab, test_lab = one_hot_encoding(train_lab), one_hot_encoding(test_lab)

    return _groups, train_fea, train_lab, test_fea, test_lab, real_id, real_fea

def read_sample(samples, groups):
    # no label -1: 4725
    # white label 0: 977884
    # black label 1: 12122

    # read, sample, fillna

    # sample to make the white and black data balance
    # the sample number is the white data number

    # read train data and convert the label to str format to avoid -1 == 0 in logic
    train = pd.read_csv('atec_anti_fraud_train.csv', nrows=samples)
    train['label'] = train['label'].apply(lambda x: str(x))
    real = pd.read_csv('atec_anti_fraud_test_a.csv')

    # sample test data
    test = train[train['label'] != '-1'].sample(frac=0.3)

    #sample train data
    white_csv = train[train['label'] == '0']
    black_csv = train[train['label'] == '1']

    black_len = len(black_csv)
    white_len = len(white_csv)

    ratio = groups if groups else int(white_len / black_len)

    # seprate train to groups
    train = map(lambda x: pd.concat([
        white_csv.sample(n=black_len, replace=True, random_state=x), black_csv
    ]).sample(frac=1).reset_index(drop=True), range(int(ratio)))
    train = list(train)



    return ratio, train, test, real

def sep_fea_lab(train, test, real):
    train_label = []
    for i in range(len(train)):
        train[i].pop('id')
        train_label.append(train[i].pop('label').astype('int32'))

    test.pop('id')
    test_label = test.pop('label').astype('int32')

    real_id = real.pop('id')
    return train, np.array(train_label), test, np.array(test_label), np.array(real_id), real

def fill_na(args):
    features, index, value= args[0], args[1], args[2]
    if value == 'mean':
        features = features.fillna(features.mean())
    else:
        features = features.fillna(value)
    return (index, features)

def normalization(train_fea, test_fea, real_fea, norm, _groups):
    print('normalization...')
    _train_fea = [i for i in range(len(train_fea))]
    _test_fea = [i for i in range(len(train_fea))]
    _real_fea = [i for i in range(len(train_fea))]
    if norm == 'standard':
        _scaler = StandardScaler()
    elif norm == 'min_max':
        _scaler = MinMaxScaler()
    elif norm == 'robust':
        _scaler = RobustScaler()
    for i in range(len(train_fea)):
        scaler = _scaler
        scaler.fit(train_fea[i])
        _train_fea[i] = scaler.transform(train_fea[i])
        _test_fea[i] = scaler.transform(test_fea)
        _real_fea[i] = scaler.transform(real_fea)
    del(train_fea, test_fea, real_fea)
    gc.collect()
    print('normalization done.')

    _train_fea = [np.array(x) for x in _train_fea]
    _test_fea = [np.array(x) for x in _test_fea]
    _real_fea = [np.array(x) for x in _real_fea]

    return _train_fea, _test_fea, _real_fea

def pca_trans(args):
    train_data, test_data, real_data, index = args[0], args[1], args[2], args[3]
    pca = PCA(copy=True, n_components=200, svd_solver='full')
    pca.fit(train_data)
    print(index, 'pca done.')
    train_features = pca.transform(train_data)
    test_features = pca.transform(test_data)
    real_test_features = pca.transform(real_data)

    return (train_features, test_features, real_test_features, index)

def sel_trans(args):
    train_fea, train_lab, test_fea, real_fea, index = args[0], args[1], args[2], args[3], args[4]
    clf = RandomForestClassifier()
    clf = clf.fit(train_fea, train_lab)
    trans_model = SelectFromModel(clf, prefit=True)
    print(index, ' feature selection done.')
    return  (trans_model.transform(train_fea), trans_model.transform(test_fea), trans_model.transform(real_fea), index)

def one_hot_encoding(values):
    # for np one hot encoding
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]




