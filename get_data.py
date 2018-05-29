import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def data_sample(data, sample_number=977884):
    # sample data
    # black data need replace
    # white data donnot need this
    white_csv = data[data['label'] == '0'].sample(n=sample_number, replace=True)
    black_csv = data[data['label'] == '1'].sample(n=sample_number, replace=True)

    _ = [white_csv, black_csv]

    # white_csv和 black csv 仍需打乱顺序
    return pd.concat(_).sample(frac=1).reset_index(drop=True)

def one_hot_encoding(values):
    # for np one hot encoding
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]

def read_data(sample_number = 977883):
    # no label -1: 4725
    # white label 0: 977884
    # black label 1: 12122

    # read, sample, fillna

    # sample to make the white and black data balance
    # the sample number is the white data number

    # read train data and convert the label to str format to avoid -1 == 0 in logic
    train_data = pd.read_csv('atec_anti_fraud_train.csv', nrows = sample_number)
    train_data['label'] = train_data['label'].apply(lambda x: str(x))
    real_test_data = pd.read_csv('atec_anti_fraud_test_a.csv')

    print('fillna')
    # fill na with train_data.mean()
    _ = train_data.pop('label')
    train_data = train_data.fillna(0)
    real_test_data = real_test_data.fillna(0)
    train_data['label'] = _

    print('sample')
    # sample train data
    # sample test data from raw train data for local testing
    train_data = data_sample(train_data, sample_number = sample_number)
    test_data = train_data[train_data['label'] != '-1'].sample(frac=0.3)

    return train_data, test_data, real_test_data

def nn_input(sample_number = 977883, standard_trans = True, one_hot = False):
    #  one hot encoding
    train_data, test_data, real_test_data = read_data(sample_number = sample_number)
    train_data.pop('id')
    train_label = train_data.pop('label').astype('int32')

    test_data.pop('id')
    test_label = test_data.pop('label').astype('int32')

    # test data deal
    real_id = real_test_data.pop('id')

    print('standard_trans')
    if standard_trans:
        average = train_data.mean()
        variance = train_data.var()
        train_data = (train_data - average) / variance
        test_data = (test_data - average) / variance
        real_test_data = (real_test_data - average) / variance

    print('to np array')
    train_features = np.array(train_data)
    test_features = np.array(test_data)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    real_id = np.array(real_id)
    real_features = np.array(real_test_data)

    if one_hot:
        train_label = one_hot_encoding(train_label)
        test_label = one_hot_encoding(test_label)

    return train_features, train_label, test_features,  test_label, real_id, real_features

def traditional_input(sample_number = 977883, select_fea = False, pca_fea = False, standard_trans = True):
    train_data, test_data, real_test_data = read_data(sample_number = sample_number)
    train_id = train_data.pop('id')
    train_label = train_data.pop('label')

    test_id = test_data.pop('id')
    test_label = test_data.pop('label')

    real_test_id = real_test_data.pop('id')
    if standard_trans:
        average = train_data.mean()
        variance = train_data.var()
        train_data = (train_data - average) / variance
        test_data = (test_data - average) / variance
        real_test_data = (real_test_data - average) / variance

    train_features = np.array(train_data)
    train_label = np.array(train_label)
    test_features = np.array(test_data)
    test_label = np.array(test_label)
    real_test_id = np.array(real_test_id)
    real_test_features = np.array(real_test_data)

    if select_fea:
        print('start feature selection...')
        clf = RandomForestClassifier()
        clf = clf.fit(train_features, train_label)
        model = SelectFromModel(clf, prefit=True)
        print('feature selection done.')
        train_features = model.transform(train_features)
        test_features = model.transform(test_features)
        real_test_features = model.transform(real_test_features)

    if pca_fea:
        print('start pca ....')
        pca = PCA(copy=True, n_components=200, svd_solver='full')
        pca.fit(train_features)
        print('pca done.')
        train_features = pca.transform(train_features)
        test_features = pca.transform(test_features)
        real_test_features = pca.transform(real_test_features)

    return train_features, train_label, test_features, test_label, real_test_id, real_test_features







