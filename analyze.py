import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


#csv.pop('id')
#csv = csv.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
def get_describe():
    csv = pd.read_csv('atec_anti_fraud_train.csv', nrows=10)
    with open('columns analyze', 'wt') as f:
        f.write('columns analysis:\n')
        for col in csv:
            content = col + '\n'
            content += str(csv[col].describe())
            content += '\n'
            f.write(content)
    print('get_describe done')

#print(feature)
def feature_selection():
    csv = pd.read_csv('atec_anti_fraud_train.csv', nrows=10)
    csv = csv.fillna(csv.mean())
    raw_data = csv[csv['label'] != -1]
    label_tmp = raw_data.pop('label')
    feature = raw_data.values
    label = label_tmp.values
    # feature selection
    clf = RandomForestClassifier()
    clf = clf.fit(feature, label)
    with open('feature_selection','wt') as f:
        f.write(clf.feature_importances_)
        f.write('\n')
        f.write(raw_data.columns)
    print('feature_selection done')

def base_analyze():
    csv = pd.read_csv('atec_anti_fraud_train.csv', nrows=10)
    print(csv['label'].unique())
    with open('base_analyze', 'wt') as f:
        f.write('no label -1: '+ str(len(csv[csv['label'] == -1])))
        f.write('\nwhite label 0: ' + str(len(csv[csv['label'] == 0])))
        f.write('\nblack label 1: ' + str(len(csv[csv['label'] == 1])))
    print('base_analyze done')

    
    
    
    
    

