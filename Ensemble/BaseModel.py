from abc import ABCMeta, abstractmethod
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.model = None
        # trans_model to fit train features and transform others
        self.scaler = None
        self.trans_model = None
        self.normalize = None
        self.fea_deal = None
        pass

    @staticmethod
    def is_rnn():
        return False

    def prepare_fea(self, features, labels=None):
        if not self.scaler:
            if self.normalize == 'standard':
                self.scaler = StandardScaler()
                self.scaler.fit(features)
            elif self.normalize == 'minmax':
                self.scaler = MinMaxScaler()
                self.scaler.fit(features)
            elif self.normalize == 'robust':
                self.scaler = RobustScaler()
                self.scaler.fit(features)

        _features = self.scaler.transform(features)

        if not self.fea_deal:
            return _features
        elif self.trans_model:
            return self.trans_model.transform(_features)
        elif self.fea_deal == 'pca':
            self.trans_model = PCA(copy=True, n_components=200, svd_solver='full')
            self.trans_model.fit(_features)
            return self.trans_model.transform(_features)
        elif self.fea_deal == 'select':
            clf = RandomForestClassifier()
            clf = clf.fit(_features, labels)
            self.trans_model = SelectFromModel(clf, prefit=True)
            return self.trans_model.transform(_features)

    @abstractmethod
    def fit(self, feature, label):pass

    @abstractmethod
    def predict_prob(self, features):pass