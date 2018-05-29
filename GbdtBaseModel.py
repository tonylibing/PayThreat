from BaseModel import BaseModel
from sklearn.ensemble import GradientBoostingClassifier

class GbdtModel(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.normalize, self.fea_deal, self.model_index = args[0], args[1], args[2]
        self._model = GradientBoostingClassifier()
        self.fit_done = 0

    @staticmethod
    def get_name():
        return 'gbdt'

    @classmethod
    def nn_model(clf):
        return False

    def fit(self, features, labels):
        _features = self.prepare_fea(features, labels)
        self._model.fit(_features, labels)
        self.fit_done = 1

    def predict_prob(self, features):
        _features = self.prepare_fea(features)
        return self._model.predict_proba(_features)[:,1]