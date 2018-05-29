import pandas as pd
import numpy as np
import sklearn
from multiprocessing import Pool
from functools import reduce
import math

class EnsembleModel(object):
    def __init__(self, models, model_args, pool_num, model_name, weights = None,):
        self.models = models
        self.model_args = model_args['args']
        self.models_num = len(self.models)
        self.weights = model_args['weights'] \
            if model_args['weights'] else [float(1/self.models_num) for i in range(self.models_num)]
        self.pool_num = pool_num
        self.model_name = model_name

    # single process for every model fit
    def _fit(self, args):
        train_data, train_label, model, model_args, model_index= args[0], args[1], args[2], args[3], args[4]

        # Operate model need sync
        init_args = list(model_args)
        init_args.append(model_index)

        model = model(init_args)
        # rnn is a special class
        if model.is_rnn():
            return (model_index, model_index)

        print('model ', model.get_name(), ' ', model_index, 'start training...')
        model.fit(train_data, train_label)
        print('model ', model.get_name(), ' ' , model_index , 'training done.')

        if model.nn_model():
            return (model_index, model_index)
        else:
            return (model_index, model)

    # fit models for every model using MULTIPROCESS
    def fit(self, features, labels):
        # Operate Event array to sync
        with Pool(self.pool_num) as m_pool:
            args = [(features[i], labels[i], self.models[i], self.model_args[i], i) for i in range(self.models_num)]
            results = m_pool.map(self._fit, args)
            for i in range(self.models_num):
                if self.models[results[i][0]].nn_model():
                    init_args = list(self.model_args[results[i][0]])
                    init_args.append(results[i][0])
                    self.models[results[i][0]] = self.models[results[i][0]](init_args)

                    # if is rnn, fit in main process
                    if self.models[results[i][0]].is_rnn():
                        print('model ', self.models[results[i][0]].get_name(),
                              ' ', results[i][0], 'start training...')
                        self.models[results[i][0]].fit(features[results[i][0]], labels[results[i][0]])
                        print('model ', self.models[results[i][0]].get_name(),
                              ' ', results[i][0], 'training done.')
                else:
                    self.models[results[i][0]] = results[i][1]

    # predict from multi feature list and gather it into a single list, get class and prob
    def _predict(self, features):
        if len(features) > 500: # the features is all same with no groups
            result_collections = [self.models[i].predict_prob(features) for i in range(self.models_num)]
        else:
            result_collections = [self.models[i].predict_prob(features[i]) for i in range(self.models_num)]


        result = reduce(lambda x, y: np.array(x) + np.array(y), list(result_collections))
        # return result probabilities
        result_ave_prob = [ x / self.models_num for x in result]
        # return class types
        result_ave_class = [1 if (x / self.models_num) >= 0.5 else 0 for x in result]
        return result_ave_class, result_ave_prob

    # get the gathered predict prob and to csv files
    def predict(self, real_test_id, features):
        _, result = self._predict(features=features)
        _ = {'id': real_test_id, 'score': result}
        result = pd.DataFrame(_)
        result.to_csv(str(self.model_name) + '_ensemble_result.csv', index=False)

    def evaluate(self, test_data, test_label):
        result, result_prob= self._predict(test_data)
        conf_mat = sklearn.metrics.confusion_matrix(test_label, result, list(set(test_label)))
        print('confusion_matrix:')
        print(conf_mat)

        print('accuracy: ', sklearn.metrics.accuracy_score(y_true=test_label, y_pred=result))
        print('verify score: ', self.verify_score(result_prob, test_label))

    def verify_score(self, y, label):
        '''
        :param y: list 预测的值
        :param label: list 标签
        :return: 得分
        '''
        length = len(label)
        labels = [int(i) for i in label]
        positive = []
        negative = []
        for i in range(len(labels)):
            if labels[i] == 1:
                positive.append(y[i])
            elif labels[i] == 0:
                negative.append(y[i])
        negative.sort(reverse=True)
        fpr1 = negative[int(math.floor(0.001 * length))]
        fpr2 = negative[int(math.floor(0.005 * length))]
        fpr3 = negative[int(math.floor(0.01 * length))]
        tpr1 = 0
        tpr2 = 0
        tpr3 = 0
        for i in range(len(positive)):
            if positive[i] > fpr1:
                tpr1 += 1
            if positive[i] > fpr2:
                tpr2 += 1
            if positive[i] > fpr3:
                tpr3 += 1
        score = 0.4 * tpr1 / len(positive) + 0.3 * tpr2 / len(positive) + 0.3 * tpr3 / len(positive)
        return score