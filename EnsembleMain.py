from functools import partial

from EnsembleData import prepare_data as prepare_data
from EnsembleModel import EnsembleModel
from GbdtBaseModel import GbdtModel
from DenseBaseModel import DenseModel
from LstmBaseModel import LSTMModel

def EnsembleMain(models, model_args, samples=994731, groups=None, processes=24,  na_value='mean',  hot=False):

    print('get data...')
    group_len, train_features, train_label, test_features, test_label, real_id, real_features = prepare_data(samples=samples, groups=groups, processes=processes, na_value=na_value, hot=hot)

    paythreat = EnsembleModel(models=models, model_args=model_args, pool_num=processes, model_name='test')
    print('train start ...')
    paythreat.fit(train_features, train_label)
    print('train done.\nevaluate start ...')
    paythreat.evaluate(test_features, test_label)
    print('evaluate done.\npredict start...')
    paythreat.predict(real_id, real_features)
    print('predict done.')

models = []

# GbdtModel args : normalize methods, feature methods, (model_index, appended in the EnsembleModel)
gbdt_args = ['standard', 'select']
gbdt_test_args = ['standard', 'select']

# DenseModel args: normalize methods, feature methods, layers, layer units, batch_size, epochs, steps
dense_args = ['standard', None, 2, (512, 256), 500, 100, 20000]
dense_test_args = ['standard', None, 2, (512, 256), 500, 1, 20]


# LstmModel args: normalize methods, features methods, time_steps, n_inputs, n_hiddens,keep_prob, n_layers, batch_size, epochs, steps
lstm_args = ['standard', None, 12, 25, 128, 0.8, 2, 500, 100, 20000]
lstm_test_args = ('standard', None, 12, 25, 128, 0.8, 2, 500, 1, 20)
model_args = {
    'args':[
        gbdt_args , # gbdt args
        dense_args,
        lstm_args
    ],
    'weights': None
}

model_test_args = {
    'args':[
        lstm_test_args
    ],
    'weights': None
}

for i in range(2):
    models.append(LSTMModel)
    model_test_args['args'].append(lstm_test_args)

EnsembleTest = partial(EnsembleMain, samples=4000, groups=len(models), processes = 2, models = models, model_args = model_test_args)
Ensemble = partial(EnsembleMain, models=models, model_args=model_args)

if __name__ == '__main__':
    EnsembleTest()