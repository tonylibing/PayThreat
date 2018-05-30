from functools import partial

from EnsembleData import prepare_data as prepare_data
from EnsembleModel import EnsembleModel
from GbdtBaseModel import GbdtModel
from DenseBaseModel import DenseModel
from LstmBaseModel import LSTMModel
from SVMBaseModel import SVMModel

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

# models = [SVMModel]
models = []
arg = []

# GbdtModel args : normalize methods, feature methods, (model_index, appended in the EnsembleModel)
gbdt_args = ['standard', 'select']
gbdt_test_args = ['standard', 'select']

svm_args = ['standard', 'select']
svm_test_args = ['standard', 'select']

# DenseModel args: normalize methods, feature methods, layers, layer units, batch_size, epochs, steps
dense_args = ['standard', None, 2, (512, 256), 500, 100, 20000]
dense_test_args = ['standard', None, 2, (512, 256), 500, 1, 20]

# LstmModel args: normalize methods, features methods, time_steps, n_inputs, n_hiddens,keep_prob, n_layers, batch_size, epochs, steps
lstm_args = ['standard', None, 12, 25, 128, 0.8, 2, 500, 100, 20000]
lstm_test_args = ['standard', None, 12, 25, 128, 0.8, 2, 500, 1, 20]

for i in range(4):
    if i < 1:
       models.append(LSTMModel)
       arg.append(lstm_args)
    if i < 2:
        models.append(DenseModel)
        arg.append(dense_args)

    elif i < 3:
        models.append(SVMModel)
        arg.append(svm_args)

    else:
        models.append(GbdtModel)
        arg.append(gbdt_args)

model_args = {
    'args': arg,
    'weights': None
}

model_test_args = {
'args':[
        # gbdt_test_args , # gbdt args
        # dense_test_args,
        # lstm_test_args,
        # svm_test_args
        fc_test_args,
    ],
    'weights': None
}
EnsembleTest = partial(EnsembleMain, samples=1000, processes = 2, models = models, model_args = model_test_args)
Ensemble = partial(EnsembleMain, models=models, model_args=model_args)
EnsembleTest()
