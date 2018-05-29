from BaseModel import BaseModel
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

class LSTMModel(BaseModel):
    def __init__(self, args):
        super().__init__()
        # feature deal, model init and fit args
        self.normalize, self.fea_deal, \
        self.time_steps, self.n_inputs, self.n_hiddens, self.keep_prob, self.n_layers, \
        self.batch_size, self.epochs, self.steps, \
        self.model_index = \
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]
        # if the model has fitted
        self.fit_done = 0
        # nn model has model_dir
        self.model_dir = './model/lstm_model_' + str(self.model_index) + '/'
        # define model
        self.model = tf.estimator.Estimator(
            model_fn=self.lstm_fn, model_dir=self.model_dir, params=(self.time_steps, self.n_inputs, self.n_hiddens, self.keep_prob, self.n_layers))

        # tensorflow init
        tf.logging.set_verbosity(tf.logging.ERROR)
        # Set up logging for predictions
        tensors_to_log = {"probabilities": "softmax_tensor"}
        self.logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

    @staticmethod
    def get_name():
        return 'lstm'

    @staticmethod
    def nn_model():
        return True

    @staticmethod
    def is_rnn():
        return True

    def change2d(self, input, pad_number):
        pad_vec = np.zeros(shape=(input.shape[0], pad_number))
        result = np.column_stack((input, pad_vec))
        return result

    def lstm_fn(self, features, labels, mode, params):
        time_steps, n_input, n_hiddens, keep_prob, n_layers = \
                                    params[0], params[1], params[2], params[3], params[4]
        input_layer = tf.cast(features['x'], tf.float32)

        input_layer = tf.reshape(input_layer, [-1, time_steps, n_input])

        def attn_cell():
            with tf.name_scope('lstm_layers'):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hiddens)
            return lstm_cell

        basic_cells = [attn_cell() for i in range(n_layers)]
        with tf.name_scope('lstm_cells_layers'):
            # stack the lstm nn
            stacked_cell = tf.contrib.rnn.MultiRNNCell(basic_cells, state_is_tuple=True)

        _init_state = stacked_cell.zero_state(tf.shape(input_layer)[0], dtype=tf.float32)
        # wrap the lstm nn
        outputs, _ = tf.nn.dynamic_rnn(stacked_cell, input_layer, initial_state=_init_state, dtype=tf.float32)

        net = slim.flatten(outputs)

        net = tf.layers.dense(net, units=128)
        #net = tf.layers.dropout(net, rate=1-keep_prob, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(net, units=2)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        one_hot_labels = tf.one_hot(labels, depth=2)
        loss = tf.losses.mean_pairwise_squared_error(labels=one_hot_labels, predictions=predictions['probabilities'])

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def fit(self, feature, label):
        batch_size, num_epochs, steps = self.batch_size, self.epochs, self.steps

        # Because the multi process cannot deepcopy estimator,
        # Cannot only build the model when use,
        # Lucky that the model can be reload by its model dir.
        # Create input fn

        # append the lstm data from 298 to 300
        _feature = self.prepare_fea(feature, label)
        _feature = self.change2d(_feature, 300 - len(_feature[0]))

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": _feature},
            y=label,
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle=True
        )
        # Training
        self.model.train(input_fn=train_input_fn,
                                   steps=steps,
                                   hooks=[self.logging_hook])
        self.fit_done = 1

    def predict_prob(self, feature):
        # Because the multi process cannot deepcopy estimator,
        # Cannot only build the model when use,
        # Lucky that the model can be reload by its model dir.

        # append the lstm features from 298 to 300
        _feature = self.prepare_fea(feature)
        _feature = self.change2d(_feature, 300-len(_feature[0]))

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": _feature},
            num_epochs=1,
            shuffle=False
        )
        predictions = list(self.model.predict(input_fn=predict_input_fn))

        predicted_prob = list(np.array([p["probabilities"] for p in predictions])[:, 1])

        return predicted_prob