from BaseModel import BaseModel
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
class DenseModel(BaseModel):
    def __init__(self, args):
        super().__init__()
        # feature deal, model init and fit args
        self.normalize, self.fea_deal, \
        self.layers, self.layer_units, \
        self.batch_size, self.epochs, self.steps, \
        self.model_index = \
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]
        # if the model has fitted
        self.fit_done = 0
        # nn model has model_dir
        self.model_dir = './model/dense_model_' + str(self.model_index) + '/'
        # define model
        self.model =  tf.estimator.Estimator(
            model_fn=self.dense_baseline_fn, model_dir=self.model_dir, params=(self.layers, self.layer_units))

        # tensorflow init
        tf.logging.set_verbosity(tf.logging.ERROR)
        # Set up logging for predictions
        tensors_to_log = {"probabilities": "softmax_tensor"}
        self.logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

    @staticmethod
    def get_name():
        return 'dense'

    @staticmethod
    def nn_model():
        return True

    def dense_baseline_fn(self, features, labels, mode, params):
        layers, layer_units = params[0], params[1]
        net = tf.cast(features['x'], tf.float32)

        for i in range(layers):
            net = tf.keras.layers.Dense(layer_units[i], activation='relu')(net)

        # layer3 256 >> 2
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

        _feature = self.prepare_fea(feature, label)

        # Create input fn
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

        #
        _feature = self.prepare_fea(feature)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": _feature},
            num_epochs=1,
            shuffle=False
        )
        predictions = list(self.model.predict(input_fn=predict_input_fn))
        # 1 / 0 + 1
        predicted_prob = list(np.array([p["probabilities"] for p in predictions])[:, 1])
        return predicted_prob