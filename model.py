import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
import sklearn
import inception_model_2
import get_data

slim = tf.contrib.slim
def deal_result(x):
    if x == 1:
        return str(1)
    else:
        return str(0)

def evaluate(y_true_train, y_predict_train, y_true_test, y_predict_test):
    # fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=y_true,y_score=y_score, pos_label=2)
    # score = 0
    # for index in range(len(fpr)):
    #     if fpr[index] == 0.001:
    #         score += 0.4 * tpr[index]
    #     elif fpr[index] == 0.005:
    #         score += 0.3 * tpr[index]
    #     elif fpr[index] == 0.01:
    #         score += 0.3 * tpr[index]
    # print('match score:', score)
    print('train acc: ', sklearn.metrics.accuracy_score(y_true_train, y_predict_train))
    print('evaluate acc: ', sklearn.metrics.accuracy_score(y_true_test, y_predict_test))

def gbdt_baseline():
    train_features, train_label, test_features, test_label, real_test_id, real_test_features = get_data.traditional_input(pca_fea=True)

    gbdt_model = GradientBoostingClassifier()
    gbdt_model.fit(train_features, train_label)

    # for local evaluate
    test_result_prob = gbdt_model.predict_proba(test_features)[:,1]
    test_result = gbdt_model.predict(test_features)
    train_result = gbdt_model.predict(train_features)
    evaluate(train_label, train_result, test_label, test_result)

    # generate result for real test
    real_test_result = gbdt_model.predict_proba(real_test_features)[:,1]
    _ = {'id':real_test_id, 'score':real_test_result}
    result = pd.DataFrame(_)
    result.to_csv('gbdt_result.csv',index=False)

def change2d(input, pad_number):
    pad_vec = np.zeros(shape=(input.shape[0], pad_number))
    result = np.column_stack((input, pad_vec))

    return result

def cnn_baseline_main():
    # train_features shape[sample_number, 298]
    train_features, train_label, test_features, test_label, real_id, real_features = get_data.nn_input()

    tf.logging.set_verbosity(tf.logging.INFO)
    train_features = change2d(train_features, 17*18-298)
    test_features = change2d(test_features, 17*18-298)
    real_features = change2d(real_features, 17*18-298)

    train_features = np.reshape(train_features, (-1, 17, 18))
    test_features = np.reshape(test_features, (-1, 17, 18))
    real_features = np.reshape(real_features, (-1, 17, 18))

    # Create the Estimator
    paythreat_classifier = tf.estimator.Estimator(
        model_fn=cnn_baseline_fn, model_dir="./model/cnn_baseline_model/")
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    # Create input fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_features},
        y = train_label,
        batch_size = 500,
        num_epochs= 10,
        shuffle=True
    )
    # Training
    paythreat_classifier.train(input_fn=train_input_fn,
                               steps=100000,
                               hooks=[logging_hook])

    # evaluate on train datasets
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_features},
        y=train_label,
        num_epochs=1,
        shuffle=False)
    eval_results = paythreat_classifier.evaluate(input_fn=eval_input_fn)
    print('train results: ', eval_results)

    # evaluate on test datasets
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_features},
        y=test_label,
        num_epochs=1,
        shuffle=False)
    eval_results = paythreat_classifier.evaluate(input_fn=eval_input_fn)
    print('evalate results: ', eval_results)

    # Predict base on the real features
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": real_features},
        num_epochs=1,
        shuffle=False
    )
    predictions = list(paythreat_classifier.predict(input_fn=predict_input_fn))
    predicted_prob = list(np.array([p["probabilities"] for p in predictions])[:,1])
    _ = {'id': real_id, 'score': predicted_prob}
    result = pd.DataFrame(_)
    result.to_csv('cnn_result.csv', index=False)

def cnn_baseline_fn(features, labels, mode):
    input_layer = tf.reshape(tf.cast(features["x"], tf.float32), [-1, 17, 18, 1])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 64])
    net = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=net, units=2)

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

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def dense_baseline_main():
    print('get data...')
    train_features, train_label, test_features, test_label, real_id, real_features = get_data.nn_input()
    print('get data done.')
    tf.logging.set_verbosity(tf.logging.INFO)
    # Create the Estimator
    paythreat_classifier = tf.estimator.Estimator(
        model_fn=dense_baseline_fn, model_dir="./model/dense_baseline_model/")
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    # Create input fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_features},
        y=train_label,
        batch_size=500,
        num_epochs=100,
        shuffle=True
    )
    # Training
    print('train paythreat....')
    paythreat_classifier.train(input_fn=train_input_fn,
                               steps=200000,
                               hooks=[logging_hook])
    print('train paythreat done.')
    # evaluate on train datasets
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_features},
        y=train_label,
        num_epochs=1,
        shuffle=False)
    eval_results = paythreat_classifier.evaluate(input_fn=eval_input_fn)
    print('train results: ', eval_results)

    # evaluate on test datasets
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_features},
        y=test_label,
        num_epochs=1,
        shuffle=False)
    eval_results = paythreat_classifier.evaluate(input_fn=eval_input_fn)
    print('evalate results: ', eval_results)

    # Predict base on the real features
    print('predict...')
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": real_features},
        num_epochs=1,
        shuffle=False
    )
    predictions = list(paythreat_classifier.predict(input_fn=predict_input_fn))
    print('predict done.')
    predicted_prob = list(np.array([p["probabilities"] for p in predictions])[:, 1])
    _ = {'id': real_id, 'score': predicted_prob}
    result = pd.DataFrame(_)
    result.to_csv('dense_result.csv', index=False)

def dense_baseline_fn(features, labels, mode):
    input_layer = tf.cast(features['x'], tf.float32)

    # layer1 298 >> 512
    layer1 = tf.keras.layers.Dense(512, activation='relu')(input_layer)

    # layer2 512 >> 256
    layer2 = tf.keras.layers.Dense(256, activation='relu')(layer1)

    # layer3 256 >> 2
    logits = tf.layers.dense(layer2, units=2)

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

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def inception_main():
    print('get data...')
    train_features, train_label, test_features, test_label, real_id, real_features = get_data.nn_input(sample_number=500)
    print('get data done.')
    tf.logging.set_verbosity(tf.logging.INFO)

    # Change the shape to 3 dims, [batch, length, channels]
    train_features = np.reshape(train_features, [-1, 298, 1])
    test_features = np.reshape(test_features, [-1, 298, 1])
    real_features = np.reshape(real_features, [-1, 298, 1])

    # Set the run config
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.estimator.RunConfig(session_config=tf.ConfigProto(gpu_options=gpu_options))

    # Create the Estimator
    paythreat_classifier = tf.estimator.Estimator(
        model_fn=inception_model_2.inception_fn, model_dir="./model/inception_model/", config=config)
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    # Create input fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_features},
        y=train_label,
        batch_size=500,
        num_epochs=1,
        shuffle=True
    )
    # Training
    print('train paythreat....')
    paythreat_classifier.train(input_fn=train_input_fn,
                               steps=1,
                               hooks=[logging_hook])
    print('train paythreat done.')
    # Evaluate the model and print results

    print('evaluate paythreat....')
    # evaluate on train datasets
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_features},
        y=train_label,
        num_epochs=1,
        shuffle=False)
    eval_results = paythreat_classifier.evaluate(input_fn=eval_input_fn)
    print('train results: ', eval_results)

    # evaluate on test datasets
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_features},
        y=test_label,
        num_epochs=1,
        shuffle=False)
    eval_results = paythreat_classifier.evaluate(input_fn=eval_input_fn)
    print('evalate results: ', eval_results)

    # Predict base on the real features
    print('predict...')
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": real_features},
        num_epochs=1,
        shuffle=False
    )
    predictions = list(paythreat_classifier.predict(input_fn=predict_input_fn))
    print('predict done.')
    predicted_prob = list(np.array([p["probabilities"] for p in predictions])[:, 1])
    _ = {'id': real_id, 'score': predicted_prob}
    result = pd.DataFrame(_)
    result.to_csv('inception_result.csv', index=False)
inception_main()




