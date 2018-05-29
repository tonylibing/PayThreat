import tensorflow as tf

slim = tf.contrib.slim


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
	"""Builds the 35x35 resnet block."""
	with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
		with tf.variable_scope('Branch_0'):
			tower_conv = tf.layers.conv1d(net, 16, 1, name='conv1d_1x1')
		with tf.variable_scope('Branch_1'):
			tower_conv1_0 = tf.layers.conv1d(net, 16, 1, name='conv1d_0a_1x1')
			tower_conv1_1 = tf.layers.conv1d(tower_conv1_0, 16, 3, name='conv1d_0b_3x3', padding='same')
		with tf.variable_scope('Branch_2'):
			tower_conv2_0 = tf.layers.conv1d(net, 16, 1, name='conv1d_0a_1x1')
			tower_conv2_1 = tf.layers.conv1d(tower_conv2_0, 24, 3, name='conv1d_0b_3x3', padding='same')
			tower_conv2_2 = tf.layers.conv1d(tower_conv2_1, 32, 3, name='conv1d_0c_3x3', padding='same')
		mixed = tf.concat(axis=2, values=[tower_conv, tower_conv1_1, tower_conv2_2])
		up = tf.layers.conv1d(mixed, net.get_shape()[2], 1,  padding='same', activation=None, name='conv1d_1x1')
		scaled_up = up * scale
		if activation_fn == tf.nn.relu6:
			# Use clip_by_value to simulate bandpass activation.
			scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

		net += scaled_up
		if activation_fn:
			net = activation_fn(net)
	return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
	"""Builds the 17x17 resnet block."""
	with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
		with tf.variable_scope('Branch_0'):
			tower_conv = tf.layers.conv1d(net, 48, 1, name='conv1d_1x1')
		with tf.variable_scope('Branch_1'):
			tower_conv1_0 = tf.layers.conv1d(net, 32, 1, name='conv1d_0a_1x1')
			tower_conv1_1 = tf.layers.conv1d(tower_conv1_0, 64, 7, padding='same',
											 name='conv1d_0b_1x7')
			tower_conv1_2 = tf.layers.conv1d(tower_conv1_1, 48, 7, padding='same',
											 name='conv1d_0c_7x1')
		mixed = tf.concat(axis=2, values=[tower_conv, tower_conv1_2])
		up = tf.layers.conv1d(mixed, net.get_shape()[2], 1, padding='same',
							  activation=None, name='conv1d_1x1')

		scaled_up = up * scale
		if activation_fn == tf.nn.relu6:
			# Use clip_by_value to simulate bandpass activation.
			scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

		net += scaled_up
		if activation_fn:
			net = activation_fn(net)
	return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
	"""Builds the 8x8 resnet block."""
	with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
		with tf.variable_scope('Branch_0'):
			tower_conv = tf.layers.conv1d(net, 48, 1, name='conv1d_1x1')
		with tf.variable_scope('Branch_1'):
			tower_conv1_0 = tf.layers.conv1d(net, 48, 1, name='conv1d_0a_1x1')
			tower_conv1_1 = tf.layers.conv1d(tower_conv1_0, 56, 3, padding='same',
											 name='conv1d_0b_1x3')
			tower_conv1_2 = tf.layers.conv1d(tower_conv1_1, 64, 3, padding='same',
											 name='conv1d_0c_3x1')
		mixed = tf.concat(axis=2, values=[tower_conv, tower_conv1_2])
		up = tf.layers.conv1d(mixed, net.get_shape()[2], 1, padding='same', activation=None, name='conv1d_1x1')

		scaled_up = up * scale
		if activation_fn == tf.nn.relu6:
			# Use clip_by_value to simulate bandpass activation.
			scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

		net += scaled_up
		if activation_fn:
			net = activation_fn(net)
	return net


def inception_fn(features, labels, mode):
	input_layer = tf.cast(features['x'], tf.float32)

	padding = 'same'
	output_strides = 16
	# 148 x 32
	net = tf.layers.conv1d(input_layer, 32, 3, strides=2, padding=padding,
						   name='conv1d_1a_3x3')

	# 146 x 32
	net = tf.layers.conv1d(net, 32, 3,
						   name='conv1d_2a_3x3')
	# 146 x 64
	net = tf.layers.conv1d(net, 64, 3, name='conv1d_2b_3x3', padding=padding)

	# 73 x 73 x 64
	net = tf.layers.max_pooling1d(net, 3, strides=2, padding=padding,
								  name='MaxPool_3a_3x3')
	# 73 x 73 x 80
	net = tf.layers.conv1d(net, 80, 1, padding=padding,
						   name='conv1d_3b_1x1')

	# 71 x 71 x 192
	net = tf.layers.conv1d(net, 192, 3,
						   name='conv1d_4a_3x3')

	# 35 x 35 x 192
	net = tf.layers.max_pooling1d(net, 3, strides=2, padding=padding,
								  name='MaxPool_5a_3x3')
	# 35 x 35 x 320
	with tf.variable_scope('Mixed_5b'):
		with tf.variable_scope('Branch_0'):
			tower_conv = tf.layers.conv1d(net, 96, 1, name='conv1d_1x1')
		with tf.variable_scope('Branch_1'):
			tower_conv1_0 = tf.layers.conv1d(net, 48, 1, name='conv1d_0a_1x1')
			tower_conv1_1 = tf.layers.conv1d(tower_conv1_0, 64, 5, padding=padding,
											 name='conv1d_0b_5x5')
		with tf.variable_scope('Branch_2'):
			tower_conv2_0 = tf.layers.conv1d(net, 64, 1, name='conv1d_0a_1x1')
			tower_conv2_1 = tf.layers.conv1d(tower_conv2_0, 96, 3, padding=padding,
											 name='conv1d_0b_3x3')
			tower_conv2_2 = tf.layers.conv1d(tower_conv2_1, 96, 3, padding=padding,
											 name='conv1d_0c_3x3')
		with tf.variable_scope('Branch_3'):
			tower_pool = tf.layers.average_pooling1d(net, 3, strides=1, padding='SAME',
													 name='AvgPool_0a_3x3')
			tower_pool_1 = tf.layers.conv1d(tower_pool, 64, 1,
											name='conv1d_0b_1x1')
		net = tf.concat(
			[tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 2)

	# TODO(alemi): Register intermediate endpoints
	net = slim.repeat(net, 10, block35, scale=0.17, activation_fn=tf.nn.relu)

	# 17 x 17 x 1088 if output_strides == 8,
	# 33 x 33 x 1088 if output_strides == 16
	use_atrous = output_strides == 8

	with tf.variable_scope('Mixed_6a'):
		with tf.variable_scope('Branch_0'):
			tower_conv = tf.layers.conv1d(net, 384, 3, strides=1 if use_atrous else 2,
										  padding=padding,
										  name='conv1d_1a_3x3')
		with tf.variable_scope('Branch_1'):
			tower_conv1_0 = tf.layers.conv1d(net, 256, 1, name='conv1d_0a_1x1')
			tower_conv1_1 = tf.layers.conv1d(tower_conv1_0, 256, 3, padding=padding,
											 name='conv1d_0b_3x3')
			tower_conv1_2 = tf.layers.conv1d(tower_conv1_1, 384, 3,
											 strides=1 if use_atrous else 2,
											 padding=padding,
											 name='conv1d_1a_3x3')
		with tf.variable_scope('Branch_2'):
			tower_pool = tf.layers.max_pooling1d(net, 3, strides=1 if use_atrous else 2,
												 padding=padding,
												 name='MaxPool_1a_3x3')
		net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 2)

	# TODO(alemi): register intermediate endpoints
	with tf.variable_scope('block17'):
		net = slim.repeat(net, 20, block17, scale=0.10,
						  activation_fn=tf.nn.relu)
	# 8 x 8 x 2080
	with tf.variable_scope('Mixed_7a'):
		with tf.variable_scope('Branch_0'):
			tower_conv = tf.layers.conv1d(net, 256, 1, name='conv1d_0a_1x1')
			tower_conv_1 = tf.layers.conv1d(tower_conv, 384, 3, strides=2,
											padding=padding,
											name='conv1d_1a_3x3')
		with tf.variable_scope('Branch_1'):
			tower_conv1 = tf.layers.conv1d(net, 256, 1, name='conv1d_0a_1x1')
			tower_conv1_1 = tf.layers.conv1d(tower_conv1, 288, 3, strides=2,
											 padding=padding,
											 name='conv1d_1a_3x3')
		with tf.variable_scope('Branch_2'):
			tower_conv2 = tf.layers.conv1d(net, 256, 1, name='conv1d_0a_1x1')
			tower_conv2_1 = tf.layers.conv1d(tower_conv2, 288, 3,
                                             padding=padding,
											 name='conv1d_0b_3x3')
			tower_conv2_2 = tf.layers.conv1d(tower_conv2_1, 320, 3, strides=2,
											 padding=padding,
											 name='conv1d_1a_3x3')
		with tf.variable_scope('Branch_3'):
			tower_pool = tf.layers.max_pooling1d(net, 3, strides=2,
												 padding=padding,
												 name='MaxPool_1a_3x3')
		net = tf.concat(
			[tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 2)

	# TODO(alemi): register intermediate endpoints
	net = slim.repeat(net, 9, block8, scale=0.20, activation_fn=tf.nn.relu)
	net = block8(net, activation_fn=None)

	# 8 x 8 x 1536
	net = tf.layers.conv1d(net, 1536, 1, name='conv1d_7b_1x1')

	with tf.variable_scope('Logits'):
		# TODO(sguada,arnoegw): Consider adding a parameter global_pool which
		# can be set to False to disable pooling here (as in resnet_*()).
		kernel_size = net.get_shape()[1:2]
		if kernel_size.is_fully_defined():
			net = tf.layers.average_pooling1d(net, kernel_size, padding='VALID', strides=1,
											  name='AvgPool_1a_8x8')
		else:
			net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
		net = slim.flatten(net)
		net = slim.dropout(net, 0.4, is_training=True, scope='Dropout')
		logits = slim.fully_connected(net, 2, activation_fn=None,
									  scope='Logits')

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

