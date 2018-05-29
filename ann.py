import tensorflow as tf

x = tf.placeholder("float", [None, 298])
w_1 = tf.Variable(tf.zeros([298, 1000]))
b_1 = tf.Variable(tf.zeros([1000]))
w_2 = tf.Variable(tf.zeros([1000, 150]))
b_2 = tf.Variable(tf.zeros([150]))
w_3 = tf.Variable(tf.zeros([150, 2]))
b_3 = tf.Variable(tf.zeros([2]))

_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)
_2 = tf.nn.relu(tf.matmul(_1, w_2) + b_2)
_3 = tf.nn.relu(tf.matmul(_2, w_3) + b_3)

y = tf.nn.softmax(_3)

real_y = tf.placeholder("float", [None,2])
cross_entropy = -tf.reduce_sum(real_y * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

train_x, train_y, test_x, test_y = []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(train_step, feed_dict = {x: train_x, real_y: train_y})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(real_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    accuracy = sess.run(accuracy, feed_dict = {x: test_x, real_y: test_y})


