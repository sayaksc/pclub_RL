import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



'''Initialiizng parameters'''
# input_layer = tf.reshape(x, [-1, 28, 28, 1])
X_ = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

W_C1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev = 0.1))
b_C1 = tf.Variable(tf.zeros([32]))

W_C3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1))
b_C3 = tf.Variable(tf.zeros([64]))

W_L5 = tf.Variable(tf.truncated_normal([7*7*64, 100], stddev = 0.1))
b_L5 = tf.Variable(tf.zeros([100]))

W_L6 = tf.Variable(tf.truncated_normal([100, 10], stddev = 0.1))
b_L6 = tf.Variable(tf.zeros([10]))

''' Architechture '''
#C1
conv1 = tf.nn.relu(tf.nn.conv2d(input = X_, filter = W_C1, strides = [1, 1, 1, 1], padding = "SAME") + b_C1)
#M2
pool1 = tf.nn.max_pool(value = conv1, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding = "SAME")
#C3
conv2 = tf.nn.relu(tf.nn.conv2d(input = pool1, filter = W_C3, strides = [1, 1, 1, 1], padding = "SAME") + b_C3)
#M4
pool2 = tf.nn.max_pool(value = conv2, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding = "SAME")
#Dense Layer
pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
#L5
dense = tf.nn.relu(tf.matmul(pool2_flat, W_L5)+b_L5)
#L6
Y_pred = tf.nn.softmax(tf.matmul(dense, W_L6) + b_L6)


cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y_pred))/100
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

'''Accuracy'''
correct_prediction = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''Training'''
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1500):
    x, target = mnist.train.next_batch(100)
    train_data = {X_: np.reshape(x, (100, 28, 28, 1)), Y_: target}
    # test_data = {X_: input_val, Y_: test_out}
    tr, loss = sess.run([train_step, cross_entropy], feed_dict = train_data)
    print "iteration: ", i, "loss: ", loss
# print "Accuracy: ", 100*sess.run(accuracy, feed_dict = test_data)
print("Accuracy: ", sess.run(accuracy, {X_: np.reshape(mnist.test.images, (-1, 28, 28, 1)), Y_: mnist.test.labels}))
