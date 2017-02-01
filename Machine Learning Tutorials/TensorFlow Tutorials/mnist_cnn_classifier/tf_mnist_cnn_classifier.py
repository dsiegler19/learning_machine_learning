import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)  # one_hot means that one output is on the rest is off

n_classes = 10

batch_size = 128

hm_epochs = 3  # Can be raised to 10

# height x width
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")  # strides is the stride size (the number of
    # pixels the window moves each iteration) of the convolution window pane. padding means when the window reaches the
    # edge of the frame, instead of hanging off it pads the edges with the same values as the last rows of the image.


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # strides is the stride size
    # of the max pooling window pane. padding means when the window reaches the edge of the frame, instead of hanging
    # off it pads the edges with the same values as the last rows of the image. ksize is the size of the window.


def convolutional_neural_network(x):
    weights = {"W_conv1": tf.Variable(tf.random_normal([5, 5, 1, 32])),  # 5x5 window, takes 1 input, makes 32 outputs
               "W_conv2": tf.Variable(tf.random_normal([5, 5, 32, 64])),  # Fully connected layer
               "W_fc": tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               "out": tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {"b_conv1": tf.Variable(tf.random_normal([32])),
               "b_conv2": tf.Variable(tf.random_normal([64])),
               "b_fc": tf.Variable(tf.random_normal([1024])),
               "out": tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights["W_conv1"]) + biases["b_conv1"])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights["W_conv2"]) + biases["b_conv2"])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights["W_fc"] + biases["b_fc"]))

    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights["out"] + biases["out"])

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch, "completed out of", hm_epochs, "loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # Comparing the predictions with the actual
        # value
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
