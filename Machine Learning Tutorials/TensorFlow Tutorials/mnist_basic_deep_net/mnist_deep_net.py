# input > weights > hidden layer 1 (activation function) > hidden layer 2 (activation function) > weights hidden layer 3
# (activation function) > weights > output layer

# compare output to intended function > cost or loss function (cross entropy)
# optimization function / optimizer > minimize cost (AdamOptimizer, SGD, AdaGrad)
# backpropagation

# feed forward + backprop = epoch
# many epochs (in this case ~10)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)  # one_hot means that one output is on the rest is off
# For example, there are 10 classes, 0 - 9
# 0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# 2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# ...

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

batch_size = 100  # Goes through batches of 100 samples (because some data sets may be too large to load all at once
# into memory)

# height x width
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")


def neural_network_model(data):
    hidden_1_layer = {"weights": tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl1]))}  # Creates a TensorFlow variable (a
    # Tensor) of random weights and biases for the first hidden layer
    # What is bias? Well:
    # input_data * weights + bias

    hidden_2_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    "biases": tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weights"]), hidden_2_layer["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer["weights"]), hidden_3_layer["biases"])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]

    return output


def train_neural_network(epoch_x):
    prediction = neural_network_model(epoch_x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Number of cycles of feed forward and backprop
    hm_epochs = 10  # Can be raised to 10

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
