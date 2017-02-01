import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)  # one_hot means that one output is on the rest is off

hm_epochs = 10  # Can be changed for speed
n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28  # Feed 28 chunks of 28 (1 image) into the RNN cell at a time
rnn_size = 128  # Single size of the RNN (there will be 128 RNN cells)

# height x width
x = tf.placeholder("float", [None, n_chunks, chunk_size])
y = tf.placeholder("float")


def recurrent_neural_network(x):

    layer = {"weights": tf.Variable(tf.random_normal([rnn_size, n_classes])),
             "biases": tf.Variable(tf.random_normal([n_classes]))}  # Creates a TensorFlow variable (a

    x = tf.transpose(x, [1, 0, 2])  # See transpose_example.py to see what transpose actually does. Basically, it just
    # formats the data in a way that tf can accepts
    x = tf.reshape(x, [-1, chunk_size])  # Reshapes the size of the matrix x
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer["weights"]) + layer["biases"]

    return output


def train_neural_network(epoch_x):

    prediction = recurrent_neural_network(epoch_x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch, "completed out of", hm_epochs, "loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # Comparing the predictions with the actual
        # value
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))

print("here")
train_neural_network(x)
