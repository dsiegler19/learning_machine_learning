# A tensor is simply an array. This means that at its core, TensorFlow is just a matrix manipulation library.
# When working with tensorflow, first a the problem is abstracted and put into code. Then tensorflow does the bulk of
# the calculations backend and returns them to Python. The abstraction process is done using a series of graphs.
# TensorFlow's calculations are done on computational DAGs.

import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.mul(x1, x2)

print(result)  # This just turns out to be an abstract Tensor in the computation graph. Since nothing has been done to
# result, this simply outputs Tensor("Mul:0", shape=(), dtype=int32). result is simply a model defining the
# multiplication of 5 and 6.

with tf.Session() as sess:  # Since a Session is like a file, it has to be opened and closed.
    print(sess.run(result))  # However, by actually running a Session on the Tensor, it produces the expected result of
    # 30. This actually runs the model that is result.
    output = sess.run(result)  # Now sess's run in stored in a Python variable, not just a computational graph

print(output)  # This means that we can access it even when sess is closed.
