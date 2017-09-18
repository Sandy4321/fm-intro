import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Import
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

#=== Build Graph

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Initialize
sess.run(tf.global_variables_initializer())

# Model
y = tf.matmul(x,W) + b

# Loss
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Evaluate
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Epochs
for _ in range(1000):
	if _ % 100 == 0:
		print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("=== FINAL ===")
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
