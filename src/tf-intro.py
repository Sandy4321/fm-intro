import tensorflow as tf

# Constants
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# Session
sess = tf.Session()
print(sess.run([node1, node2, node3]))


# Placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
add_and_tripple = adder_node * 3

print(sess.run(adder_node, {a: [1, 4], b: [2, 9]}))
print(sess.run(add_and_tripple, {a: [1, 4], b: [2, 9]}))


# Variables
W = tf.Variable([-1.9], dtype=tf.float32)
b = tf.Variable([1], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = b + W * x

# Loss Functions
squared_delta = tf.square(y - linear_model)
loss = tf.reduce_sum(squared_delta)

# Initialize
init = tf.global_variables_initializer()
sess.run(init)

# Run model
print(sess.run(loss, {x: [1, 2, 3, 4], y: [-1, 0, 1, 2]}))

# Update
# fixW = tf.assign(W, [1.])
# fixb = tf.assign(b, [-2.])
# sess.run([fixW, fixb])
# print(sess.run(loss, {x: [1, 2, 3, 4], y: [-1, 0, 1, 2]}))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
	if i % 100 == 0:
		print('Epoch ', i, 'loss', sess.run(loss, {x: [1, 2, 3, 4], y: [-1, 0, 1, 2]}))
	sess.run(train, {x: [1, 2, 3, 4], y: [-1, 0, 1, 2]})

print(sess.run([W, b]))