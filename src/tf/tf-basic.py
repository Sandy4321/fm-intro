import tensorflow as tf
import numpy as np


# Feature space
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# Optimizing
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# Data
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# Plugin data
input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": x_train}, 
	y=y_train,
	batch_size=4,
	num_epochs=None,
	shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": x_train},
	y=y_train,
	batch_size=4,
	num_epochs=1000,
	shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": x_eval},
	y=y_eval,
	batch_size=4,
	num_epochs=1000,
	shuffle=False)

# Estimator already built into the LinearRegressor
estimator.train(input_fn=input_fn, steps=1000)

# Evaluate how our model performed
train_metrics = estimator.evaluate(input_fn=train_input_fn)
test_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% test_metrics)
