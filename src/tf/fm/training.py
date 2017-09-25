
import argparse
import os
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Local modules
import fm as model
import load_data

# Basic model parameters as external flags.
class Flags(object):

	def __init__(self):
		self.batch_size = 1000
		self.fake_data = False
		self.input_data_dir = "../../../data/ml-100k"
		self.input_data_filename = 'u.data'
		self.log_dir = './logs'
		self.latent_factors = 1
		self.hidden_2 = 32
		self.learning_rate = 0.01
		self.max_steps = 1000

# Initialize
FLAGS = Flags()


def placeholder_inputs(batch_size, num_feature):
	"""Generate placeholder for inputs
	"""

	input_placeholder = tf.placeholder(
		tf.float32,
		shape=(batch_size, num_feature))
	labels_placeholder = tf.placeholder(
		tf.int32,
		shape=(batch_size))
	return input_placeholder, labels_placeholder


def fill_feed_dict(data_set, feature_pl, labels_pl):
	"""Fill the placeholders with actual data

	Return:
		This is what get fed into the evaluation data
	"""

	# Create the feed_dict output for the placeholders filled with the 
	# batch data
	input_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
	feed_dict = {
		feature_pl: input_feed,
		labels_pl: labels_feed
	}
	return feed_dict


def do_eval(sess,
			eval_correct,
			input_placeholder,
			labels_placeholder,
			data_set):
	"""Evaluate using the model.eval_correct()
	"""

	true_count = 0  # Count how many correct
	steps_per_epoch = data_set.num_examples // FLAGS.batch_size
	num_examples = steps_per_epoch * FLAGS.batch_size
	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(
			data_set, 
			input_placeholder, 
			labels_placeholder)

		print('feed_dict')
		print(feed_dict)
		true_count += sess.run(eval_correct, feed_dict=feed_dict)
	precision = float(true_count) / num_examples
	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
		(num_examples, true_count, precision))



def run_training():
	"""Train data for a number of steps
	"""
	# Get data
	data_sets = load_data.sparse_to_dense(
		path=FLAGS.input_data_dir, 
		filename=FLAGS.input_data_filename, 
		header=['user_id','item_id','rating','timestamp'],
		cols=['user_id','item_id','rating'])

	# Build a single graph
	with tf.Graph().as_default():
		
		# Generate placeholders
		input_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size, data_sets.num_feature)

		# Build a Graph that computes predictions
		logits = model.model(
			input_placeholder, 
			FLAGS.latent_factors,
			data_sets.num_feature)

		# Get Loss from data
		loss = model.loss(logits, labels_placeholder)

		# Training op
		train_op = model.training(loss, FLAGS.learning_rate)

		# Evaluate
		eval_correct = model.evaluation(logits, labels_placeholder)

		# Get Summary Tensor
		summary = tf.summary.merge_all()

		# Initialize
		init = tf.global_variables_initializer()

		# Save the training checkpoints
		saver = tf.train.Saver()

		# Session for the Op and Graph
		sess = tf.Session()

		# Instantiate a SummaryWriter to output summaries and the Graph.
		summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

		# Run
		sess.run(init)

		# Start the training loop
		for step in xrange(FLAGS.max_steps):

			# Record time
			start_time = time.time()

			# Print Dataset
			if step == 0:
				pass
				# print(vars(data_sets))

			# Feed the training data
			feed_dict = fill_feed_dict(
				data_sets.train,
				input_placeholder,
				labels_placeholder)


			# Print Dataset
			if step == 0:
				print(feed_dict)

			# Run a step of the model defined by 'train_op' and 'loss'
			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

			# End Time
			duration = time.time() - start_time

			# Print Evaluation
			if step % 100 == 0:
				# Print stdout
				print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

				# Update the events file.
				summary_str = sess.run(summary, feed_dict=feed_dict)  # This gets a summary of all practical metrics
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
				saver.save(sess, checkpoint_file, global_step=step)
				# Evaluate on all three datasets
				print('Training Data Eval:')
				do_eval(sess,
						eval_correct,
						input_placeholder,
						labels_placeholder,
						data_sets.train)
				# Evaluate against the validation set.
				print('Validation Data Eval:')
				do_eval(sess,
						eval_correct,
						input_placeholder,
						labels_placeholder,
						data_sets.validation)
				# Evaluate against the test set.
				print('Test Data Eval:')
				do_eval(sess,
						eval_correct,
						input_placeholder,
						labels_placeholder,
						data_sets.test)



def main(_):
	# Make Log Directory if it doesn't exist
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	run_training()


if __name__ == '__main__':
	tf.app.run(main=main, argv=[sys.argv[0]])