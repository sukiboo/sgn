'''
	this file contains functions related to neural network training and evaluation
'''


import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import matplotlib.pyplot as plt
import numpy as np



'''
	train a shallow neural network with (num_nodes) nodes
	and optional weights initialization (weight_init)
	training data is given by (x,f)
	returns the trained weights (A,B,C,c)

	optionally display the model summary and plots the training process
'''
def train_nn(x, f, num_nodes, weights_init=None, summary=False, plot=False):

	# set hyperparameters
	lr_start = 1e-03
	lr_decay = .9995
	learning_rate = LearningRateScheduler(lambda epoch: lr_start * lr_decay**epoch)
	epochs = 1000 if weights_init is not None else 10000
	batch_size = x.shape[-1] if weights_init is not None else 1

	# establish the dimensionality of the data
	if len(x.shape) == 1:
		dim = 1
	elif len(x.shape) == 2:
		x = x.T
		dim = 2
	else:
		print('\nerror: the dimensionality of x is not 1 or 2')
		return

	# initialize weights if A,B,C are provided
	if weights_init is not None:
		def A_init(shape, dtype=None):
			return weights_init[0].T
		def B_init(shape, dtype=None):
			return weights_init[1].reshape(-1)
		def C_init(shape, dtype=None):
			return weights_init[2]

	# define the model
	model = Sequential()
	if weights_init:
		model.add(Dense(num_nodes, input_shape=(dim,), \
			kernel_initializer=A_init, bias_initializer=B_init))
		model.add(Activation('relu'))
		model.add(Dense(1, \
			kernel_initializer=C_init))
	else:
		model.add(Dense(num_nodes, input_shape=(dim,), \
			kernel_initializer=tf.keras.initializers.TruncatedNormal(), \
			bias_initializer=tf.keras.initializers.Zeros()))
		model.add(Activation('relu'))
		model.add(Dense(1, \
			kernel_initializer=tf.keras.initializers.TruncatedNormal(), \
			bias_initializer=tf.keras.initializers.Zeros()))

	# compile the model
	model.compile(loss='mean_squared_error', \
		optimizer=tf.keras.optimizers.Adam(lr=lr_start))

	# display the model
	if summary:
		print()
		model.summary()

	# display network parameters
	print('\ntraining the neural network...')
	print('network parameters: epochs = {:d}, batch size = {:d}'.format(epochs, batch_size))
	if weights_init:
		label = 'GSN loss'
		print('training with GSN initialization...')
		print('this usually takes up to 30 seconds...\n')
	else:
		label = 'random loss'
		print('training network using random initialization...')
		print('this usually takes up to 3 minutes in 1d case and up to 30 minutes in 2d...\n')

	# train the model
	history = model.fit(x, f, batch_size=batch_size, epochs=epochs, \
						verbose=0, callbacks=[learning_rate])

	# plot the training process
	if plot:
		fig = plt.figure(figsize=(20,4))
		plt.semilogy(history.history['loss'], label=label)
		plt.legend(loc='upper right')
		plt.show()

	# return trained weights
	A = model.layers[0].get_weights()[0].T
	B = model.layers[0].get_weights()[1].reshape((-1,1))
	C = model.layers[-1].get_weights()[0].reshape((-1,1))
	c = model.layers[-1].get_weights()[1]

	# evaluate the model
	model.evaluate(x, f, batch_size=x.shape[0])

	return A, B, C, c




''' this function evaluates the network C*sigma(Ax+B)+c at points (x) '''
def eval_nn(A, B, C, c, sigma, x):

	# evaluate the network in 1d case
	if len(np.array(x).shape) == 1:
		f = (np.matmul(C.T, sigma(A*x + B)) + c).ravel()

	# evaluate the network in 2d case
	elif len(np.array(x).shape) == 2:
		f = np.matmul(C.T, sigma(np.matmul(A, x) + B) + c).ravel()

	return f



