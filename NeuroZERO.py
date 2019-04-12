import numpy as np
import tensorflow as tf
import os
import copy
import pickle 
import struct
import sys
import argparse
import importlib
import shutil
import time
from scipy.optimize import linear_sum_assignment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NeuroZERO_filename = "NeuroZERO.obj"
neuron_base_name = "neuron_"
weight_base_name = "weight_"
bias_base_name = "bias_"
layer_base_name = "layer_"
baseline_network_name = 'baseline'
extended_network_name = "extended"

class NeuroZERO:
	def __init__(self):
		self.network_no = 1
		self.network_list = []
		self.network_dic = {}

	def parse_layers(self, layers_str):
		layers_list_str = layers_str.split(',')
		layers_list = []
		for layer_str in layers_list_str:
			layer_dimension_list = []
			layer_dimension_list_str = layer_str.split('*')

			for layer_dimension_str in layer_dimension_list_str:
				layer_dimension_list.append(int(layer_dimension_str))

			layers_list.append(layer_dimension_list)

		return layers_list

	def create_extended_network(self, base_layers):
		conv_layers = []

		for layer in range(0, len(base_layers)):
			if len(base_layers[layer]) >= 4:
				conv_layers.append(base_layers[layer])

		target_conv_layers = conv_layers[-2:]
		num_of_steps = target_conv_layers[0][3]

		steps = []
		for step in range(0, num_of_steps):
			step_up = []
			for i in range(0, len(conv_layers) - len(target_conv_layers)):
				step_up.append(conv_layers[i])
			for i in range(0, len(target_conv_layers)):
				conv = copy.deepcopy(target_conv_layers[i])
				conv[-1] = conv[-1] // num_of_steps * (step+1)
				if i == 1:
					conv[-2] = conv[-2] // num_of_steps * (step+1)
				step_up.append(conv)

			steps.append(step_up)

		extended_network = []
		for step in range(0, num_of_steps):
			step_up_network = copy.deepcopy(base_layers)
			conv_layer = 0
			for layer in range(0, len(step_up_network)):

				if len(step_up_network[layer]) >= 4:
					step_up_network[layer] = steps[step][conv_layer]
					conv_layer += 1
				elif len(step_up_network[layer]) == 1:
					if layer != 0 and layer != len(step_up_network)-1:
						step_up_network[layer] = [step_up_network[layer][0] // num_of_steps * (step+1) // 2]

			extended_network.append(step_up_network)

		return extended_network

	def constructNetwork(self, layers, name=None):
		print "constructNetwork %d:" % self.network_no, layers

		if name is None:
			network_name = "Network_" + str(self.network_no)
		else:
			network_name = name

		network = Network(self.network_no, network_name, layers, None, None)
		self.network_list.append((self.network_no, network, network_name))
		self.network_dic[network_name] = ((self.network_no, network))
		self.network_no += 1

	def constructNetworkExtended(self, layers, base_network, name=None):
		print "constructNetworkExtended %d:" % self.network_no, layers
		print "base_network:", base_network

		if name is None:
			network_name = "Network_" + str(self.network_no)
		else:
			network_name = name

		network = Network(self.network_no, network_name, layers, base_network, 0)
		self.network_list.append((self.network_no, network, network_name))
		self.network_dic[network_name] = ((self.network_no, network))
		self.network_no += 1

class Network:
	def __init__(self, network_no, network_name, layers, base_network=None, acceleration_mode=None):
		self.network_no = network_no
		self.network_state = 0
		self.network_name = network_name
		self.layers = layers
		self.layer_type, self.num_of_neuron_per_layer, self.num_of_weight_per_layer, self.num_of_bias_per_layer = self.calculate_num_of_weight(self.layers)

		self.num_of_neuron = 0
		for layer in self.num_of_neuron_per_layer:
			self.num_of_neuron += np.prod(layer)
			
		self.num_of_weight = sum(self.num_of_weight_per_layer)
		self.num_of_bias = sum(self.num_of_bias_per_layer)
		self.weight_vector = None
		self.bias_vector = None
		self.network_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.network_name)
		self.network_file_name = self.network_name
		self.network_file_path = os.path.join(self.network_dir, self.network_file_name)
		self.base_network = base_network
		self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

		graph = tf.Graph()
		with graph.as_default():
			with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
				if base_network is None and acceleration_mode is None:
					self.buildNetwork(sess)
				elif base_network is not None and acceleration_mode==0:
					self.buildNetworkExtended(sess, base_network, graph)

				self.saveNetwork(sess)
				self.updateParameterFromNetwork(sess, graph)

	def calculate_num_of_weight(self, layers, pad=0, stride=1):
		layer_type = []
		num_of_weight_per_layer = []
		num_of_bias_per_layer = []
		num_of_neuron_per_layer = []

		for layer in layers:
			if layer is layers[0]:
				type = 'input' # input
				layer_type.append(type)
				num_of_neuron_per_layer.append(layer)

			elif layer is layers[-1]:
				type = 'output' # output, fully-connected
				layer_type.append(type) 
				num_of_weight_per_layer.append(np.prod(layer)*np.prod(num_of_neuron_per_layer[-1]))
				num_of_bias_per_layer.append(np.prod(layer))
				num_of_neuron_per_layer.append(layer)

			elif len(layer) >= 4:
				type = 'conv' # convolutional
				layer_type.append(type) 

				#if len(layer) == 4:
				num_of_weight_per_layer.append(np.prod(layer))
				#if len(layer) == 5:
				#	num_of_weight_per_layer.append(np.prod(layer)/layer[-1])
				num_of_bias_per_layer.append(layer[3])

				h = (num_of_neuron_per_layer[-1][0] - layer[0] + 2*pad) / stride + 1
				w = (num_of_neuron_per_layer[-1][1] - layer[1] + 2*pad) / stride + 1
				d = layer[3]

				max_pool_f = 2
				max_pool_stride = 2

				h_max_pool = (h - max_pool_f) / max_pool_stride + 1
				w_max_pool = (w - max_pool_f) / max_pool_stride + 1
				d_max_pool = d

				#if len(layer) == 4:
				num_of_neuron_per_layer.append([h_max_pool,w_max_pool,d_max_pool])
				#if len(layer) == 5:
				#	num_of_neuron_per_layer.append([h_max_pool,w_max_pool,d_max_pool, 2])

				layer_type.append('max_pool') 

			else:
				type = 'hidden' # fully-connected
				layer_type.append(type) 
				num_of_weight_per_layer.append(np.prod(layer)*np.prod(num_of_neuron_per_layer[-1]))
				num_of_bias_per_layer.append(np.prod(layer))
				num_of_neuron_per_layer.append(layer)

		print 'layer_type:', layer_type 
		print 'num_of_neuron_per_layer:', num_of_neuron_per_layer
		print 'num_of_weight_per_layer:', num_of_weight_per_layer
		print 'num_of_bias_per_layer:', num_of_bias_per_layer

		return layer_type, num_of_neuron_per_layer, num_of_weight_per_layer, num_of_bias_per_layer

	def buildNetworkExtended(self, sess, base_network, graph):
		layer_type = copy.deepcopy(base_network.layer_type)
		layer_type = filter(lambda type: type != 'max_pool', layer_type)
		layers = base_network.layers
		print 'layers', layers
		parameters = {}
		base_neurons = {}
		parameters_to_regularize = []

		keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')

		base_neurons[0] = tf.placeholder(tf.float32, [None]+layers[0], name=neuron_base_name+'0')
		print base_neurons[0]

		for layer_no in range(1, len(layers)):
			weight_name = weight_base_name + str(layer_no-1) + '_base'
			bias_name = bias_base_name + str(layer_no-1) + '_base'
			neuron_name = neuron_base_name + str(layer_no) + '_base'
	
			if layer_type[layer_no] == "conv":
				conv_parameter = {
					'weights': tf.get_variable(weight_name,
						shape=(layers[layer_no]), trainable=False),
					'biases' : tf.get_variable(bias_name,
						shape=(layers[layer_no][3]), trainable=False),
				}

				parameters[layer_no-1] = conv_parameter
				print 'conv_parameter', parameters[layer_no-1]

				rank = sess.run(tf.rank(base_neurons[layer_no-1]))

				for _ in range(4 - rank):
					base_neurons[layer_no-1] = tf.expand_dims(base_neurons[layer_no-1], -1)

				# CNN
				strides = 1
				output = tf.nn.conv2d(base_neurons[layer_no-1],
					conv_parameter['weights'],
					strides=[1, strides, strides, 1], padding='VALID')
				output_biased = tf.nn.bias_add(output, conv_parameter['biases'])

				# max pooling
				k = 2
				new_neuron = tf.nn.max_pool(tf.nn.leaky_relu(output_biased),
				#new_neuron = tf.nn.max_pool(tf.nn.sigmoid(output_biased),
					ksize=[1, k, k, 1],
					strides=[1, k, k, 1], padding='VALID', name=neuron_name)

				base_neurons[layer_no] = new_neuron
				print 'new_neuron', new_neuron

		layer_type = copy.deepcopy(self.layer_type)
		layer_type = filter(lambda type: type != 'max_pool', layer_type)
		layers = self.layers
		parameters = {}
		new_neurons = {}
		parameters_to_regularize = []

		new_neurons[0] = base_neurons[0]
		print new_neurons[0]

		for layer_no in range(1, len(layers)):
			weight_name = weight_base_name + str(layer_no-1)
			bias_name = bias_base_name + str(layer_no-1)
			neuron_name = neuron_base_name + str(layer_no)
	
			if layer_type[layer_no] == "conv":
				conv_parameter = {
					'weights': tf.get_variable(weight_name,
						shape=(layers[layer_no]),
						initializer=tf.contrib.layers.xavier_initializer()),
					'biases' : tf.get_variable(bias_name,
						shape=(layers[layer_no][3]),
						initializer=tf.contrib.layers.xavier_initializer()),
				}

				#parameters_to_regularize.append(tf.reshape(conv_parameter['weights'], [tf.size(conv_parameter['weights'])]))
				#parameters_to_regularize.append(tf.reshape(conv_parameter['biases'], [tf.size(conv_parameter['biases'])]))

				parameters[layer_no-1] = conv_parameter
				print 'conv_parameter', parameters[layer_no-1]

				rank = sess.run(tf.rank(new_neurons[layer_no-1]))

				for _ in range(4 - rank):
					new_neurons[layer_no-1] = tf.expand_dims(new_neurons[layer_no-1], -1)

				# CNN
				strides = 1
				output = tf.nn.conv2d(new_neurons[layer_no-1],
					conv_parameter['weights'],
					strides=[1, strides, strides, 1], padding='VALID')
				output_biased = tf.nn.bias_add(output, conv_parameter['biases'])

				# max pooling
				k = 2
				new_neuron = tf.nn.max_pool(tf.nn.leaky_relu(output_biased),
				#new_neuron = tf.nn.max_pool(tf.nn.sigmoid(output_biased),
					ksize=[1, k, k, 1],
					strides=[1, k, k, 1], padding='VALID', name=neuron_name)
				new_neurons[layer_no] = new_neuron
				print 'new_neuron', new_neuron

		neurons = new_neurons

		for layer_no in range(1, len(layers)):
			weight_name = weight_base_name + str(layer_no-1)
			bias_name = bias_base_name + str(layer_no-1)
			neuron_name = neuron_base_name + str(layer_no)

			base_weight_name = weight_base_name + str(layer_no-1) + '_base'
			base_bias_name = bias_base_name + str(layer_no-1) + '_base'
			base_neuron_name = neuron_base_name + str(layer_no) + '_base'

			new_weight_name = weight_base_name + str(layer_no-1) + '_new'
			new_bias_name = bias_base_name + str(layer_no-1) + '_new'
			new_neuron_name = neuron_base_name + str(layer_no) + '_new'

			if layer_type[layer_no] == "hidden" or layer_type[layer_no] == "output":

				if layer_type[layer_no] == "hidden":
					base_fc_parameter_untrainable = {
						'weights': tf.get_variable(base_weight_name,
							shape=(np.prod(base_network.num_of_neuron_per_layer[layer_no-1]),
							np.prod(base_network.num_of_neuron_per_layer[layer_no])),
							trainable=False),
					}

					base_fc_parameter_trainable = {
						'weights': tf.get_variable(base_weight_name + '_trainable',
							shape=(np.prod(base_network.num_of_neuron_per_layer[layer_no-1]),
							np.prod(self.num_of_neuron_per_layer[layer_no])),
							initializer=tf.contrib.layers.xavier_initializer()),
					}

					base_fc_parameter = {
						#'weights': tf.concat([base_fc_parameter_untrainable['weights'], base_fc_parameter_trainable['weights']], 1,
						'weights': tf.concat([base_fc_parameter_trainable['weights'], base_fc_parameter_untrainable['weights']], 1,
								name=base_weight_name),
					}

					new_fc_parameter = {
						'weights': tf.get_variable(new_weight_name,
							shape=(np.prod(self.num_of_neuron_per_layer[layer_no-1]),
							np.prod(base_network.num_of_neuron_per_layer[layer_no])+np.prod(self.num_of_neuron_per_layer[layer_no])),
							initializer=tf.contrib.layers.xavier_initializer()),
					}

					fc_parameter = {
						#'weights': tf.concat([base_fc_parameter['weights'], new_fc_parameter['weights']], 0, name=weight_name),
						'weights': tf.concat([new_fc_parameter['weights'], base_fc_parameter['weights']], 0, name=weight_name),
						'biases' : tf.get_variable(bias_name,
							shape=(np.prod(base_network.num_of_neuron_per_layer[layer_no])+np.prod(self.num_of_neuron_per_layer[layer_no])),
							initializer=tf.contrib.layers.xavier_initializer()),

					}
					#self.extended_layers[layer_no] = [np.prod(base_network.num_of_neuron_per_layer[layer_no]) + np.prod(self.num_of_neuron_per_layer[layer_no])]

				elif layer_type[layer_no] == "output":
					base_fc_parameter_untrainable = {
						'weights': tf.get_variable(base_weight_name,
							shape=(np.prod(base_network.num_of_neuron_per_layer[layer_no-1]),
							np.prod(base_network.num_of_neuron_per_layer[layer_no])),
							trainable=False),
					}

					base_fc_parameter = {
						'weights': tf.identity(base_fc_parameter_untrainable['weights'], name=base_weight_name),
					}

					new_fc_parameter = {
						'weights': tf.get_variable(new_weight_name,
							shape=(np.prod(self.num_of_neuron_per_layer[layer_no-1]),
							np.prod(base_network.num_of_neuron_per_layer[layer_no])),
							initializer=tf.contrib.layers.xavier_initializer()),
					}

					fc_parameter = {
						#'weights': tf.concat([base_fc_parameter['weights'], new_fc_parameter['weights']], 0, name=weight_name),
						'weights': tf.concat([new_fc_parameter['weights'], base_fc_parameter['weights']], 0, name=weight_name),
						'biases' : tf.get_variable(bias_name,
							shape=(np.prod(base_network.num_of_neuron_per_layer[layer_no])),
							initializer=tf.contrib.layers.xavier_initializer()),
					}

				if layer_type[layer_no-1] == "conv":
					flattened_base = tf.reshape(base_neurons[layer_no-1], [-1, np.prod(base_network.num_of_neuron_per_layer[layer_no-1])]) 
					flattened_new = tf.reshape(new_neurons[layer_no-1], [-1, np.prod(self.num_of_neuron_per_layer[layer_no-1])]) 
					flattened = tf.concat([flattened_new, flattened_base], 1)
					neuron_drop = tf.nn.dropout(flattened, keep_prob=keep_prob)
					#self.extended_layers[layer_no-1].append(2) 
				else: 
					flattened = tf.reshape(neurons[layer_no-1],
						[-1, np.prod(base_network.num_of_neuron_per_layer[layer_no-1])+np.prod(self.num_of_neuron_per_layer[layer_no-1])]) 
					neuron_drop = tf.nn.dropout(flattened, keep_prob=keep_prob)

				if layer_type[layer_no] == "hidden":
					new_neuron = tf.nn.leaky_relu(tf.matmul(neuron_drop, fc_parameter['weights']) + fc_parameter['biases'],
					#new_neuron = tf.nn.sigmoid(tf.matmul(neuron_drop, fc_parameter['weights']) + fc_parameter['biases'],
						name=neuron_name)

				elif layer_type[layer_no] == "output":
					y_b = tf.matmul(neuron_drop, fc_parameter['weights']) + fc_parameter['biases']
					new_neuron = tf.div(tf.exp(y_b-tf.reduce_max(y_b)),
						tf.reduce_sum(tf.exp(y_b-tf.reduce_max(y_b))), name=neuron_name)

				parameters_to_regularize.append(tf.reshape(fc_parameter['weights'], [tf.size(fc_parameter['weights'])]))
		                parameters_to_regularize.append(tf.reshape(fc_parameter['biases'], [tf.size(fc_parameter['biases'])]))

				parameters[layer_no-1] = fc_parameter
				print 'fc_parameter', parameters[layer_no-1]

				neurons[layer_no] = new_neuron
				print 'new_neuron', new_neuron

		# input
		x = neurons[0]

		# output
		y = neurons[len(layers)-1]

		# correct labels
		y_ = tf.placeholder(tf.float32, [None] + layers[-1], name='y_')

		# define the loss function
		regularization = 0.000001 * tf.nn.l2_loss(tf.concat(parameters_to_regularize, 0))
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy') + regularization

		# define accuracy
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name='correct_prediction')
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

		# for training
		learning_rate = 0.001
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(cross_entropy)

		init = tf.global_variables_initializer()
		sess.run(init)

		if not os.path.exists(self.network_dir):
			os.makedirs(self.network_dir)

		# upload and freeze weights of base_network
		assign_tensor_weight = []
		assign_tensor_bias = []

		weight_vector_start_idx = 0
		weight_vector_end_idx = 0

		bias_vector_start_idx = 0
		bias_vector_end_idx = 0

		for layer in range(len(base_network.layers)-1):
			if len(base_network.layers[layer+1]) >= 4:
				tensor_weight_name = weight_base_name + str(layer) + "_base" + ":0"
				tensor_bias_name = bias_base_name + str(layer) + "_base" + ":0"
			else:
				tensor_weight_name = weight_base_name + str(layer) + "_base" + ":0"
				tensor_bias_name = bias_base_name + str(layer) + "_base" + ":0"

			weight = graph.get_tensor_by_name(tensor_weight_name) 
			weight_vector_end_idx = weight_vector_start_idx + base_network.num_of_weight_per_layer[layer]
			assign_weight = tf.assign(weight, base_network.weight_vector[weight_vector_start_idx:weight_vector_end_idx].reshape(weight.shape))
			sess.run(assign_weight)	
			weight_np = sess.run(weight)
			assign_tensor_weight.append(assign_weight)
			weight_vector_start_idx = weight_vector_end_idx

			if len(base_network.layers[layer+1]) >= 4:
				bias = graph.get_tensor_by_name(tensor_bias_name)
				bias_vector_end_idx = bias_vector_start_idx + base_network.num_of_bias_per_layer[layer]
				assign_bias = tf.assign(bias, base_network.bias_vector[bias_vector_start_idx:bias_vector_end_idx])
				sess.run(assign_bias)	
				bias_np = sess.run(bias)
				assign_tensor_bias.append(assign_bias)
				bias_vector_start_idx = bias_vector_end_idx

		sess.run([assign_tensor_weight, assign_tensor_bias])

	def buildNetwork(self, sess):
		layer_type = copy.deepcopy(self.layer_type)
		layer_type = filter(lambda type: type != 'max_pool', layer_type)
		layers = self.layers
		print 'layers', layers
		parameters = {}
		neurons = {}
		parameters_to_regularize = []

		keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')

		neurons[0] = tf.placeholder(tf.float32, [None]+layers[0], name=neuron_base_name+'0')
		print neurons[0]

		for layer_no in range(1, len(layers)):
			weight_name = weight_base_name + str(layer_no-1)
			bias_name = bias_base_name + str(layer_no-1)
			neuron_name = neuron_base_name + str(layer_no)
	
			if layer_type[layer_no] == "conv":
				conv_parameter = {
					'weights': tf.get_variable(weight_name,
						shape=(layers[layer_no]),
						initializer=tf.contrib.layers.xavier_initializer()),
					'biases' : tf.get_variable(bias_name,
						shape=(layers[layer_no][3]),
						initializer=tf.contrib.layers.xavier_initializer()),
				}

				#parameters_to_regularize.append(tf.reshape(conv_parameter['weights'], [tf.size(conv_parameter['weights'])]))
				#parameters_to_regularize.append(tf.reshape(conv_parameter['biases'], [tf.size(conv_parameter['biases'])]))

				parameters[layer_no-1] = conv_parameter
				print 'conv_parameter', parameters[layer_no-1]
				rank = sess.run(tf.rank(neurons[layer_no-1]))

				for _ in range(4 - rank):
					neurons[layer_no-1] = tf.expand_dims(neurons[layer_no-1], -1)

				# CNN
				strides = 1
				output = tf.nn.conv2d(neurons[layer_no-1],
					conv_parameter['weights'],
					strides=[1, strides, strides, 1], padding='VALID')
				output_biased = tf.nn.bias_add(output, conv_parameter['biases'])

				# max pooling
				k = 2
				new_neuron = tf.nn.max_pool(tf.nn.leaky_relu(output_biased),
				#new_neuron = tf.nn.max_pool(tf.nn.sigmoid(output_biased),
					ksize=[1, k, k, 1],
					strides=[1, k, k, 1], padding='VALID', name=neuron_name)
				neurons[layer_no] = new_neuron
				print 'new_neuron', new_neuron

			elif layer_type[layer_no] == "hidden" or layer_type[layer_no] == "output":
				fc_parameter = {
					'weights': tf.get_variable(weight_name,
        	                        	shape=(np.prod(self.num_of_neuron_per_layer[layer_no-1]),
                	                        np.prod(self.num_of_neuron_per_layer[layer_no])),
                        	                initializer=tf.contrib.layers.xavier_initializer()),
					'biases' : tf.get_variable(bias_name,
						shape=(np.prod(self.num_of_neuron_per_layer[layer_no])),
						initializer=tf.contrib.layers.xavier_initializer()),
				}

				parameters_to_regularize.append(tf.reshape(fc_parameter['weights'], [tf.size(fc_parameter['weights'])]))
	                        parameters_to_regularize.append(tf.reshape(fc_parameter['biases'], [tf.size(fc_parameter['biases'])]))

				parameters[layer_no-1] = fc_parameter
				print 'fc_parameter', parameters[layer_no-1]

				# fully-connected
				flattened = tf.reshape(neurons[layer_no-1],
					[-1, np.prod(self.num_of_neuron_per_layer[layer_no-1])]) 
				neuron_drop = tf.nn.dropout(flattened, keep_prob=keep_prob)

				if layer_type[layer_no] == "hidden":
					new_neuron = tf.nn.leaky_relu(tf.matmul(neuron_drop, fc_parameter['weights']) + fc_parameter['biases'],
					#new_neuron = tf.nn.sigmoid(tf.matmul(neuron_drop, fc_parameter['weights']) + fc_parameter['biases'],
						name=neuron_name)

				elif layer_type[layer_no] == "output":
					y_b = tf.matmul(neuron_drop, fc_parameter['weights']) + fc_parameter['biases']
					new_neuron = tf.div(tf.exp(y_b-tf.reduce_max(y_b)),
						tf.reduce_sum(tf.exp(y_b-tf.reduce_max(y_b))), name=neuron_name)

				neurons[layer_no] = new_neuron
				print 'new_neuron', new_neuron

		# input
		x = neurons[0]

		# output
		y = neurons[len(layers)-1]

		# correct labels
		y_ = tf.placeholder(tf.float32, [None] + layers[-1], name='y_')

		# define the loss function
		regularization = 0.000001 * tf.nn.l2_loss(tf.concat(parameters_to_regularize, 0))
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy') + regularization

		# define accuracy
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name='correct_prediction')
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

		# for training
		learning_rate = 0.001
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(cross_entropy)

		init = tf.global_variables_initializer()
		sess.run(init)

		if not os.path.exists(self.network_dir):
			os.makedirs(self.network_dir)
	
	def loadNetwork(self, sess):
		saver = tf.train.import_meta_graph(self.network_file_path + '.meta')
		saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(self.network_file_path)))

	def saveNetwork(self, sess):
		saver = tf.train.Saver()
		saver.save(sess, self.network_file_path)

	def next_batch(self, data_set, batch_size):
		data = data_set[0]
		label = data_set[1] # one-hot vectors

		data_num = np.random.choice(data.shape[0], size=batch_size, replace=False)
		batch = data[data_num,:]
		label = label[data_num,:] # one-hot vectors
		
		return batch, label

	def doTrain(self, sess, graph, train_set, validation_set, batch_size, train_iteration, optimizer):
		# get tensors
		tensor_x_name = "neuron_0:0"
		x = graph.get_tensor_by_name("neuron_0:0")
		y_ = graph.get_tensor_by_name("y_:0")
		keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
		keep_prob = graph.get_tensor_by_name("keep_prob:0")
		accuracy = graph.get_tensor_by_name("accuracy:0")

		input_images_validation = validation_set[0]
		input_images_validation_reshaped = np.reshape(validation_set[0], ([-1] + x.get_shape().as_list()[1:]))
		labels_validation = validation_set[1]

		time1 = time.time()
		for i in range(train_iteration):
			input_data, labels = self.next_batch(train_set, batch_size)
			input_data_reshpaed = np.reshape(input_data, ([-1] + x.get_shape().as_list()[1:]))

			if i % (100) == 0 or i == (train_iteration-1):
				train_accuracy = sess.run(accuracy, feed_dict={
					x: input_data_reshpaed, y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})
				print("step %d, training accuracy: %f" % (i, train_accuracy))
			
				# validate
				test_accuracy = sess.run(accuracy, feed_dict={
					x: input_images_validation_reshaped, y_: labels_validation,
					keep_prob_input: 1.0, keep_prob: 1.0})
				print("step %d, validation accuracy: %f" % (i, test_accuracy))

			sess.run(optimizer, feed_dict={x: input_data_reshpaed,
				y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})

		time2 = time.time()
	        print 'took %0.3f ms' % ((time2-time1)*1000.0)

	def train(self, train_set, validation_set, batch_size, train_iteration):
		print "train"
		graph = tf.Graph()
		with graph.as_default():
			with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
				self.loadNetwork(sess)
				optimizer = graph.get_operation_by_name("optimizer")
				self.doTrain(sess, graph, train_set, validation_set, batch_size, train_iteration, optimizer)
				self.saveNetwork(sess)
				self.updateParameterFromNetwork(sess, graph)

	def doInfer(self, sess, graph, data_set, label=None):
		tensor_x_name = "neuron_0:0"
		x = graph.get_tensor_by_name(tensor_x_name)
		tensor_y_name = "neuron_" + str(len(self.layers)-1) + ":0"
		y = graph.get_tensor_by_name(tensor_y_name)
		keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
		keep_prob = graph.get_tensor_by_name("keep_prob:0")

		# infer
		data_set_reshaped = np.reshape(data_set, ([-1] + x.get_shape().as_list()[1:]))
		infer_result = sess.run(y, feed_dict={
			x: data_set_reshaped, keep_prob_input: 1.0, keep_prob: 1.0})

		if label is not None:
			# validate (this is for test)
			y_ = graph.get_tensor_by_name("y_:0")
			accuracy = graph.get_tensor_by_name("accuracy:0")
			test_accuracy = sess.run(accuracy, feed_dict={
				x: data_set_reshaped, y_: label, keep_prob_input: 1.0, keep_prob: 1.0})
			print("Inference accuracy: %f" % test_accuracy)

		return infer_result

	def infer(self, data_set, label=None):
		print "infer"

		graph = tf.Graph()
		with graph.as_default():
			with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
				self.loadNetwork(sess)
				return self.doInfer(sess, graph, data_set, label)

	def updateParameterFromNetwork(self, sess, graph):
		tensor_weight = []
		tensor_bias = []

		for layer in range(len(self.layers)-1):
			tensor_weight_name = weight_base_name + str(layer) + ":0"
			weight = graph.get_tensor_by_name(tensor_weight_name)
			tensor_weight.append(weight)

			tensor_bias_name = bias_base_name + str(layer) + ":0"
			bias = graph.get_tensor_by_name(tensor_bias_name)
			tensor_bias.append(bias)

		network_weight, network_bias = sess.run([tensor_weight, tensor_bias])

		assert (len(network_weight) == len(network_bias))

		weight_vector = []
		bias_vector = []

		for i in range(len(network_weight)):
			weight_vector = np.concatenate([weight_vector, (network_weight[i].reshape((network_weight[i].size)))])
			bias_vector = np.concatenate([bias_vector, (network_bias[i].reshape((network_bias[i].size)))])

		self.weight_vector = weight_vector
		self.bias_vector = bias_vector

	def exportLayerFromNetwork(self, layer_filename):
		if self.base_network is None:
			f = open(layer_filename, 'w')
			layer_len = len(self.layers)
			f.write(struct.pack('i', layer_len))

			for i in range(layer_len):
				for j in range(len(self.layers[i])):
					f.write(struct.pack('i', self.layers[i][j]))
				f.write(struct.pack('i', 0))
			f.close()
		else:
			f = open(layer_filename, 'w')
			layer_len = len(self.layers)
			f.write(struct.pack('i', layer_len))

			for i in range(layer_len):
				baseline_layer = 0
				if len(self.layers[i]) == 1 and i != layer_len-1:
					baseline_layer = self.base_network.layers[i][0]
				for j in range(len(self.layers[i])):
					f.write(struct.pack('i', self.layers[i][j] + baseline_layer))
				f.write(struct.pack('i', 0))
			f.close()

	def exportParameterFromNetwork(self, weight_filename, bias_filename):
		graph = tf.Graph()
		with graph.as_default():
			with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
				self.loadNetwork(sess)
				tensor_weight = []
				tensor_bias = []

				for layer in range(len(self.layers)-1):
					tensor_weight_name = weight_base_name + str(layer) + ":0"
					weight = graph.get_tensor_by_name(tensor_weight_name)
					tensor_weight.append(weight)

					tensor_bias_name = bias_base_name + str(layer) + ":0"
					bias = graph.get_tensor_by_name(tensor_bias_name)
					tensor_bias.append(bias)

				network_weight, network_bias = sess.run([tensor_weight, tensor_bias])
				assert (len(network_weight) == len(network_bias))

				weight_vector = []
				bias_vector = []

				for i in range(len(network_weight)):
					if len(network_weight[i].shape) >= 4: # conv
						transposed = network_weight[i].transpose(3,2,0,1)
						weight_vector = np.concatenate([weight_vector, (transposed.reshape((network_weight[i].size)))])
					else: # fc
						transposed = network_weight[i].transpose()
						new_weight_matrix = []
						if i > 0 and len(network_weight[i-1].shape) >= 4:
							if self.base_network is not None:
								new_network_weight = network_weight[i][0:np.prod(self.num_of_neuron_per_layer[i])]
								total_k = network_weight[i-1].shape[3]
								for k in range(total_k): 
									extracted = new_network_weight[k::total_k]
									new_weight_matrix.append(extracted)

								new_network_weight = network_weight[i][np.prod(self.num_of_neuron_per_layer[i]):]
								total_k = network_weight[i-1].shape[3]
								#print total_k
								for k in range(total_k): 
									extracted = new_network_weight[k::total_k]
									new_weight_matrix.append(extracted)

								new_weight_matrix = np.vstack(new_weight_matrix).transpose()
								weight_vector = np.concatenate([weight_vector, (new_weight_matrix.reshape((new_weight_matrix.size)))])
							else:
								total_k = network_weight[i-1].shape[3]
								for k in range(total_k): 
									extracted = network_weight[i][k::total_k]
									new_weight_matrix.append(extracted)

								new_weight_matrix = np.vstack(new_weight_matrix).transpose()
								weight_vector = np.concatenate([weight_vector, (new_weight_matrix.reshape((network_weight[i].size)))])
							
						else:
							weight_vector = np.concatenate([weight_vector, (transposed.reshape((network_weight[i].size)))])

					bias_vector = np.concatenate([bias_vector, (network_bias[i].reshape((network_bias[i].size)))])

				# float
				f = open(weight_filename, 'w')
				weight_len = len(weight_vector)
				print 'weight_len', weight_len
				f.write(struct.pack('i', weight_len))

				for i in range(len(weight_vector)):
					f.write(struct.pack('f', weight_vector[i]))
				f.close()

				f = open(bias_filename, 'w')
				bias_len = len(bias_vector)
				print 'bias_len', bias_len
				f.write(struct.pack('i', bias_len))

				for i in range(len(bias_vector)):
					f.write(struct.pack('f', bias_vector[i]))
				f.close()

				# q
				f = open(weight_filename + '_q', 'w')
				weight_len = len(weight_vector)
				print 'weight_len', weight_len
				f.write(struct.pack('i', weight_len))

				for i in range(len(weight_vector)):
					weight_q = weight_vector[i] * 2**8
					q_max = (1 << 15) - 1
					q_min = -(1 << 15)
					if weight_q > q_max:
						weight_q = q_max
					elif weight_q < q_min:
						weight_q = q_min
					f.write(struct.pack('h', weight_q))
				f.close()

				f = open(bias_filename + '_q', 'w')
				bias_len = len(bias_vector)
				print 'bias_len', bias_len
				f.write(struct.pack('i', bias_len))

				for i in range(len(bias_vector)):
					bias_q = bias_vector[i] * 2**8
					q_max = (1 << 15) - 1
					q_min = -(1 << 15)
					if bias_q > q_max:
						bias_q = q_max
					elif bias_q < q_min:
						bias_q = q_min
					f.write(struct.pack('h', bias_q))
				f.close()

def main(args):
	nz = None
	if os.path.exists(NeuroZERO_filename):
		print '[] Load NeuroZERO'
		nz_file = open(NeuroZERO_filename, 'r') 
		nz = pickle.load(nz_file)
	else:
		print '[] Create a new NeuroZERO'
		nz = NeuroZERO()

	data = None
	if args.data is not None and args.data != '':
		data = __import__(args.data)

	if args.mode == 'b':
		print '[b] constructing a baseline network'

		if args.layers == None or args.layers == '':
			print '[b] No layer. Use --layers'
			return

		print '[b] layers:', args.layers
		nz.constructNetwork(nz.parse_layers(args.layers), baseline_network_name)

	elif args.mode == 't':
		print '[t] train'
		if args.network is None:
			print '[t] No network. Use --network'
			return

		if data == None:
			print '[t] No data. Use --data'
			return

		print '[t] network:', args.network
		print '[t] data:', args.data, 'train/test.size:', data.train_set()[0].shape, data.test_set()[0].shape

		batch_size = 100
		train_iteration = 5000
		nz.network_dic[args.network][1].train(data.train_set(), data.test_set(), batch_size, train_iteration)

	elif args.mode == 'i':
		print '[i] inference'
		if args.network is None:
			print '[i] No network Use --network'
			return

		if data == None:
			print '[i] No data. Use --data'
			return

		nz.network_dic[args.network][1].infer(data.test_set()[0], data.test_set()[1])
		return

	elif args.mode == 'ext':
		print '[ext] constructing a extended network'

		if nz.network_no < 2:
			print '[ext] No baseline network. Use --mode=c to create the baseline network first'

		base_network = nz.network_dic[baseline_network_name][1]
		extended_network = nz.create_extended_network(base_network.layers)
		step = len(extended_network)
		nz.constructNetworkExtended(extended_network[step-1], base_network, extended_network_name)

	elif 'e' in args.mode:
		print '[e] export weights and biases of a network to a file'
		if args.network is None:
			print '[e] No network. Use --network'
			return

		network = nz.network_dic[args.network][1]
		network.exportLayerFromNetwork(layer_base_name+args.network)
		network.exportParameterFromNetwork(weight_base_name+args.network, bias_base_name+args.network)

	if args.save != False:
		print '[] Save NeuroZERO'
		nz_file = open(NeuroZERO_filename, 'w') 
		pickle.dump(nz, nz_file)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', type=str,	help='mode', default='s')
	parser.add_argument('--layers', type=str, help='layers', default=None)
	parser.add_argument('--network', type=str, help='network', default=None)
	parser.add_argument('--data', type=str, help='data', default=None)
	parser.add_argument('--save', type=bool, help='save NeuroZERO?', default=True)

	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
