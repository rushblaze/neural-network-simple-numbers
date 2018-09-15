'''
Out of a book, please add the correct title and maybe page.

@author: ??? ???
'''

import numpy
# scipy.special for sigmoid function expit()
import scipy.special

# neural network class definition
class NeuralNetwork:
	
	# initialise the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# set number of nodes in each input, hidden, output layer
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		# link weights matrices, wih (from input to hidden) and who (from hidden to output)
		# weights inside the arrays are w_i_j, where link is from node i to j in the next layer
		# w11 w21
		# w12 w22 etc
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0,5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0,5), (self.onodes, self.hnodes))
		# learning rate
		self.lr = learningrate
		# activation is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)
		pass

	# train the neural network
	def train(self, input_list, targets_list):
		# convert input list to 2d array
		inputs = numpy.array(input_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T
		# calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)
		# calculate signals into final output layer
		final_inputs = numpy.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)
		# output layer error is the (target - actual)
		output_errors = targets - final_outputs
		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = numpy.dot(self.who.T, output_errors)
		# update the weights for the links between the hidden and the output layers
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
		numpy.transpose(hidden_outputs))
		# update the weights for the links between the input and hidden layer
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
		numpy.transpose(inputs))
		pass

	# query the neural network
	def query(self, input_list):
		# convert input list to 2d array
		inputs = numpy.array(input_list, ndmin=2).T
		# calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs)
		# calculate the signals emerging the hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)
		# calculate signals into final output layer
		final_inputs = numpy.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)
		return final_outputs
		pass
	
	# FIXME: export all relevant parameter of the given network to fix #3
	def export(self):
		pass