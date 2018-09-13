'''
Created on 13.09.2018

@author: Roland
'''
import numpy
#scipy.special for sigmoid function expit()
import scipy.special
#libary for plotting arrays
import matplotlib.pyplot
import time
#ensure the plots are inside this notebook, not an external window
#matplotlib inline

#neural network class definition
class neuralNetwork:
	
	#initialise the neural network
	def __init__(self ,inputnodes, hidden1nodes, hidden2nodes, outputnodes, learningrate):
		#set number of nodes in each input, hidden1, hidden2, output layer
		self.inodes = inputnodes
		self.h1nodes = hidden1nodes
		self.h2nodes = hidden2nodes
		self.onodes = outputnodes
		# link weights matrices, wih (from input to hidden) and who (from hidden to output)
		# weights inside the arrays are w_i_j, where link is from node i to j in the next layer
		# w11 w21
		# w12 w22 etc
		self.wih1 = numpy.random.normal(0.0, pow(self.h1nodes, -0,5), (self.h1nodes, self.inodes))
		self.wh1h2= numpy.random.normal(0.0, pow(self.h2nodes, -0.5), (self.h2nodes, self.h1nodes))
		self.wh2o = numpy.random.normal(0.0, pow(self.onodes, -0,5), (self.onodes, self.h2nodes))
		#learning rate
		self.lr = learningrate
		#activation is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)
		pass
	
	#train the neural network
	def train(self, input_list, targets_list):
		#convert input list to 2d array
		inputs = numpy.array(input_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T
		#calculate signals into hidden layer
		hidden1_inputs = numpy.dot(self.wih1, inputs)
		#calculate the signals emerging from hidden layer
		hidden1_outputs = self.activation_function(hidden1_inputs)
		#calculate signals into final output layer
		hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
		#calculate the signals emerging from hidden layer
		hidden2_outputs = self.activation_function(hidden2_inputs)
		#calculate signals into final output layer
		final_inputs = numpy.dot(self.wh2o, hidden2_outputs)
		#calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)
		#output layer error is the (target - actual)
		output_errors = targets - final_outputs
		#hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden2_errors = numpy.dot(self.wh2o.T, output_errors)
		#hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden1_errors = numpy.dot(self.wh1h2.T, hidden2_errors)		
		#update the weights for the links between the hidden and the output layers
		self.wh2o += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
		numpy.transpose(hidden2_outputs))
		#update the weights for the links between the input and hidden layer
		self.wh1h2 += self.lr * numpy.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)),
		numpy.transpose(hidden1_outputs))
		#update the weights for the links between the input and hidden layer
		self.wih1 += self.lr * numpy.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)),
		numpy.transpose(inputs))
		pass

	#query the neural network
	def query(self, input_list):
		#convert input list to 2d array
		inputs = numpy.array(input_list, ndmin=2).T
		#calculate signals into hidden layer
		hidden1_inputs = numpy.dot(self.wih1, inputs)
		#calculate the signals emerging the hidden layer
		hidden1_outputs = self.activation_function(hidden1_inputs)
		#calculate signals into hidden layer
		hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
		#calculate the signals emerging the hidden layer
		hidden2_outputs = self.activation_function(hidden2_inputs)
		#calculate signals into final output layer
		final_inputs = numpy.dot(self.wh2o, hidden2_outputs)
		#calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)
		return final_outputs
		pass

# number of input, hidden and output nodes
input_nodes = 784
hidden1_nodes = 200
hidden2_nodes = 50
output_nodes = 10
# learning rate
learning_rate = 0.1

#create instance of neural network
n = neuralNetwork(input_nodes,hidden1_nodes, hidden2_nodes, output_nodes, learning_rate)

#load the mist training data csv file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
#train the neural network
# epochs is the number of times the training data set is used for training
epochs = 6
#setting the full time to zero
gesamtzeit=0
for e in range(epochs):
	#go through all records in the training data set
	print("Start Epoche {0:1d} von {1:1d} " .format(e+1,epochs))
	#get the starting point for time calculation
	start = time.time()
	for record in training_data_list:
		#split the record by the ',' commas
		all_values = record.split(',')
		#scale and shift the inputs
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		#create the target output values (all 0.01, except the desired label which is 0.99)
		targets = numpy.zeros(output_nodes) + 0.01
		#all values[0] is the target label for this record
		targets[int(all_values[0])] = 0.99
		n.train(inputs, targets)
		pass
	#getting the endpoint of time
	ende = time.time()
	#calculate the full time
	gesamtzeit = gesamtzeit +(ende - start)
	print("Dauer des Trainingsdurchlaufes {:5.1f}s" .format(ende-start))
	pass
print("Die Gesamtdauer betrug {0:5.1f}s bzw. {1:1.1f}min" .format(gesamtzeit, gesamtzeit/60))
#load the mnist test data csv file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#test the neural network
#scorecard for how well the network performs. initially empty
scorecard = []
#go through all the records in the test data set
for record in test_data_list:
	#split the record by the ',' commas
	all_values = record.split(',')
	#correct answer is first value
	correct_label = int(all_values[0])
	#scale and shift the inputs
	inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	#query the network
	outputs = n.query(inputs)
	#the index of the highest value corresponds to the label
	label = numpy.argmax(outputs)
	#append correct or incorrect to list
	if (label == correct_label):
		#networks answer matches correct answer, add 1 to scorecard
		scorecard.append(1)
	else:
		#networks answer doesn't match correct answer, add 0 to scorecard
		scorecard.append(0)
		pass
	pass

#calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("Testmenge = ", scorecard_array.size)
print ("korrekt erkannt:  ", scorecard_array.sum() )
print ("performance = ", scorecard_array.sum() / scorecard_array.size)