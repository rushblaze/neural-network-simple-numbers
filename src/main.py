'''
Created on 15.09.2018

@author: rushblaze
'''
# library for number operations
import numpy
# library for runtime time measurements
import time

# class representing a neural network with only one hidden layer
import NN_1h
# helper class for all MNIST relevant tasks
import mnist_preparation

# path to your resource folder, where pre-created csv files of data set reside
path_to_resources = "../res/"

# parameters to configure the neural network
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

# epochs is the number of times the training data set is used for training
epochs = 6

# initialization of the network and data sets
network = NN_1h.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
training_data_list = mnist_preparation.MNISTHelper().getTrainingDataFromCSV(path=path_to_resources)
test_data_list = mnist_preparation.MNISTHelper().getTestDataFromCSV(path=path_to_resources)

# train the neural network
# setting the full time to zero
duration = 0
#TODO: we might want to put this into the neural network as well.
for e in range(epochs):
	# go through all records in the training data set
	print("Start Epoche {0:1d} von {1:1d} " .format(e+1, epochs))
	# get the starting point for time calculation
	start = time.time()
	for record in training_data_list:
		# split the record by the ',' commas
		all_values = record.split(',')
		# scale and shift the inputs
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		# create the target output values (all 0.01, except the desired label which is 0.99)
		targets = numpy.zeros(output_nodes) + 0.01
		# all values[0] is the target label for this record
		targets[int(all_values[0])] = 0.99
		network.train(inputs, targets)
		pass
	# getting the endpoint of time
	stop = time.time()
	# calculate the full time
	duration = duration + (stop - start)
	print("Dauer des Trainingsdurchlaufes {:5.1f}s" .format(stop-start))
	pass
print("Die Gesamtdauer betrug {0:5.1f}s bzw. {1:1.1f}min" .format(duration, duration/60))

# test the neural network
# scorecard for how well the network performs. initially empty
scorecard = []
# go through all the records in the test data set
for record in test_data_list:
	# split the record by the ',' commas
	all_values = record.split(',')
	# correct answer is first value
	correct_label = int(all_values[0])
	# scale and shift the inputs
	inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	# query the network
	outputs = network.query(inputs)
	# the index of the highest value corresponds to the label
	label = numpy.argmax(outputs)
	# append correct or incorrect to list
	if (label == correct_label):
		# networks answer matches correct answer, add 1 to scorecard
		scorecard.append(1)
	else:
		# networks answer doesn't match correct answer, add 0 to scorecard
		scorecard.append(0)
		pass
	pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("Testmenge = ", scorecard_array.size)
print ("korrekt erkannt:  ", scorecard_array.sum() )
print ("performance = ", scorecard_array.sum() / scorecard_array.size)