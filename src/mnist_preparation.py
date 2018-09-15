#neural network class definition
class MNISTHelper:

	def getTrainingDataFromCSV(self, path, source="mnist", format="pixel"):
		"""
		Parse an existing CSV file with MNIST training data set into a list of pixels.
		
		The input variables expect the following:
			source - the database all of our input data should be taken from
				"mnist" - the standard MNIST database with 60.000 training samples and 10.000 test samples.
				"???" - any other data source needs to be implemented separately and is not yet supported.
			format - the format in which data set is returned
				"pixel" - pixel-based representation as given in the initial database.
				"sum" - calculates the (normalized) pixel sums for each row and each column.
			path - the relative path to the resource folder
		"""
		if (format == "sum"):
			return self.convertPixelToSum()
		else:
			# returning everything else pixel-based
			# FIXME: Exception handling for input not in line with "format"
			return self.getDataFromCSV(path, source + "_train.csv")
	
	def getTestDataFromCSV(self, path, source="mnist", format="pixel"):
		"""
		Parse an existing CSV file with MNIST testing data set into a list of pixels.
		
		The input variables expect the following:
			source - the database all of our input data should be taken from
				"mnist" - the standard MNIST database with 60.000 training samples and 10.000 test samples.
				"???" - any other data source needs to be implemented separately and is not yet supported.
			format - the format in which data set is returned
				"pixel" - pixel-based representation as given in the initial database.
				"sum" - calculates the (normalized) pixel sums for each row and each column.
			path - the relative path to the resource folder
		"""
		if (format == "sum"):
			return self.convertPixelToSum()
		else:
			# returning everything else pixel-based
			# FIXME: Exception handling for input not in line with "format"
			return self.getDataFromCSV(path, source + "_test.csv")
	
	def getDataFromCSV(self, path, filename):
		"""Parse an existing CSV file with MNIST data set into a list of pixels.
		"""
		data_file = open(path + filename, 'r')
		data_list = data_file.readlines()
		data_file.close()
		return data_list
	
	def getDataFromMNIST(self, imgf, labelf, outf, network):
		# FIXME: test this code and integrate it into the tool to fix #1
		"""Get MNIST data from online repository and parse it accordingly.
		"""
		f = open(imgf, "rb")
		o = open(outf, "w")
		l = open(labelf, "rb")
		
		f.read(16)
		l.read(8)
		images = []
		
		for i in range(network):
			image = [ord(l.read(1))]
			for j in range(28*28):
				image.append(ord(f.read(1)))
			images.append(image)
		
		for image in images:
			o.write(",".join(str(pix) for pix in image)+"\network")
		f.close()
		o.close()
		l.close()

	def convertPixelToSum(self, data_list, source="mnist", normalize="true"):
		# FIXME: calculate sum of rows and columns and normalize them if required to fix #2
		pass

	def addNoiseToData(self, data_list, weight):
		# FIXME: Implement to solve #5
		pass
	
#getDataFromMNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "mnist_train.csv", 60000)
#getDataFromMNIST("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "mnist_test.csv", 10000)