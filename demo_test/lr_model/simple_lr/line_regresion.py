from numpy import *
import logging
import copy

def loadData(file_name):
	"""
	load from csv
	"""
	data = genfromtxt(file_name, delimiter=",", usecols = (2))
	label = genfromtxt(file_name, delimiter=",", usecols = (3))
	print data
	print label
	data = mat(data).T
	label = mat(label).T
	return data, label
	

def initlize(params_num):
	"""
	initlize randomly from 0,1
	in: params_num
	return: parmaters
	"""
	weight = random.random(params_num)
	return weight


def linear_func(inZ):
	"""
	line_func
	in: inZ
	return: inZ
	"""
	return inZ

def propagation(inZ, func):
	"""
	call propagation 
	in: inZ, theta * X
	in: func, func for inZ
	return value
	"""
	return func(inZ)

def bgd(dataMatIn, classLabels, weights, learning_rate=0.0001, iter_num=200):
	"""
	batch gradient descent for linear regression
	in: dataMatIn = X (m x n)
	in: classLabels = y (n x 1)
	in: weights = theta (n x 1)
	in: learning_rate
	in: iter_num
	"""
	m, n = shape(dataMatIn)
	for iter_time in range(iter_num):
		# gradient is n vector
		try:
			gradient = dataMatIn.T * (propagation(dataMatIn * weights, linear_func) - classLabels)
			weights = weights - learning_rate * gradient
		except Exception as e:
			logging.warning("exception occur during training bgd, error info %s" % e)
			weights = zeros((n,1))
			return weights
	return weights


def normal_equation_solving(weights, dataMatrix, labelMat):
	"""
	using Matrix Algebra
	"""
	n, m = shape(weights)
	try:
		weights = ((dataMatrix.T * dataMatrix).I) * dataMatrix.T * labelMat
	except Exception as e:
		logging.warning("error reason: %s" % e)
		weights = zeros((n,1))
	return weights

def main():
    file_name = "/root/data/regression_data/iris/iris.csv"
    #dataMat,labelMat=loadDataSet()
    dataMat, labelMat = loadData(file_name)
    m,n = shape(dataMat)
    weights_1 = initlize((n,1))
    weights_1 = bgd(dataMat,labelMat, weights_1)
    print weights_1.T
    weights_2 = copy.deepcopy(weights_1)
    weights_2 = normal_equation_solving(weights_2, dataMat, labelMat)
    print weights_2.T
     
if __name__=='__main__':
    main()
