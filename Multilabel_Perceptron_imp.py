import numpy as np
import random as random



def multilabel_perceptron(X, Y_t):
	#expects Y to have values between 1 and len(Y)


	#identify set of labels that occur in Y
	labels = list(set(Y_t))
	#labels.sort()
	# Y_t = np.zeros(len(Y), 1)

	# for l in labels:
	# 	Y_t = Y_t + l*np.where(Y==labels[l])
	

	#initialize perceptron matrix
	perceptron_mat = np.zeros((len(labels), X.shape[1]))

	#randomize the rows of X
	idx = range(0, X.shape[0])
	random.shuffle(idx)


	for i in idx:

		sums = np.dot(X[i,:], perceptron_mat.T)
		#print 'sums.shape', sums.shape
		best_index = np.argmax(sums)	
		best_label = best+1  #plus 1 because labels are [1...N] not [0..N-1]
		
		correct_index = Y_t[i] - 1

		if best_label != Y_t[i]:
			#updata perceptron matrix

			perceptron_mat[best_index,:] -=  X[i,:]	
			perceptron_mat[correct_index,:] +=  X[i,:]

		#print 'iter ', i	
		#print perceptron_mat	

	#print perceptron_mat
	#print np.argmax(np.dot(X, perceptron_mat.T), axis=1) + 1

	#print np.dot(X, perceptron_mat.T)

	return perceptron_mat
