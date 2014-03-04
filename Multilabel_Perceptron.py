rom __future__ import division

import random
import math
import misc
import numpy as np
import itertools as it
#import sys
import sklearn as skl
from sklearn import linear_model
import sys
import random as random


import Classifier

class Perceptron_Multilabel_Classifier(Classifier): 
	def __init__(self,n_iter,verbosity=5):
		self.n_iter=n_iter #The number of passes through the data before the next update
		self.verbosity=verbosity
		assert verbosity in range(11)

		#perceptron matrix	
		self.percep=None #set later

		self.labels = None
		self.best_label = None
		self.best_index = None
		self.correct_label = None

		self.__reset_perceptron__()

		self.initialized = False



	def __reset_perceptron__(self, numlabels, numsamples): #Not part of external interface
		#self.percep=linear_model.Perceptron(n_iter=self.n_iter,warm_start=True)
		self.percep = np.zeros((numlabels, numsamples))



	def train(self,X,Y, warm_start=True):
		"""This will train the classifier"""
		assert X.shape[0]==len(Y)
		labels=np.unique(Y)
		assert all(labels==np.arange(1,len(labels)+1)) #Y should contain labels from 1 to n with no breaks, otherwise this code might not work!
		
		

		if not self.trained:
			percep = np.zeros((len(labels), X.shape[1]))
			self.initialized = True		

		if not warm_start: 
			self.__reset_perceptron__(labels, X.shape[1])

		assert percep.shape[0] == len(labels)	
		assert percep.shape[1] == X.shape[1]
		

		#randomize the rows of X
		idx = range(0, X.shape[0])
		random.shuffle(idx)


		for i in idx:

			sums = np.dot(X[i,:], percep.T)
			#print 'sums.shape', sums.shape
			best_index = np.argmax(sums)	
			best_label = best_index+1  #plus 1 because labels are [1...N] not [0..N-1]
			
			correct_index = Y[i] - 1

			if best_label != Y[i]:
				#updata perceptron matrix
				percep[best_index,:] -=  X[i,:]	
				percep[correct_index,:] +=  X[i,:]






	def predict_labels_and_confidences(self,X):
		n=X.shape[0]
		scores=self.percep.decision_function(X)
		assert scores.shape[0]==n
		# print "scores", scores
		ind1=scores.argmax(axis=1).astype(np.int32)
		labels = ind1+1
		c1=scores[np.arange(n),ind1]
		scores[np.arange(n),ind1]=-100
		ind2=scores.argmax(axis=1).astype(np.int32)
		c2=scores[np.arange(n),ind2]
		confidence = c1-c2
		def print_stuff():
			# print "X", X
			# print "scores", scores
			print "ind1", ind1
			print "ind2", ind2
			print "labels", labels		
			print "confidence",confidence
			print "self.percep.predict(X)",self.percep.predict(X)

		if self.verbosity>6: print_stuff()
		assert all(labels == self.percep.predict(X))
		return (labels, confidence)



	def predict_labels(self,X):
		labels = np.argmax(np.dot(X[:,:], percep.T), axis=1) +1

		return labels



	def __str__(self):
		my_str=",  number of passes through the data =" + str(self.n_iter)+\
			"\n,  percep values =\n"+str(self.percep.coef_)
		return my_str


	def short_description(self):
		return "per"+str(self.n_iter)




if __name__=="__main__" and 'perceptron_classifier' in classes_to_test:

	(train_X,train_Y),(test_X,test_Y)=\
		misc.create_synthetic_data(	num_labels=4,\
									num_train=100,\
									num_feats=10,\
									frac_labelled=1,\
									num_test=50,\
									sparsity=2,\
									skew=2,\
									rand_seed=0)
	print '-'*50
	p=perceptron_classifier(5)
	p.train(train_X,train_Y)
	labels, confidences = p.predict_labels_and_confidences(train_X)
	lab_theirs=p.percep.predict(train_X)
	print "Train perceptron - my class: {:.2%} correct".format(sum(labels==train_Y)/len(labels))
	print "Train perceptron - sklearn: {:.2%}={:.2%} correct".format(p.percep.score(train_X, train_Y),sum(lab_theirs==train_Y)/len(labels))
	print "np.column_stack((labels, confidences, train_Y, correct?, lab_theirs, correct?, ours and theirs same?)):\n",\
		np.column_stack((labels, confidences, train_Y, labels==train_Y,lab_theirs,lab_theirs==train_Y,lab_theirs==labels))[:10]

	labels, confidences = p.predict_labels_and_confidences(test_X)
	lab_theirs=p.percep.predict(test_X)
	print "Test: {:.2%} correct".format(sum(labels==test_Y)/len(labels))
	print "np.column_stack((labels, confidences, test_Y, correct?, lab_theirs, correct?, ours and theirs same?)):\n",\
		np.column_stack((labels, confidences, test_Y, labels==test_Y,lab_theirs,lab_theirs==test_Y,lab_theirs==labels))[:10]


	labels, confidences = p.predict_labels_and_confidences(train_X)
	print "Train averaged perceptron: {:.2%} correct".format(sum(labels==train_Y)/len(labels))
	print "Train perceptron: {:.2%} correct".format(p.percep.score(train_X, train_Y))
	print p

