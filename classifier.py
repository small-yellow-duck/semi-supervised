"""This is meant to be a superclass 
of a bunch of classifiers that will 
be used for semi-supervised learning"""

from __future__ import division

import math
import misc
import numpy as np
import itertools as it
#import sys
import sklearn as skl
from sklearn import linear_model

classes_to_test={'averaged_perceptron_classifier'}

class classifier:
	def __init__(self):
		assert(false) #this should be over-ridden in the subclasses
	def train(self,X,Y, warm_start=True):
		"""This will train the classifier"""
		sys.exit(1) #This should be implemented in the subclass
		pass

	# def predict_label_weights_and_confidence(self,set_of_feats):
	# 	pass
	# def predict_label_weights(self,set_of_feats):
	# 	pass
	# def predict_label(self,set_of_feats):
	# 	pass
	# def predict_confidence(self,set_of_feats):
	# 	pass

	"""Vectorized Versions of the previous 3 functions"""
	def predict_label_weights_and_confidence(self,X):
		sys.exit(1) #This should be implemented in the subclass
		pass
	def predict_label_weights(self,X):
		sys.exit(1) #This should be implemented in the subclass
		pass
	def predict_labels(self,X):
		sys.exit(1) #This should be implemented in the subclass
		pass
	def predict_confidence(self,X):
		sys.exit(1) #This should be implemented in the subclass
		pass

	def __str__(self):
		pass
		#print something useful out!

class averaged_perceptron_classifier(classifier):
	def __init__(self,sample_frequency_for_averaging,n_iter):
		self.sf=sample_frequency_for_averaging #The number of points to train on between samples used for the averaged perceptron
		self.n_iter=n_iter #The number of passes through the data before the next update
		self.percep=None #set later
		self.averaged_perceptron=None
		self.reset_perceptron()
	def reset_perceptron(self):
		self.percep=linear_model.Perceptron(n_iter=1,warm_start=True)
		self.averaged_perceptron=None
	def train(self,X,Y, warm_start=True):
		"""This will train the classifier"""
		assert X.shape[0]==len(Y)
		if not warm_start: 
			self.reset_perceptron
		samples=[self.percep.get_params()]
		num_its=int(n_iter*len(Y)/self.sf)
		for i in xrange(num_its):
			self.percep.fit(X,Y)
			samples.append(self.percep.get_params())
		self.averaged_perceptron=sum(samples)/len(samples)


	def predict_label_weights_and_confidence(self,X):
		sys.exit(1) #This should be implemented in the subclass
		pass
	def predict_label_weights(self,X):
		sys.exit(1) #This should be implemented in the subclass
		pass
	def predict_labels(self,X):
		sys.exit(1) #This should be implemented in the subclass
		pass
	def predict_confidence(self,X):
		sys.exit(1) #This should be implemented in the subclass
		pass

	def __str__(self):
		pass
		#print something useful out!

if __name__=="__main__" and 'averaged_perceptron_classifier' in classes_to_test:
	(train_X,train_Y),(test_X,test_Y)=data_manager.create_synthetic_data(num_labels=4,\
									num_train=20,\
									num_feats=10,\
									frac_labelled=.3,\
									num_test=5)
	print (train_X,train_Y),(test_X,test_Y)

