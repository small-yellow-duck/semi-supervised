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
import sys

classes_to_test={'perceptron_classifier'}

class classifier:
	def __init__(self):
		assert(false) #this should be over-ridden in the subclasses
	def train(self,X,Y, warm_start=True): #ABSTRACT CLASS!!!!! ABSTRACT CLASS!!!!!
		"""This will train the classifier"""
		sys.exit(1) #This should be implemented in the subclass
		pass
	
	"""Vectorized Versions of the previous 3 functions"""
	def predict_label_and_confidence(self,x): #ABSTRACT CLASS!!!!! ABSTRACT CLASS!!!!!
		"""This function does the work of predicting a label and
		estimating the confidence of this label prediction.

		The rest of the 'predict' functions ultimately call this."""
		sys.exit(1) #This should be implemented in the subclass
		label, confidence=None,None
		return (label, confidence)
	def predict_label(self,x):
		l,c=self.predict_label_and_confidence(x)
		return l
	def predict_confidence(self,x):
		l,c=self.predict_label_and_confidence(x)
		return c

	"""Vectorized Versions of the previous 3 functions"""
	def predict_labels_and_confidences(self,X):
		labels,confidences=[],[]
		for x in X:
			l,c=self.predict_label_and_confidence(x)			
			labels.append(l)
			confidences.append(c)
		return np.array(labels,int),np.array(confidences)
	def predict_labels(self,X):
		ls,cs=self.predict_labels_and_confidences(X)
		return ls
	def predict_confidences(self,X):
		ls,cs=self.predict_labels_and_confidences(X)
		return ls

	def __str__(self): #ABSTRACT CLASS!!!!! ABSTRACT CLASS!!!!!
		sys.exit(1) #This should be implemented in the subclass
		#print something useful out!

	def short_description(self):
		print "This should be implemented in the subclass!"
		assert False

class averaged_perceptron_classifier(classifier): 
	def __init__(self,n_iter,sample_frequency_for_averaging,verbosity=5):
		"""TODO - TODO - allow this to do averaging or not"""
		self.n_iter=n_iter #The number of passes through the data before the next update
		self.sf=sample_frequency_for_averaging #The number of points to train on between samples used for the averaged perceptron

		self.percep=None #set later
		self.averaged_perceptron=None
		self.__reset_perceptron__()
		self.verbosity=verbosity

	def __reset_perceptron__(self): #Not part of external interface
		self.percep=linear_model.Perceptron(n_iter=1,warm_start=True)
		self.averaged_perceptron=None

	def train(self,X,Y, warm_start=True):
		"""This will train the classifier"""
		assert X.shape[0]==len(Y)
		u=np.unique(Y)
		assert all(u==np.arange(1,len(u)+1)) #Y should contain labels from 1 to n with no breaks, otherwise this code might not work!
		if not warm_start: 
			self.__reset_perceptron__()
		samples=[]#[self.percep.coef_.copy()]
		num_its=int(self.n_iter*len(Y)/self.sf)
		for i in xrange(num_its):
			if self.verbosity>6: print "in averaged_perceptron_classifier.train,",i*self.sf,"of",self.n_iter*len(Y),"points trained on (with duplication)"
			ind=np.zeros(1,dtype=int) #Need to set data type to int b/c using as index!
			ui=np.unique(Y[ind])
			num_failed_samples=-1
			while len(ui)!=len(u) or any(ui!=u):
				ind=np.random.randint(0,len(Y),self.sf)
				ui=np.unique(Y[ind])
				num_failed_samples+=1
				if num_failed_samples>5: 
					print num_failed_samples, len(ui), len(u)
					print ui, u
					print any(ui!=u)
					if num_failed_samples>50: sys.exit(1)
			if self.verbosity > 8:
				print "X[ind]\n", X[ind][:3]
				print "Y[ind]\n", Y[ind]
			self.percep.fit(X[ind],Y[ind])
			samples.append(self.percep.coef_.copy())
		if self.verbosity>6:
			print "perceptron samples:"
			for s in samples: print s
		self.averaged_perceptron=sum(samples)/len(samples)
		print "averaged_perceptron:\n",self.averaged_perceptron


	def predict_label_and_confidence(self,x): 
		"""This function does the work of predicting a label and
		estimating the confidence of this label prediction.

		The rest of the 'predict' functions ultimately call this."""
		scores=(x*self.averaged_perceptron.transpose())[0]
		ind=scores.argsort()
		label = ind[-1]+1
		confidence = scores[ind[-1]] - scores[ind[-2]]
		return (label, confidence)

	def __str__(self):
		my_str="num pts to train on between samples to use in average ="+str(self.sf)+\
			",  number of passes through the data =" + str(self.n_iter)+\
			",  percep values =\n"+str(self.percep.coef_)+\
			",  Ave percep values =\n"+str(self.averaged_perceptron)
		return my_str

		#print something useful out!

if __name__=="__main__" and 'averaged_perceptron_classifier' in classes_to_test:
	
	(train_X,train_Y),(test_X,test_Y)=\
		misc.create_synthetic_data(	num_labels=4,\
									num_train=20,\
									num_feats=10,\
									frac_labelled=1,\
									num_test=20,\
									sparsity=2,\
									skew=2,\
									rand_seed=0)
	#print (train_X,train_Y),(test_X,test_Y)
	p=averaged_perceptron_classifier(1,5)
	p.train(train_X,train_Y)
	# p.predict_label_and_confidence(train_X[0])
	# p.predict_label_and_confidence(train_X[1])
	# p.predict_label_and_confidence(train_X[2])
	labels, confidences = p.predict_labels_and_confidences(train_X)
	lab_theirs=p.percep.predict(train_X)
	print "Train averaged perceptron: {:.2%} correct".format(sum(labels==train_Y)/len(labels))
	print "Train perceptron: {:.2%}={:.2%} correct".format(p.percep.score(train_X, train_Y),sum(lab_theirs==train_Y)/len(labels))
	print "np.column_stack((labels, confidences, train_Y, correct?, lab_theirs, correct?, ours and theirs same?)):\n",\
		np.column_stack((labels, confidences, train_Y, labels==train_Y,lab_theirs,lab_theirs==train_Y,lab_theirs==labels))[:50]

	labels, confidences = p.predict_labels_and_confidences(test_X)
	lab_theirs=p.percep.predict(test_X)
	print "Test: {:.2%} correct".format(sum(labels==test_Y)/len(labels))
	print "np.column_stack((labels, confidences, test_Y, correct?, lab_theirs, correct?, ours and theirs same?)):\n",\
		np.column_stack((labels, confidences, test_Y, labels==test_Y,lab_theirs,lab_theirs==test_Y,lab_theirs==labels))[:50]


	labels, confidences = p.predict_labels_and_confidences(train_X)
	print "Train averaged perceptron: {:.2%} correct".format(sum(labels==train_Y)/len(labels))
	print "Train perceptron: {:.2%} correct".format(p.percep.score(train_X, train_Y))
	print p
	p.averaged_perceptron=p.percep.coef_.copy()
	labels, confidences = p.predict_labels_and_confidences(train_X)
	print "Train perceptron from my class: {:.2%} correct".format(sum(labels==train_Y)/len(labels))
	print p

class perceptron_classifier(classifier): 
	def __init__(self,n_iter,verbosity=5):
		"""TODO - TODO - allow this to do averaging or not"""
		self.n_iter=n_iter #The number of passes through the data before the next update
		self.verbosity=verbosity
		assert verbosity in range(11)

		self.percep=None #set later
		self.__reset_perceptron__()

	def __reset_perceptron__(self): #Not part of external interface
		self.percep=linear_model.Perceptron(n_iter=self.n_iter,warm_start=True)

	def train(self,X,Y, warm_start=True):
		"""This will train the classifier"""
		assert X.shape[0]==len(Y)
		u=np.unique(Y)
		assert all(u==np.arange(1,len(u)+1)) #Y should contain labels from 1 to n with no breaks, otherwise this code might not work!
		if not warm_start: 
			self.__reset_perceptron__()
		self.percep.fit(X,Y)

	def predict_label_and_confidence(self,x): 
		"""This function predicts a label and estimates the 
		confidence of this label prediction."""
		scores=self.percep.decision_function(x)
		assert scores.shape[0]==1
		scores=scores[0]
		ind=scores.argsort()
		label = ind[-1]+1
		confidence = scores[ind[-1]] - scores[ind[-2]]
		def print_stuff():
			print "x", x
			print "score", scores
			print "ind", ind
			print "label", label		
			print "confidence",confidence
			print "self.percep.predict(x)",self.percep.predict(x)

		if self.verbosity>9: print_stuff()
		if label != self.percep.predict(x):
			if confidence>0 or self.verbosity>8:
				print "MY PREDICTION DIFFERS FROM THE PERCEPTRONS"
				print_stuff()
			if confidence>0:
				print "CONFIDENCE GREATER THAN 0, SO THIS SHOULDN'T HAPPEN - EXITING"
				sys.exit(1)

		return (label, confidence)

	def predict_labels_and_confidences(self,X):
		"""TODO"""		
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
		labels=self.percep.predict(X)
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



