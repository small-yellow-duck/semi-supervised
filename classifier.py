"""This is meant to be a superclass 
of a bunch of classifiers that will 
be used for semi-supervised learning"""

from __future__ import division

import math
import misc
import numpy as np
import itertools as it

class classifier:
	def __init__(self):
		assert(false) #this should be over-ridden in the subclasses
	def train(self,list_of_sets_of_feats,list_of_labels, warm_start=True):
		"""This will train the classifier"""
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
	def predict_label_weights_and_confidence(self,list_of_sets_of_feats):
		pass
	def predict_label_weights(self,list_of_sets_of_feats):
		pass
	def predict_labels(self,list_of_sets_of_feats):
		pass
	def predict_confidence(self,list_of_sets_of_feats):
		pass

	def __str__(self):
		#print something useful out!