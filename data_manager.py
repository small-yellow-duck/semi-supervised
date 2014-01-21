"""This class is designed to manage both the labelled and 
unlabelled data for semi-supervised learning (in different instances).

Some of the features may be deleted for semi-supervised
learning, which this class will keep track of so that
at the end a classifier can be trained on all of the features
"""

import scipy as sp
import scipy.sparse as sparse
import random
import itertools as it

class data_manager:
	def __init__(self,\
		csr_features_array,\
		labels_array=None,\
		max_training_examples=None,\
		has_dropout=False,\
		dropout_rate=None,\
		num_dropout_corruptions_per_point=None):
		self.num_points=len(csr_features_array)
		self.num_feats=csr_features_array.shape[1]

		"""self.num_rows can be greater than self.num_points, 
		leaving room for more points to be added later"""
		if max_training_examples==None: 
			self.num_rows=self.num_points
		else
			assert max_training_examples>=self.num_points
			self.num_rows=max_training_examples #Some rows may not be used yet
	
		self.matrix=sparse.csr_matrix(self.num_rows, self.num_feats)
		self.matrix[:self.num_points]=csr_features_array #make a copy of the features array
		self.has_labels = False
		if labels_array != None:
			self.has_labels = False
			assert self.num_points==len(labels_array)
			self.labels=np.zeros(self.num_rows)
			self.labels[:self.num_points]=labels_array #make a copy of the labels array

		self.active_points=np.arange(self.num_rows)<num_points #Used for boolean indexing
		self.active_features=np.ones(self.num_feats)=1 #Use all features to start with.

		"""TODO: MAKE IT SO THAT THERE IS ONLY 1 DATA MANAGER THAT KEEPS TRACK OF WHAT IS 
		LABELLED AND UNLABELLED, INCLUDING THE CORRESPONDENCE WITH THE CORRUPTED VERSIONS"""

		def make_corrupted_copies():
			assert dropout_rate>0 and dropout_rate<1
			assert isinstance(num_dropout_corruptions_per_point,int)
			self.has_dropout=has_dropout
			self.dropout_rate=dropout_rate
			self.num_dropout_corruptions_per_point=num_dropout_corruptions_per_point
			vertical_ones_matrix=np.ones(num_dropout_corruptions_per_point).reshape(num_dropout_corruptions_per_point,1)
			self.dropout_matrix=sparse.kron(vertical_ones_matrix,self.matrix,'lil')
			if self.has_labels:
				self.dropout_labels=np.kron(vertical_ones_matrix,self.labels)
			row,col=self.dropout_matrix.nonzero()
			for r,c in it.izip(row,col):

			i=0
			for row_num in xrange(self.num_points):
				for 
			'''TODO: convert lil_matrix to csr_matrix at end!'''
		if has_dropout: make_corrupted_copies()



	def pop_top_n(n):
		assert not has_dropout
		'''TODO'''
	def get_feat_counts():
	def get_feats_with_dropout():
	def get_feats_no_dropout():
