"""This class is designed to manage both the labelled and 
unlabelled data for semi-supervised learning (in different instances).

Some of the features may be deleted for semi-supervised
learning, which this class will keep track of so that
at the end a classifier can be trained on all of the features
"""

from __future__ import division

import scipy as sp
import scipy.sparse as sparse
import random
import itertools as it

class data_manager:
	def __init__(self,\
		csr_train_feats,\
		train_labels___0_means_unlabelled,\
		csr_test_feats,\
		test_labels,\
		has_random_dropout=False,\
		dropout_rate=None,\
		num_dropout_corruptions_per_point=None):
		self.csr_train_feats=csr_train_feats
		self.csr_test_feats=csr_test_feats

		self.train_labels___0_means_unlabelled__minus_1_means_excluded=train_labels___0_means_unlabelled
		self.test_labels=test_labels

		self.bool_train_labelled=(self.train_labels___0_means_unlabelled__minus_1_means_excluded>0)
		self.bool_train_unlabelled=(self.train_labels___0_means_unlabelled__minus_1_means_excluded=0)
		self.bool_train_excluded=(self.train_labels___0_means_unlabelled__minus_1_means_excluded<0)

		self.bool_feat_included=(np.ones(self.csr_train_feats.shape[1])>0) #Should be all True now
		self.bool_feat_excluded=~self.bool_feat_included #Should be all False now
		self.bool_feat_given_notice=~self.bool_feat_included #Used when want to remove a feature in the near future
		self.feat_time_left=np.ones(self.csr_train_feats.shape[1])*-1 #Time left until removed

		self.has_random_dropout=has_random_dropout
		self.dropout_rate=dropout_rate
		self.num_dropout_corruptions_per_point=num_dropout_corruptions_per_point
		self.array_to_kron_with=None #Define Later if has_random_dropout
		self.dropout_matrix=None #Define Later if has_random_dropout

		def check_init():
			assert not any(self.bool_train_excluded)
			assert all(self.bool_train_labelled^self.bool_train_unlabelled)
			assert len(self.csr_train_feats)\
				== len(self.train_labels___0_means_unlabelled__minus_1_means_excluded)\
				== len(self.bool_train_labelled)\
				== len(self.bool_train_unlabelled)\
				== len(self.bool_train_excluded)
			assert len(self.csr_test_feats)\
				== len(self.test_labels)
			assert self.csr_train_feats.shape[1]\
				== self.csr_test_feats.shape[1]\
				== len(self.bool_feat_included)\
				== len(self.bool_feat_excluded)\
				== len(self.bool_feat_given_notice)\
				== len(self.feat_time_left)
			assert all(self.bool_feat_included)
			assert not any(self.bool_feat_excluded)
			assert isinstance(self.has_random_dropout,bool)
		check_init()

		def make_corrupted_copies():
			def check_dropout():
				assert dropout_rate>0 and dropout_rate<1
				assert isinstance(num_dropout_corruptions_per_point,int)
				assert num_dropout_corruptions_per_point>0
				assert self.has_random_dropout
			check_dropout()

			self.array_to_kron_with=np.ones(num_dropout_corruptions_per_point).reshape(num_dropout_corruptions_per_point,1) # a vertical ones matrix
			self.dropout_matrix=self.get_kron_matrix(self.matrix)
			row, col=self.dropout_matrix.nonzero()
			print "Before dropout the matrix has", len(row), "non-zero elements"
			assert len(row) = len(self.train_labels___0_means_unlabelled__minus_1_means_excluded)
			for r, c in it.izip(row,col):
				if random.random()<dropout_rate:
					self.dropout_matrix[r,c]=0
			self.dropout_matrix=sparse.csr_matrix(self.dropout_matrix)
			row, col=self.dropout_matrix.nonzero()
			print "After dropout the matrix has", len(row), "non-zero elements"
		if has_random_dropout: make_corrupted_copies()
	def get_kron_matrix(self, matrix):
		return sparse.kron(self.array_to_kron_with,matrix,'lil')
	def get_kron_array(self, array):
		return np.kron(self.array_to_kron_with,array)
	def label_top_n(n):
		assert not has_random_dropout
		'''TODO'''
	def get_feat_counts():
		num_unlabelled=self.csr_train_feats[self.bool_train_unlabelled].sum(axis=0)
		num_labelled=self.csr_train_feats[self.bool_train_labelled].sum(axis=0)
		return num_unlabelled, num_labelled
	def get_feats_for_semi_supervised():
		"""Get feature arrays for semi-supervised learning and for calculating performance on test_labels
		set using the same classifier (which is probably not the best classifier for using on the test set)"""

		"""Select the corrupted data if applicable, and otherwise the original training data"""
		if self.has_random_dropout:
			train_feats_dropout=self.dropout_matrix
			train_labels_dropout=self.get_kron_array(self.train_labels___0_means_unlabelled__minus_1_means_excluded)
			bool_train_labelled_dropout=self.get_kron_array(self.bool_train_labelled)
			
			train_feats_labelled=train_feats_dropout[bool_train_labelled_dropout]
			train_labels=train_labels_dropout[bool_train_labelled_dropout]
		else:
			train_feats_labelled=self.csr_train_feats[self.bool_train_labelled]
			train_labels=self.train_labels[self.bool_train_labelled]

		train_feats_unlabelled=self.csr_train_feats[self.bool_train_unlabelled] #No dropout for the unlabelled data!
		test_feats=self.test_feats
		test_labels=self.test_labels

		 """Select only those features that are still used for semi-supervised"""
		test_feats=test_feats[:,self.bool_feat_included]
		train_feats_labelled=train_feats_labelled[:,self.bool_feat_included]
		train_feats_unlabelled=train_feats_unlabelled[:,self.bool_feat_included]

		return ((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels))

	def get_feats_no_dropout():
		train_feats_labelled=self.csr_train_feats[self.bool_train_labelled]
		train_labels=self.train_labels[self.bool_train_labelled]
		train_feats_unlabelled=self.csr_train_feats[self.bool_train_unlabelled]
		test_feats=self.test_feats
		test_labels=self.test_labels
		return ((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels))
	def get_feats_only_random_dropout():