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
		self.initialize_all_attribute_variables()
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
				== len(self.feat_time_left)
			assert all(self.bool_feat_included)
			assert not any(self.bool_feat_excluded)
			assert isinstance(self.has_random_dropout,bool)
			assert set(self.test_labels)-{0}<self.set_labels
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

	def initialize_all_attribute_variables():
		self.bool_verbose=False

		self.csr_train_feats=csr_train_feats
		self.csr_test_feats=csr_test_feats

		self.train_labels___0_means_unlabelled__minus_1_means_excluded=train_labels___0_means_unlabelled
		self.set_labels=set(train_labels___0_means_unlabelled)-{}
		self.test_labels=test_labels

		self.bool_train_labelled=(self.train_labels___0_means_unlabelled__minus_1_means_excluded>0)
		self.bool_train_unlabelled=(self.train_labels___0_means_unlabelled__minus_1_means_excluded=0)
		self.bool_train_excluded=(self.train_labels___0_means_unlabelled__minus_1_means_excluded<0)
		self.bool_train_labelled_initially=self.bool_train_labelled #This shouldn't change!

		self.bool_feat_included=(np.ones(self.csr_train_feats.shape[1])>0) #Should be all True now
		self.bool_feat_excluded=~self.bool_feat_included #Should be all False now
		self.feat_time_left=np.ones(self.csr_train_feats.shape[1])*-1 #Time left until removed

		self.has_random_dropout=has_random_dropout
		self.dropout_rate=dropout_rate
		self.num_dropout_corruptions_per_point=num_dropout_corruptions_per_point
		self.array_to_kron_with=None #Define Later if has_random_dropout
		self.dropout_matrix=None #Define Later if has_random_dropout

	def set_verbose(self,bool_verbose):
		assert isinstance(bool_verbose,bool)
		self.bool_verbose=bool_verbose

	def get_kron_matrix(self, matrix):
		return sparse.kron(self.array_to_kron_with,matrix,'lil')
	def get_kron_array(self, array):
		return np.kron(self.array_to_kron_with,array)

"""Functions to label unlabelled training data and move it to the labelled data"""
	def forget_labels(labels_to_forget="none"): """TODO"""
		"""labels_to_forget = a string specifying which (if any) labels to forget prior to labelling some unlabelled 
							training data.  Forgetting allows the semi-supervised algorithm to "change it's mind".
							To make progress, be sure to label more unlabelled data than you are forgetting!
							There are 3 acceptable values:
			"none" - Forget no labels
			"originally unlabelled" - Forget all labels that weren't given to the data manager's constructor
			"all" - Forget all labels, so you could even change your mind about the initially labelled examples
		"""
	def label_top_n(self,n,classifier,labels_to_forget="none"):
		"""labels some unlabelled training data

			n = an integer or a dictionary.  
				if n is an integer then the top n most confident points are added
				if n is a dictionary, it should contain all the keys in self.set_labels
			classifier = an object with a method 
				def get_labels_and_confidence(self,csr_train_feats)
					...
					return labels, confidences  <- These are numpy arrays
			labels_to_forget = a string specifying which (if any) labels to forget prior to labelling some unlabelled 
								training data.  Forgetting allows the semi-supervised algorithm to "change it's mind".
								To make progress, be sure to label more unlabelled data than you are forgetting!
								There are 3 acceptable values:
				"none" - Forget no labels
				"originally unlabelled" - Forget all labels that weren't given to the data manager's constructor
				"all" - Forget all labels, so you could even change your mind about the initially labelled examples
			"""
		labels,confidences=classifier.get_labels_and_confidence(self.csr_train_feats)
		confidences[self.bool_train_labelled]=0 #So we don't try to re-label these!
		def label_top_k(k,labels,confidences):
			bool_indices_to_label=confidences.argsort()[-k:]
			self.train_labels___0_means_unlabelled__minus_1_means_excluded[bool_indices_to_label]=labels[bool_indices_to_label]
			assert not any(self.bool_train_labelled[bool_indices_to_label])
			assert all(self.bool_train_unlabelled[bool_indices_to_label])
			self.bool_train_labelled[bool_indices_to_label]=True
			self.bool_train_unlabelled[bool_indices_to_label]=False
		if isinstance(n,int):
			label_top_k(n,labels,confidences)
		elif isinstance(n,dict):
			assert n.keys()=self.set_labels
			for label, k in n.iteritems():
				conf_of_label=confidences*(labels==label)
				assert set(labels[conf_of_label!=0])=={label}
				label_top_k(k,labels,conf_of_label)
		else:
			raise TypeError("Unexpected Type passed to label_top_n.  First method argument should be an int or a dictionary")
	def label_top_n_of_each_type(self,n,classifier,labels_to_forget="none"):
		assert isinstance(n,int)
		N={l:n for l in self.set_labels}
		self.label_top_n(N,classifier,labels_to_forget)

"""Functions for accessing the data"""
	def get_feats_for_semi_supervised(self,percent_dropout,num_dropout_corruptions_per_point):
		"""Get feature arrays for semi-supervised learning and for calculating performance on test_labels
		set using the same classifier (which is probably not the best classifier for using on the test set)"""

		"""Select the corrupted data if applicable, and otherwise the original training data"""
		((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels))\
			=self.get_feats_only_random_dropout()

		 """Select only those features that are still used for semi-supervised"""
		test_feats=test_feats[:,self.bool_feat_included]
		train_feats_labelled=train_feats_labelled[:,self.bool_feat_included]
		train_feats_unlabelled=train_feats_unlabelled[:,self.bool_feat_included]

		return ((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels)) """TODO UPDATE"""
	def get_feats_no_dropout(self):
		train_feats_labelled=self.csr_train_feats[self.bool_train_labelled]
		train_labels=self.train_labels[self.bool_train_labelled]
		train_feats_unlabelled=self.csr_train_feats[self.bool_train_unlabelled]
		test_feats=self.test_feats
		test_labels=self.test_labels
		return ((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels))
	def get_feats_only_random_dropout(self):		
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

		return ((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels))

"""Functions for removing features"""
	def get_feat_counts(self):
		num_unlabelled=self.csr_train_feats[self.bool_train_unlabelled].sum(axis=0)
		num_labelled=self.csr_train_feats[self.bool_train_labelled].sum(axis=0)
		return num_unlabelled, num_labelled
	def give_notice(self,imbalance_threshold, notice):
		"""For each feature, 
			Let 
				Fl = the fraction of examples in the labelled training
						data containing that feature
				Fu = the fraction of examples in the unlabelled training
						data containing that feature
			if Fl/Fu > imbalance_threshold, then the feature is "given notice"
			that it will be removed soon (after decrement_notice() is called
			'notice' many times).  When it is removed, all unlabelled examples
			containing the feature are also removed.
			if Fu=0, then the feature is removed immediately

			The idea is that once the imbalance becomes large the feature may 
			still be useful to label the last few unlabelled examples, so we
			give it a little bit of time to do so, but then it is removed so
			that the classifier focuses on using features that generalize better.
			"""
		assert imbalance_threshold>=1
		if notice==None: notice=0
		assert isinstance(notice,int)
		assert notice>=0

		train_unlabelled_feats, train_labelled_feats=self.get_feat_counts()

		"""Remove features not present in the unlabelled training data immediately"""
		bool_feats_to_remove=(train_unlabelled_feats==0)&self.bool_feat_included&(self.feat_time_left==-1)
		self.remove_feats(bool_feats_to_remove)

		"""Give features present but rare in the unlabelled training data notice of removal in the near future"""
		bool_feats_to_give_notice=train_labelled_feats/len(train_labelled_feats)>\
							train_unlabelled_feats/len(train_unlabelled_feats)*imbalance_threshold
		bool_feats_to_give_notice=bool_feats_to_give_notice&self.bool_feat_included&(self.feat_time_left==-1)
		
		if notice==0:
			self.remove_feats(bool_feats_to_give_notice)
		else:
			self.feat_time_left[bool_feats_to_give_notice]=notice

		if self.bool_verbose:
			print '-'*10,'give_notice','-'*10
			print "NOTICE:",notice,"given to",len(bool_feats_to_give_notice),"new feats.",\
				"Tot",(self.feat_time_left>0).sum(),"feats on notice,",self.bool_feat_excluded.sum(),
				"feats removed, and", self.bool_feat_included.sum(),"feats included."
	def decrement_notice(self):
		"""decrements self.feat_time_left and removes feats whose time has run out."""
		assert not any(self.feat_time_left==0)
		self.feat_time_left[self.feat_time_left>0]-=1
		bool_feats_to_remove=(self.feat_time_left==0)
		self.remove_feats(bool_feats_to_remove)
	def remove_feats(self,bool_indices):		
		"""Removes feats specified by the given boolean indices

		all unlabelled training examples containing removed feats are also removed"""
		self.feat_time_left[bool_feats_to_remove]=-2
		assert not any(self.bool_feat_excluded[bool_feats_to_remove])
		assert all(self.bool_feat_included[bool_feats_to_remove])

		#remove feats from included, add to excluded
		self.bool_feat_included[bool_feats_to_remove]=False
		self.bool_feat_excluded[bool_feats_to_remove]=True
		assert all(self.bool_feat_excluded^self.bool_feat_included)

		#remove unlabelled examples containing the feature
		bool_points_to_remove=(self.csr_train_feats.dot(bool_feats_to_remove)!=0)&self.bool_train_unlabelled
		self.bool_train_unlabelled[bool_points_to_remove]=False
		self.bool_train_excluded[bool_points_to_remove]=True
		assert all(self.bool_train_labelled^self.bool_train_unlabelled^self.bool_train_excluded)

		if self.bool_verbose:
			print '-'*10,'decrement_notice','-'*10
			print "FEATS: Removed ", bool_feats_to_remove.sum(), ".",\
				 self.bool_feat_included.sum(),	"used, and",\
				 self.bool_feat_excluded.sum(),"excluded, including ",\
				(self.feat_time_left>0).sum(),"given notice."
			print "POINTS: Removed ", bool_points_to_remove.sum(), ".",\
				self.bool_train_labelled.sum(), "feats labelled",\
				self.bool_train_unlabelled.sum(), "feats unlabelled",\
				self.bool_train_excluded.sum(), "feats excluded"

"""Misc Functions"""
	def run_checks(): "TODO - Run a whole bunch of checks"
		
		
		
		if True: """Check Dimensions"""
			assert self.csr_train_feats.shape[0]==\
					len(self.train_labels___0_means_unlabelled__minus_1_means_excluded)==\
					len(self.bool_train_labelled)==\
					len(self.bool_train_unlabelled)==\
					len(self.bool_train_excluded)==\
					len(self.bool_train_labelled_initially)
			assert self.csr_train_feats.shape[1]==\
					self.csr_test_feats.shape[1]==\
					len(self.bool_feat_included)==\
					len(self.bool_feat_excluded)==\
					len(self.feat_time_left)
			assert self.csr_test_feats.shape[0]==\
					len(self.test_labels)
		if True: """Check boolean arrays (mostly for Mutual Exclusivity)"""
			assert all(self.bool_train_labelled^self.bool_train_unlabelled^self.bool_train_excluded)
			assert 

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
			== len(self.feat_time_left)
		assert all(self.bool_feat_included)
		assert not any(self.bool_feat_excluded)
		assert isinstance(self.has_random_dropout,bool)
		assert set(self.test_labels)-{0}<self.set_labels
	def get_fraction_labelled(): "TODO"
		pass
	def get_num_points_labelled(): "TODO - return dict with keys orig, new, tot"
		return num_labelled_orig,num_labelled_new,num_labelled_tot
	def get_num_train(): "TODO - return dict with keys orig, new, tot"
		return num_train
	def print_self_summary(succinctness="succinct"):
		assert succinctness in {"succinct","normal","verbose"}
		print '^'*15,"printing data_manager:",succinctness,'^'*15
		print "FEATS: active:", self.bool_feat_included.sum(),\
						"excluded:", self.bool_feat_excluded,\
						"tot:", len(self.bool_feat_included)
		print "TRAIN: orig. labelled:",self.bool_train_labelled_initially.sum(),\
						"labelled:",self.bool_train_labelled.sum(),\
						"unlabelled:",self.bool_train_unlabelled.sum(),\
						"excluded:",self.bool_train_excluded.sum(),\
						"tot:",len(self.csr_train_feats)
		if succinctness="succinct":
			pass
		elif succinctness="normal":
			pass
		elif succinctness="verbose":
			print "self.bool_verbose=", self.bool_verbose

			print "self.csr_train_feats.shape =", self.csr_train_feats.shape
			print "self.csr_test_feats.shape = ", self.csr_test_feats.shape

			def print_array_summary(l_arr,l_str):				
				lab_counts=np.bincount(l_arr)
				ind_non_zero=lab_counts.nonzero()[0]
				print zip(ind_non_zero,lab_counts[ind_non_zero])," tot:",len(l_arr),":",l_str
			print_array_summary(self.train_labels___0_means_unlabelled__minus_1_means_excluded,"train_labels___0_means_unlabelled__minus_1_means_excluded")
			print_array_summary(self.test_labels,"test_labels")
			print "self.set_labels:",self.set_labels
			print "BOOL LABELS"
			print_array_summary(self.bool_train_labelled,"bool_train_labelled")
			print_array_summary(self.bool_train_unlabelled,"bool_train_unlabelled")
			print_array_summary(self.bool_train_excluded,"bool_train_excluded")
			print_array_summary(self.bool_train_labelled_initially,"bool_train_labelled_initially")
			print "FEATS"
			print_array_summary(self.bool_feat_included,"bool_feat_included")
			print_array_summary(self.bool_feat_excluded,"bool_feat_excluded")
			print_array_summary(self.feat_time_left,"feat_time_left")
			print_array_summary(self.test_labels,"test_labels")
			self.bool_feat_included=(np.ones(self.csr_train_feats.shape[1])>0) #Should be all True now
			self.bool_feat_excluded=~self.bool_feat_included #Should be all False now
			self.feat_time_left=np.ones(self.csr_train_feats.shape[1])*-1 #Time left until removed

			print "RANDOM DROPOUT"
			print "self.has_random_dropout=",self.has_random_dropout
			if self.has_random_dropout:
				print "dropout_rate=",self.dropout_rate,"num_dropout_corruptions_per_point=",self.num_dropout_corruptions_per_point
				print "array_to_kron_with=",self.array_to_kron_with
				print "dropout_matrix.shape=",self.dropout_matrix.shape
			self.num_dropout_corruptions_per_point=num_dropout_corruptions_per_point
			self.array_to_kron_with=None #Define Later if has_random_dropout
			self.dropout_matrix=None #Define Later if has_random_dropout
		else:
			raise ValueError
		print '-'*15,"end printing data_manager:",'-'*15
	def reset_to_initial_condition():
		"""Make this ready to start another semi-supervised learning trial"""

if __name__ == "__main__":
	pass 
	"""MAKE SOME CODE TO TEST THIS CLASS HERE!"""









