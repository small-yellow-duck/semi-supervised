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
import numpy as np
import sys




class data_manager:
	INIT=True
	"""Functions Related to the Initialization"""
	if INIT:
		def __init__(self,\
			csr_train_feats,\
			train_labels___0_means_unlabelled,\
			csr_test_feats,\
			test_labels,\
			dropout_rates=set(),\
			max_num_dropout_corruptions_per_point=None,\
			verbosity=5):

			def initialize_all_attribute_variables():
				self.csr_train_feats=csr_train_feats
				self.csr_test_feats=csr_test_feats

				self.dropout_rates=dropout_rates #make sure this contains 0!
				self.dict_csr_rand_dropout_matrices={dr:None for dr in dropout_rates|{0}} #"""TODO"""
				self.dict_csr_rand_and_targetted_dropout_matrices={} #These will start as copies of dict_csr_rand_dropout_matrices, but then certain features will be deleted in a targeted manner

				self.train_orig_labels=train_labels___0_means_unlabelled #This shouldn't change.
				self.bool_train_orig_labelled=(self.train_orig_labels>0) #This shouldn't change!
				self.test_labels=test_labels

				self.dropout_rates=dropout_rates
				self.max_num_dropout_corruptions_per_point=max_num_dropout_corruptions_per_point
				self.verbosity=verbosity

				self.reset_to_initial_condition(check=False)
			initialize_all_attribute_variables()

			def initialize_dropout_matrices():
				self.dict_csr_rand_dropout_matrices[0]=self.csr_train_feats
				if len(self.dropout_rates)==0: return
				for dr in self.dropout_rates:
					print "initializing matrix for dr =", dr
					def check_dropout():
						assert dr>0 and dr<1
						if not isinstance(self.max_num_dropout_corruptions_per_point,int):
							print self.max_num_dropout_corruptions_per_point
							print type(self.max_num_dropout_corruptions_per_point)
							assert False
						assert self.max_num_dropout_corruptions_per_point>0
					check_dropout()
					self.dict_csr_rand_dropout_matrices[dr]=self.get_kron_matrix(
						self.csr_train_feats,self.max_num_dropout_corruptions_per_point,type="csr")
				rand=np.random.random(len(self.dict_csr_rand_dropout_matrices[dr].data))
				for dr in sorted(self.dropout_rates):
					print "applying dropout to dropout_rate",dr
					self.dict_csr_rand_dropout_matrices[dr].data*=(rand>dr)
					print "converting back to csr"
					self.dict_csr_rand_dropout_matrices[dr]=sparse.csr_matrix(self.dict_csr_rand_dropout_matrices[dr])

				print "dropout applied"
				for dr in sorted(self.dict_csr_rand_dropout_matrices):
					matrix=self.dict_csr_rand_dropout_matrices[dr]
					matrix.eliminate_zeros()
					print "for dropout rate",dr,"there are", matrix.getnnz(),"nonzero elements"
				#self.dict_csr_rand_and_targetted_dropout_matrices matrices are only created when needed
			initialize_dropout_matrices()

			self.check_init()

		def reset_to_initial_condition(self,check=True):
			"""Make this ready to start another semi-supervised learning trial"""
			self.dict_csr_rand_and_targetted_dropout_matrices={} #These will start as copies of dict_csr_rand_dropout_matrices, but then certain features will be deleted in a targeted manner
			self.train_labels___0_unlab__neg1_exclud=self.train_orig_labels.copy()
			self.set_labels=set(self.train_orig_labels)-{0}

			self.bool_train_labelled=self.bool_train_orig_labelled.copy()
			self.bool_train_unlabelled=(self.train_labels___0_unlab__neg1_exclud==0)
			self.bool_train_excluded=(self.train_labels___0_unlab__neg1_exclud<0)
			
			self.num_train=self.csr_train_feats.shape[0]
			self.num_train_labelled=sum(self.bool_train_labelled)
			self.num_train_unlabelled=sum(self.bool_train_unlabelled)
			self.num_train_excluded=sum(self.bool_train_excluded)
			self.num_train_labelled_initially=sum(self.bool_train_orig_labelled)

			self.bool_feat_included=(np.ones(self.csr_train_feats.shape[1])>0) #Should be all True now
			self.bool_feat_excluded=~self.bool_feat_included #Should be all False now
			self.feat_time_left=np.ones(self.csr_train_feats.shape[1], int)*-1 #Time left until removed

			if check:
				self.check_init()

		def check_init(self):
			self.run_checks()
			assert not any(self.bool_train_excluded)
			assert all(self.bool_train_labelled^self.bool_train_unlabelled)
			assert self.csr_train_feats.shape[0]\
				== self.dict_csr_rand_dropout_matrices[0].shape[0] 
			nt=self.csr_train_feats.shape[0]
			for dr in self.dropout_rates:
				assert dr in self.dict_csr_rand_dropout_matrices
				assert self.dict_csr_rand_dropout_matrices[dr]!=None
			assert len(self.dict_csr_rand_and_targetted_dropout_matrices)==0
			assert all(self.bool_feat_included)
			assert not any(self.bool_feat_excluded)
			assert set(self.test_labels)-{0}<=self.set_labels
			assert self.verbosity in range(0,11)
	
	
	def set_verbosity(self,verbosity):
		assert verbosity in range(11)
		self.verbosity=verbosity

	def get_kron_matrix(self, matrix,num_duplicates,type="csr"):
		array_to_kron_with=np.ones(num_duplicates,int).reshape(num_duplicates,1) # a vertical ones matrix
		return sparse.kron(array_to_kron_with,matrix,type)

	def get_kron_array(self, array, num_duplicates):
		array_to_kron_with=np.ones(num_duplicates,int)#.reshape(num_duplicates,1) # a vertical ones matrix
		# print "array_to_kron_with.dtype",array_to_kron_with.dtype
		if isinstance(array[0],np.bool_):
			if self.verbosity > 6:	print "Changing to Booleans!"
			array_to_kron_with=array_to_kron_with>0
		kronned_array=np.kron(array_to_kron_with, array)
		if self.verbosity > 6:
			print "*"*50
			print "---get_kron_array---"
			print "array\n", array
			print "array[0]", array[0]
			print "type(array[0])", type(array[0])
			print "isinstance(array[0],np.bool_)", isinstance(array[0],np.bool_)
			print "array_to_kron_with\n", array_to_kron_with
			print "kronned_array\n", kronned_array
			print "-"*50
		return kronned_array

	LABELLING=True
	"""Functions to label unlabelled training data and move it to the labelled data"""
	if LABELLING:
		def forget_labels(labels_to_forget="none"):
			"""labels_to_forget = a string specifying which (if any) labels to forget prior to labelling some unlabelled 
								training data.  Forgetting allows the semi-supervised algorithm to "change it's mind".
								To make progress, be sure to label more unlabelled data than you are forgetting!
								There are 3 acceptable values:
				"none" - Forget no labels
				"originally unlabelled" - Forget all labels that weren't given to the data manager's constructor
				"all" - Forget all labels, so you could even change your mind about the initially labelled examples
			"""
			assert labels_to_forget in {"none","originally unlabelled","all"}
			if labels_to_forget != "none":
				if labels_to_forget == "originally unlabelled":
					self.train_labels___0_unlab__neg1_exclud=self.train_orig_labels.copy()
				elif labels_to_forget == "all":
					self.train_labels___0_unlab__neg1_exclud=np.zeros(self.num_train)
				else:
					assert False
				self.bool_train_labelled=(self.train_labels___0_unlab__neg1_exclud>0)
				self.bool_train_unlabelled=(self.train_labels___0_unlab__neg1_exclud==0)
				self.bool_train_excluded=(self.train_labels___0_unlab__neg1_exclud<0)
				self.num_train_labelled=sum(self.bool_train_labelled)
				self.num_train_unlabelled=sum(self.bool_train_unlabelled)
				self.num_train_excluded=sum(self.bool_train_excluded)

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
			print "  Pred confs:",
			labels = np.ones(self.csr_train_feats.shape[0],int)*-100
			confidences = np.ones(self.csr_train_feats.shape[0],int)*-100
			labels[self.bool_train_unlabelled],confidences[self.bool_train_unlabelled]\
				=classifier.predict_labels_and_confidences(self.csr_train_feats[self.bool_train_unlabelled])
			# print "dtykj labels", labels
			# print "asfd confidences", confidences
			print "  Sort & label:",
			confidences[self.bool_train_labelled]=-1 #So we don't try to re-label these!
			def label_top_k(k,labels,confidences):
				bool_indices_to_label=confidences.argsort()[-k:]
				self.train_labels___0_unlab__neg1_exclud[\
					bool_indices_to_label]=labels[bool_indices_to_label]
				assert not any(self.bool_train_labelled[bool_indices_to_label])
				assert all(self.bool_train_unlabelled[bool_indices_to_label])
				self.bool_train_labelled[bool_indices_to_label]=True
				self.bool_train_unlabelled[bool_indices_to_label]=False
				self.num_train_labelled+=k
				self.num_train_unlabelled-=k
			if isinstance(n,int):
				label_top_k(n,labels,confidences)
			elif isinstance(n,dict):
				assert n.keys()==self.set_labels
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
	def __get_data__(self,percent_dropout,num_dropout_corruptions_per_point,bool_targetted_dropout):
		if percent_dropout==0:
			assert num_dropout_corruptions_per_point==1
		else:
			assert percent_dropout in self.dropout_rates
			assert 1<=num_dropout_corruptions_per_point<=self.max_num_dropout_corruptions_per_point
		assert len(self.dict_csr_rand_and_targetted_dropout_matrices)<=1 #Can remove - right now I just don't see why we would have more than one defined at a time!
		if bool_targetted_dropout and percent_dropout in self.dict_csr_rand_and_targetted_dropout_matrices:
			matrix=self.dict_csr_rand_and_targetted_dropout_matrices[percent_dropout]
		else:
			matrix=self.dict_csr_rand_dropout_matrices[percent_dropout]
		if bool_targetted_dropout and percent_dropout not in self.dict_csr_rand_and_targetted_dropout_matrices:
			print "Warning: Want to get matrix with targeted feature dropout, but this matrix does not exist.  Returning a matrix with only random dropout at most."
		matrix=matrix[:self.num_train*num_dropout_corruptions_per_point]
		# print "self.train_labels___0_unlab__neg1_exclud.dtype",self.train_labels___0_unlab__neg1_exclud.dtype

		labels=self.get_kron_array(self.train_labels___0_unlab__neg1_exclud, num_dropout_corruptions_per_point)
		labelled=self.get_kron_array(self.bool_train_labelled,num_dropout_corruptions_per_point)
		tr_X=matrix[labelled]
		tr_Y=labels[labelled]
		# print "tr_Y.dtype",tr_Y.dtype
		tr_XU=self.csr_train_feats[self.bool_train_unlabelled] #No dropout for the unlabelled data!
		te_X=self.csr_test_feats
		te_Y=self.test_labels
		return ((tr_X,tr_Y),tr_XU,(te_X,te_Y))

	def get_data_for_semi_supervised(self,percent_dropout,num_dropout_corruptions_per_point):
		"""Get feature arrays for semi-supervised learning and for calculating performance on test_labels
		set using the same classifier (which is probably not the best classifier for using on the test set)"""

		"""Select the corrupted data if applicable, and otherwise the original training data"""
		return self.__get_data__(percent_dropout,num_dropout_corruptions_per_point,bool_targetted_dropout=True)
	def get_data_no_dropout(self):
		return self.__get_data__(percent_dropout=0,num_dropout_corruptions_per_point=1,bool_targetted_dropout=False)
	def get_data_only_random_dropout(self,percent_dropout,num_dropout_corruptions_per_point):		
		"""Get feature arrays for semi-supervised learning and for calculating performance on test_labels
		set using the same classifier (which is probably not the best classifier for using on the test set)"""

		"""Select the corrupted data if applicable, and otherwise the original training data"""
		return self.__get_data__(percent_dropout,num_dropout_corruptions_per_point,bool_targetted_dropout=False)

	"""Functions for removing features"""
	def get_feat_counts(self):
		num_unlabelled=self.csr_train_feats[self.bool_train_unlabelled].sum(axis=0)
		num_labelled=self.csr_train_feats[self.bool_train_labelled].sum(axis=0)
		return num_unlabelled, num_labelled
	def give_notice(self,imbalance_threshold, notice=None):
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

		train_unlabelled_feat_counts, train_labelled_feat_counts=self.get_feat_counts()

		"""Remove features not present in the unlabelled training data immediately"""
		bool_feats_to_remove=(train_unlabelled_feat_counts==0)&self.bool_feat_included&(self.feat_time_left==-1)
		self.remove_feats(bool_feats_to_remove)

		"""Give features present but rare in the unlabelled training data notice of removal in the near future"""
		bool_feats_to_give_notice=train_labelled_feat_counts/len(train_labelled_feat_counts)>\
							train_unlabelled_feat_counts/len(train_unlabelled_feat_counts)*imbalance_threshold
		bool_feats_to_give_notice=bool_feats_to_give_notice&self.bool_feat_included&(self.feat_time_left==-1)
		
		if notice==0:
			self.remove_feats(bool_feats_to_give_notice)
		else:
			self.feat_time_left[bool_feats_to_give_notice]=notice

		if self.verbosity>6:
			print '-'*10,'give_notice','-'*10
			print "NOTICE:",notice,"given to",len(bool_feats_to_give_notice),"new feats.",\
				"Tot",(self.feat_time_left>0).sum(),"feats on notice,",self.bool_feat_excluded.sum(),\
				"feats removed, and", self.bool_feat_included.sum(),"feats included."
	def decrement_notice(self):
		"""decrements self.feat_time_left and removes feats whose time has run out."""
		assert not any(self.feat_time_left==0)
		self.feat_time_left[self.feat_time_left>0]-=1
		bool_feats_to_remove=(self.feat_time_left==0)
		self.remove_feats(bool_feats_to_remove)
	def remove_feats(self,bool_indices,dropout_rate):		
		"""Removes feats specified by the given boolean indices

		all unlabelled training examples containing removed feats are also removed"""
		num_to_remove=bool_feats_to_remove.sum()
		if num_to_remove<=0:
			return
		"""Create a new targeted dropout matrix if needed"""
		if dropout_rate not in self.dict_csr_rand_and_targetted_dropout_matrices:
			assert len(self.dict_csr_rand_and_targetted_dropout_matrices)==0
			assert dropout_rate in self.dict_csr_rand_dropout_matrices
			self.dict_csr_rand_and_targetted_dropout_matrices[dropout_rate]=self.dict_csr_rand_dropout_matrices[dropout_rate].copy()
		"""Set values for features that have been removed to 0"""
		ind=[i for i in xrange(len(bool_feats_to_remove)) if bool_feats_to_remove[i]==True]
		assert len(ind)==num_to_remove
		for i in ind:
			rows,zeros=self.dict_csr_rand_and_targetted_dropout_matrices[dropout_rate][:,i].nonzero()
			for r in rows:
				assert self.dict_csr_rand_and_targetted_dropout_matrices[dropout_rate][r,i]==1
				self.dict_csr_rand_and_targetted_dropout_matrices[dropout_rate][r,i]=0
			assert sum(self.dict_csr_rand_and_targetted_dropout_matrices[dropout_rate][:,i])==0

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

		if self.verbosity>6:
			print '-'*10,'decrement_notice','-'*10
			print "FEATS: Removed ", num_to_remove, ".",\
				 self.bool_feat_included.sum(),	"used, and",\
				 self.bool_feat_excluded.sum(),"excluded, including ",\
				(self.feat_time_left>0).sum(),"given notice."
			print "POINTS: Removed ", bool_points_to_remove.sum(), ".",\
				self.bool_train_labelled.sum(), "feats labelled",\
				self.bool_train_unlabelled.sum(), "feats unlabelled",\
				self.bool_train_excluded.sum(), "feats excluded"

	"""Misc Functions"""
	def run_checks(self): 
		"Run a whole bunch of checks"
		Check_Dimensions=True
		if Check_Dimensions:
			assert self.csr_train_feats.shape[0]==\
				len(self.train_labels___0_unlab__neg1_exclud)==\
				len(self.bool_train_labelled)==\
				len(self.bool_train_unlabelled)==\
				len(self.bool_train_excluded)==\
				len(self.bool_train_orig_labelled)
			nt=self.csr_train_feats.shape[0]
			def check_dims_in_dict(m_dict):
				for dr in m_dict:
					if dr == 0:
						assert nt==m_dict[dr].shape[0]
					else:
						assert nt*self.max_num_dropout_corruptions_per_point==m_dict[dr].shape[0]
			check_dims_in_dict(self.dict_csr_rand_dropout_matrices)
			check_dims_in_dict(self.dict_csr_rand_and_targetted_dropout_matrices)
			assert len(self.dict_csr_rand_and_targetted_dropout_matrices)<=1
			assert self.csr_train_feats.shape[1]==\
					self.csr_test_feats.shape[1]==\
					len(self.bool_feat_included)==\
					len(self.bool_feat_excluded)==\
					len(self.feat_time_left)
			assert self.csr_test_feats.shape[0]==\
					len(self.test_labels)
		Check_boolean_arrays_mostly_for_Mutual_Exclusivity = True	
		if Check_boolean_arrays_mostly_for_Mutual_Exclusivity: 
			assert all(self.bool_train_labelled^self.bool_train_unlabelled^self.bool_train_excluded)
			assert not any(self.bool_train_excluded)
			assert all(self.bool_train_labelled^self.bool_train_unlabelled)
			assert self.csr_train_feats.shape[0]\
				== len(self.train_labels___0_unlab__neg1_exclud)\
				== len(self.bool_train_labelled)\
				== len(self.bool_train_unlabelled)\
				== len(self.bool_train_excluded)
			assert self.csr_test_feats.shape[0]\
				== len(self.test_labels)
			assert self.csr_train_feats.shape[1]\
				== self.csr_test_feats.shape[1]\
				== len(self.bool_feat_included)\
				== len(self.bool_feat_excluded)\
				== len(self.feat_time_left)
			assert all(self.bool_feat_included)
			assert not any(self.bool_feat_excluded)
			if not set(self.test_labels)-{0}<=self.set_labels:
				print "set(self.test_labels)-{0}", set(self.test_labels)-{0}
				print "self.set_labels", self.set_labels
				print "train labels:", len(self.train_labels___0_unlab__neg1_exclud)
				print "test labels:", len(self.test_labels)
				assert False
	"TODO"
	def get_fraction_labelled(self): 
		return self.num_train_labelled/len(self.bool_train_labelled)
	def get_num_points_labelled(self): 
		# self.bool_train_labelled=(self.train_labels___0_unlab__neg1_exclud>0)
		# self.bool_train_unlabelled=(self.train_labels___0_unlab__neg1_exclud==0)
		# self.bool_train_excluded=(self.train_labels___0_unlab__neg1_exclud<0)
		# self.bool_train_orig_labelled=self.bool_train_labelled.copy() #This shouldn't change!
		
		num_labels_tot=self.num_train_labelled
		num_labels_orig=self.num_train_labelled_initially
		num_labels_added=num_labels_tot-num_labels_orig
		num_pts_excluded=self.num_train_excluded
		num_pts_unlabelled=self.num_train_unlabelled
		tot_pts_train=self.csr_train_feats.shape[0]
		RANDOM_CHECK=random.random()<.02
		if RANDOM_CHECK:
			print "PERFORMING RANDOM CHECK ON NUM LABELLED TALLYING"
			num_labels_tot==sum(self.bool_train_labelled)
			num_labels_orig==sum(self.bool_train_orig_labelled)
			num_pts_excluded==sum(self.bool_train_excluded)
			num_pts_unlabelled==sum(self.bool_train_unlabelled)
		assert tot_pts_train==num_labels_tot+num_pts_excluded+num_pts_unlabelled
		return dict(num_labels_tot=self.num_train_labelled,\
					num_labels_orig=self.bool_train_orig_labelled,\
					num_labels_added=num_labels_added,\
					num_pts_excluded=num_pts_excluded,\
					num_pts_unlabelled=num_pts_unlabelled,\
					tot_pts_train=tot_pts_train)
	def print_self_summary(succinctness="succinct"):
		assert succinctness in {"succinct","normal","verbose"}
		print '^'*15,"printing data_manager:",succinctness,'^'*15
		print "FEATS: active:", self.bool_feat_included.sum(),\
						"excluded:", self.bool_feat_excluded,\
						"tot:", len(self.bool_feat_included)
		print "TRAIN: orig. labelled:",self.bool_train_orig_labelled.sum(),\
						"labelled:",self.bool_train_labelled.sum(),\
						"unlabelled:",self.bool_train_unlabelled.sum(),\
						"excluded:",self.bool_train_excluded.sum(),\
						"tot:",len(self.csr_train_feats)
		if succinctness=="succinct":
			pass
		elif succinctness=="normal":
			pass
		elif succinctness=="verbose":
			print "self.verbosity=", self.verbosity

			print "self.csr_train_feats.shape =", self.csr_train_feats.shape
			print "self.csr_test_feats.shape = ", self.csr_test_feats.shape

			def print_array_summary(l_arr,l_str):				
				lab_counts=np.bincount(l_arr)
				ind_non_zero=lab_counts.nonzero()[0]
				print zip(ind_non_zero,lab_counts[ind_non_zero])," tot:",len(l_arr),":",l_str
			print_array_summary(self.train_labels___0_unlab__neg1_exclud,"train_labels___0_unlab__neg1_exclud")
			print_array_summary(self.test_labels,"test_labels")
			print "self.set_labels:",self.set_labels
			print "BOOL LABELS"
			print_array_summary(self.bool_train_labelled,"bool_train_labelled")
			print_array_summary(self.bool_train_unlabelled,"bool_train_unlabelled")
			print_array_summary(self.bool_train_excluded,"bool_train_excluded")
			print_array_summary(self.bool_train_orig_labelled,"bool_train_orig_labelled")
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


if __name__ == "__main__":
	np.set_printoptions(precision=4, suppress=True)
	create_synthetic_data(num_labels=3,num_train=20,num_feats=10,frac_labelled=.6,num_test=8)
	"""MAKE SOME CODE TO TEST THIS CLASS HERE!"""





