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
		dropout_rates={0},\
		max_num_dropout_corruptions_per_point=None):

		def initialize_all_attribute_variables():
			self.bool_verbose=False

			self.csr_train_feats=csr_train_feats
			self.num_train=csr_train_feats.shape[0]
			self.csr_test_feats=csr_test_feats

			self.dropout_rates=dropout_rates #make sure this contains 0!
			self.dict_csr_rand_dropout_matrices={dr:None for dr in dropout_rates|{0}} #"""TODO"""
			self.dict_csr_rand_and_targetted_dropout_matrices={} #These will start as copies of dict_csr_rand_dropout_matrices, but then certain features will be deleted in a targeted manner


			self.train_labels___0_means_unlabelled__minus_1_means_excluded=train_labels___0_means_unlabelled
			self.orig_labels=train_labels___0_means_unlabelled.copy() #This shouldn't change.
			self.set_labels=set(train_labels___0_means_unlabelled)-{0}
			self.test_labels=test_labels

			self.bool_train_labelled=(self.train_labels___0_means_unlabelled__minus_1_means_excluded>0)
			self.bool_train_unlabelled=(self.train_labels___0_means_unlabelled__minus_1_means_excluded==0)
			self.bool_train_excluded=(self.train_labels___0_means_unlabelled__minus_1_means_excluded<0)
			self.bool_train_labelled_initially=self.bool_train_labelled.copy() #This shouldn't change!

			self.bool_feat_included=(np.ones(self.csr_train_feats.shape[1])>0) #Should be all True now
			self.bool_feat_excluded=~self.bool_feat_included #Should be all False now
			self.feat_time_left=np.ones(self.csr_train_feats.shape[1])*-1 #Time left until removed

			self.dropout_rates=dropout_rates
			self.max_num_dropout_corruptions_per_point=max_num_dropout_corruptions_per_point
		initialize_all_attribute_variables()
		def initialize_dropout_matrices():
			self.dict_csr_rand_dropout_matrices[0]=self.csr_train_feats
			if len(self.dropout_rates)==0: return
			for dr in self.dropout_rates:
				def check_dropout():
					assert dr>0 and dr<1
					assert isinstance(self.max_num_dropout_corruptions_per_point,int)
					assert self.max_num_dropout_corruptions_per_point>0
				check_dropout()
				self.dict_csr_rand_dropout_matrices[dr]=self.get_kron_matrix(
					self.csr_train_feats,self.max_num_dropout_corruptions_per_point)
			row,column=self.dict_csr_rand_dropout_matrices[dr].nonzero()
			for r,c in it.izip(row,column):
				rand=random.random()
				for dr in self.dropout_rates: #the non-zeros of higher drop-out matrices will be non-zeros of less drop-out matrices
					if rand<dr:
						self.dict_csr_rand_dropout_matrices[dr][r,c]=0
			for dr, matrix in self.dict_csr_rand_dropout_matrices.iteritems():
				matrix.eliminate_zeros()
				print "for dropout rate",dr,"there are", matrix.getnnz(),"nonzero elements"
			#self.dict_csr_rand_and_targetted_dropout_matrices matrices are only created when needed
		initialize_dropout_matrices()
		def check_init():
			self.run_checks()
			assert not any(self.bool_train_excluded)
			assert all(self.bool_train_labelled^self.bool_train_unlabelled)
			assert self.csr_train_feats.shape[0]\
				== self.dict_csr_rand_dropout_matrices[0].shape[0] 
			nt=self.csr_train_feats.shape[0]
			for dr in self.dropout_rates:
				assert dr in self.dict_csr_rand_dropout_matrices
			assert all(self.bool_feat_included)
			assert not any(self.bool_feat_excluded)
			assert set(self.test_labels)-{0}<self.set_labels
		check_init()

	# 	def make_corrupted_copies():
		# 	def check_dropout():
		# 		assert dropout_rate>0 and dropout_rate<1
		# 		assert isinstance(num_dropout_corruptions_per_point,int)
		# 		assert num_dropout_corruptions_per_point>0
		# 		assert self.has_random_dropout
		# 	check_dropout()

		# 	self.array_to_kron_with=np.ones(num_dropout_corruptions_per_point).reshape(num_dropout_corruptions_per_point,1) # a vertical ones matrix
		# 	self.dropout_matrix=self.get_kron_matrix(self.matrix)
		# 	row, col=self.dropout_matrix.nonzero()
		# 	print "Before dropout the matrix has", len(row), "non-zero elements"
		# 	assert len(row) = len(self.train_labels___0_means_unlabelled__minus_1_means_excluded)
		# 	for r, c in it.izip(row,col):
		# 		if random.random()<dropout_rate:
		# 			self.dropout_matrix[r,c]=0
		# 	self.dropout_matrix=sparse.csr_matrix(self.dropout_matrix)
		# 	row, col=self.dropout_matrix.nonzero()
		# 	print "After dropout the matrix has", len(row), "non-zero elements"
		# if has_random_dropout: make_corrupted_copies()

	def set_verbose(self,bool_verbose):
		assert isinstance(bool_verbose,bool)
		self.bool_verbose=bool_verbose

	def get_kron_matrix(self, matrix,num_duplicates):
		array_to_kron_with=np.ones(num_duplicates).reshape(num_duplicates,1) # a vertical ones matrix
		return sparse.kron(self.array_to_kron_with,matrix,'csr')

	def get_kron_array(self, array,num_duplicates):
		array_to_kron_with=np.ones(num_duplicates).reshape(num_duplicates,1) # a vertical ones matrix
		return np.kron(self.array_to_kron_with,array)

	"""Functions to label unlabelled training data and move it to the labelled data"""
	def forget_labels(labels_to_forget="none"):
		"""labels_to_forget = a string specifying which (if any) labels to forget prior to labelling some unlabelled 
							training data.  Forgetting allows the semi-supervised algorithm to "change it's mind".
							To make progress, be sure to label more unlabelled data than you are forgetting!
							There are 3 acceptable values:
			"none" - Forget no labels
			"originally unlabelled" - Forget all labels that weren't given to the data manager's constructor
			"all" - Forget all labels, so you could even change your mind about the initially labelled examples
		"""
		# 	self.train_labels___0_means_unlabelled__minus_1_means_excluded=train_labels___0_means_unlabelled
		# 	self.bool_train_labelled=(self.train_labels___0_means_unlabelled__minus_1_means_excluded>0)
		# 	self.bool_train_unlabelled=(self.train_labels___0_means_unlabelled__minus_1_means_excluded==0)
		# 	self.bool_train_excluded=(self.train_labels___0_means_unlabelled__minus_1_means_excluded<0)
		# 	self.bool_train_labelled_initially=self.bool_train_labelled.copy() #This shouldn't change!
		# 	self.orig_labels=train_labels___0_means_unlabelled.copy() #This shouldn't change.
		assert labels_to_forget in {"none","originally unlabelled","all"}
		if labels_to_forget != "none":
			if labels_to_forget == "originally unlabelled":
				self.train_labels___0_means_unlabelled__minus_1_means_excluded=self.orig_labels.copy()
			elif labels_to_forget == "all":
				self.train_labels___0_means_unlabelled__minus_1_means_excluded=np.zeros(self.num_train)
			else:
				sys.exit(1)
			self.bool_train_labelled=(self.train_labels___0_means_unlabelled__minus_1_means_excluded>0)
			self.bool_train_unlabelled=(self.train_labels___0_means_unlabelled__minus_1_means_excluded==0)
			self.bool_train_excluded=(self.train_labels___0_means_unlabelled__minus_1_means_excluded<0)

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
	def __get_data__(self,percent_dropout,num_dropout_corruptions_per_point,bool_targetted_dropout):
		if percent_dropout==0:
			assert num_dropout_corruptions_per_point==1
		else:
			assert percent_dropout in self.dropout_rates
			assert 1<=num_dropout_corruptions_per_point<=self.max_num_dropout_corruptions_per_point
		assert len(self.dict_csr_rand_and_targetted_dropout_matrices)<=1 #Can remove - right now I just don't see why we would have more than one defined at a time!
		if bool_targetted_dropout and percent_dropout in self.dict_csr_rand_and_targetted_dropout_matrices:
			matrix=self.dict_csr_rand_and_targetted_dropout_matrices[percent_dropout]
		else
			matrix=self.dict_csr_rand_dropout_matrices[percent_dropout]
		if bool_targetted_dropout and percent_dropout not in self.dict_csr_rand_and_targetted_dropout_matrices:
			print "Warning: Want to get matrix with targeted feature dropout, but this matrix does not exist.  Returning a matrix with only random dropout at most."
		matrix=matrix[:self.num_train*num_dropout_corruptions_per_point]
		labels=self.get_kron_array(self.train_labels___0_means_unlabelled__minus_1_means_excluded, num_dropout_corruptions_per_point)
		labelled=self.get_kron_array(self.bool_train_labelled,num_dropout_corruptions_per_point)
		tr_X=matrix[labelled]
		tr_Y=labels[labelled]
		tr_XU=self.csr_train_feats[self.bool_train_unlabelled] #No dropout for the unlabelled data!
		te_X=self.csr_test_feats
		te_Y=self.test_labels
		return ((tr_X,tr_Y),tr_XU,(te_X,te_Y))

	def get_data_for_semi_supervised(self,percent_dropout,num_dropout_corruptions_per_point):
		"""Get feature arrays for semi-supervised learning and for calculating performance on test_labels
		set using the same classifier (which is probably not the best classifier for using on the test set)"""

		"""Select the corrupted data if applicable, and otherwise the original training data"""
		return __get_data__(self,percent_dropout,num_dropout_corruptions_per_point,bool_targetted_dropout=True)
	def get_data_no_dropout(self):
		return __get_data__(self,percent_dropout=0,num_dropout_corruptions_per_point=1,bool_targetted_dropout=False)
	def get_data_only_random_dropout(self,percent_dropout,num_dropout_corruptions_per_point):		
		"""Get feature arrays for semi-supervised learning and for calculating performance on test_labels
		set using the same classifier (which is probably not the best classifier for using on the test set)"""

		"""Select the corrupted data if applicable, and otherwise the original training data"""
		return __get_data__(self,percent_dropout,num_dropout_corruptions_per_point,bool_targetted_dropout=False)

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

		if self.bool_verbose:
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
	def run_checks(): "TODO - Run a whole bunch of checks"
		
		
		
		if True: """Check Dimensions"""
			assert self.csr_train_feats.shape[0]==\
					len(self.train_labels___0_means_unlabelled__minus_1_means_excluded)==\
					len(self.bool_train_labelled)==\
					len(self.bool_train_unlabelled)==\
					len(self.bool_train_excluded)==\
					len(self.bool_train_labelled_initially)
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
		if True: """Check boolean arrays (mostly for Mutual Exclusivity)"""
			assert all(self.bool_train_labelled^self.bool_train_unlabelled^self.bool_train_excluded)
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
	np.set_printoptions(precision=4, suppress=True)
	create_synthetic_data(num_labels=3,num_train=20,num_feats=10,frac_labelled=.6,num_test=8)
	"""MAKE SOME CODE TO TEST THIS CLASS HERE!"""


def create_synthetic_data(num_labels,num_train,num_feats,frac_labelled,num_test):
	feat_probs=np.random.rand(num_labels,num_feats)**2*np.random.rand(num_feats)**2
	assert 0<frac_labelled<1
	def create_data(num_points):
		labels=np.random.randint(1,num_labels+1,num_points)
		assert len(np.unique(labels))==num_labels
		feats=(np.random.rand(num_points,num_feats)<feat_probs[labels])*1
		return feats,labels
	train_X,train_Y=create_data(num_train)
	test_X,test_Y=create_data(num_test)
	train_Y*=np.random.rand(num_train)<frac_labelled
	print "train labels | features"
	print np.column_stack(train_Y,np.ones(num_train),train_X)
	print np.column_stack(test_Y,np.ones(num_test),test_X)	
	return ((train_X,train_Y),(test_X,test_Y))






