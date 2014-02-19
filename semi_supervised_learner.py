"""This class will actually perform 
semi-supervised learning using a
classifier instance and the data_manager class"""

"""TODO

"""
"""TODO LATER
-Add in forgetting
-Add in using different classifiers (with different levels of dropout) on the test set
"""
class semi_supervised_learner:

	def __init__(self,data_manager,classifier,\
				notice_for_feature_removal=None,\
				imbalance_ratio_to_trigger_notice=10,\
				num_to_add_each_iteration=1000,\ 
				max_labelled_frac=.9,\ 
				random_drop_out_rate=0,\
				num_corruptions_per_data_point=1\
				):
		self.data_manager=data_manager
		self.classifier=classifier
		self.notice_for_feature_removal=notice_for_feature_removal
		self.imbalance_ratio_to_trigger_notice=imbalance_ratio_to_trigger_notice
		self.num_to_add_each_iteration=num_to_add_each_iteration #if this is a dictionary it will be interpreted as the number of each class
		self.max_labelled_frac=max_labelled_frac
		self.random_drop_out_rate=random_drop_out_rate
		self.num_corruptions_per_data_point=num_corruptions_per_data_point

		#Default Values
		self.max_iterations_if_not_adding=5
		self.bool_warm_start=True
		self.str_labels_to_forget="none" #"none","originally unlabelled","all"

		def check(): """TODO"""
			pass

		self.results=dict(num_labelled=[],\
			num_added_to_labelled=[],\
			test_error_with_dropout=[],\
			test_error_no_dropout=[],\
			num_of_each_type=[],\
			labels=[]) #For storing labels if we want to keep track of the algorithm changing its mind.

	def get_error(self,list_of_sets_of_feats,list_of_labels):
		assert list_of_sets_of_feats.shape[0]==len(list_of_labels)
		labels_pred=self.classifier.predict_labels(list_of_sets_of_feats)
		n=len(list_of_labels)
		error=(1/n)*sum(labels_pred[d]!=list_of_labels[d] for d in range(n))
		return error

	def do_semi_supervised_learning(self):
		assert self.results[num_labelled]==[] #Each semi-supervised learner is intended to be use only once
		dm=self.data_manager
		dm.reset_to_initial_condition()

		while dm.get_fraction_labelled()\
					 < self.max_labelled_frac:
			num_labelled_orig,num_labelled_new,num_labelled_tot=dm.get_num_points_labelled()
			def print_status(self):
				num_train=dm.get_num_train()
				print "num labels provided:",num_labelled_orig,\
					"num labels added:",num_train-num_labelled_orig,\
					"num unlabelled",num_train-num_labelled_tot,\
					". Starting next iteration "
			self.print_status()
			((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels))=\
				dm.get_feats_for_semi_supervised(self.random_drop_out_rate,\
													self.num_corruptions_per_data_point)
			self.classifier.train(train_feats_labelled,train_labels,self.bool_warm_start)
			def record_results():"""TODO"""
				test_error_with_dropout=self.get_error(test_feats,test_labels)
				lab_counts=np.bincount(train_labels)

				self.results['num_labelled'].append(num_labelled_tot)
				self.results['num_added_to_labelled'].append(num_labelled_new)
				self.results['test_error_with_dropout'].append(test_error_with_dropout)
				self.results['test_error_no_dropout'].append(None)
				self.results['num_of_each_type'].append(lab_counts)
				self.results['labels'].append(train_labels.copy())
			record_results()
			dm.label_top_n(self.num_to_add_each_iteration,self.classifier,self.str_labels_to_forget)

			#Remove features that no longer help with generalization
			dm.decrement_notice()
			dm.give_notice(self.imbalance_ratio_to_trigger_notice, self.notice_for_feature_removal)
		return self.results_points_N_test_error, self.results_epochs_points_N_test_error
