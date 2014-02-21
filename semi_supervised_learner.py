
from __future__ import division
"""This class will actually perform 
semi-supervised learning using a
classifier instance and the data_manager class"""

"""TODO

"""
"""TODO LATER
-Add in forgetting
-Add in using different classifiers (with different levels of dropout) on the test set
"""
import numpy as np

class semi_supervised_learner:

	def __init__(self,\
				data_manager,\
				classifier,\
				bool_remove_features=False,\
				notice_for_feature_removal=None,\
				imbalance_ratio_to_trigger_notice=10,\
				num_to_add_each_iteration=1000,\
				max_labelled_frac=.9,\
				random_drop_out_rate=0,\
				num_corruptions_per_data_point=1\
				):
		self.data_manager=data_manager
		self.classifier=classifier

		self.bool_remove_features=bool_remove_features
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

		"""TODO"""
		def check(): 
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
		assert self.results["num_labelled"]==[] #Each semi-supervised learner is intended to be use only once
		print "1"
		dm=self.data_manager
		print "1"
		dm.reset_to_initial_condition()
		print "1"
		if self.bool_warm_start:			
			((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels))=\
					dm.__get_data__(percent_dropout=self.random_drop_out_rate,\
									num_dropout_corruptions_per_point=self.num_corruptions_per_data_point,\
									bool_targetted_dropout=self.bool_remove_features)
			for i in range(3):
				self.classifier.train(train_feats_labelled,train_labels,self.bool_warm_start)
		print "1"
		print "dm.get_fraction_labelled()",dm.get_fraction_labelled()
		while dm.get_fraction_labelled()\
					 < self.max_labelled_frac:
			dict_labelled_amts=dm.get_num_points_labelled()
			print "--Start It--, #lab:", dict_labelled_amts["num_labels_tot"],"  #unlab:", dict_labelled_amts["num_pts_unlabelled"]
			num_labels_orig=dict_labelled_amts["num_labels_orig"]
			num_labels_added=dict_labelled_amts["num_labels_added"]
			num_labels_tot=dict_labelled_amts["num_labels_tot"]
			# def print_status(self):
			# 	num_train=dm.get_num_train()
			# 	print "num labels provided:",num_labels_orig,\
			# 		"num labels added:",num_train-num_labels_orig,\
			# 		"num unlabelled",num_train-num_labels_tot,\
			# 		". Starting next iteration "
			# self.print_status()
			((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels))=\
					dm.__get_data__(percent_dropout=self.random_drop_out_rate,\
									num_dropout_corruptions_per_point=self.num_corruptions_per_data_point,\
									bool_targetted_dropout=self.bool_remove_features)
			print "start train,",
			self.classifier.train(train_feats_labelled,train_labels,self.bool_warm_start)
			def record_results():
				test_error_with_dropout=self.get_error(test_feats,test_labels)
				# print type(train_labels)
				# print train_labels[:10]
				lab_counts=np.bincount(train_labels)
				print "test err:",test_error_with_dropout,"  # each lab:",lab_counts[1:],

				self.results['num_labelled'].append(num_labels_tot)
				self.results['num_added_to_labelled'].append(num_labels_added)
				self.results['test_error_with_dropout'].append(test_error_with_dropout)
				self.results['test_error_no_dropout'].append(None)
				self.results['num_of_each_type'].append(lab_counts)
				# self.results['labels'].append(train_labels.copy())
				print "Results Recorded",
			record_results()

			dm.label_top_n(self.num_to_add_each_iteration,self.classifier,self.str_labels_to_forget)
			print "top n labelled"

			#Remove features that no longer help with generalization
			if self.bool_remove_features:
				dm.decrement_notice()
				dm.give_notice(self.imbalance_ratio_to_trigger_notice, self.notice_for_feature_removal)
		return self.results
