
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
				ssl_Classifier_DropoutRate_Bundle,\
				list_test_set_Classifier_DropoutRate_Bundles={},\
				bool_remove_features=False,\
				notice_for_feature_removal=None,\
				imbalance_ratio_to_trigger_notice=10,\
				num_to_add_each_iteration=1000,\
				max_labelled_frac=.95,\
				num_corruptions_per_data_point=1\
				):
		self.data_manager=data_manager
		self.cls_dr_for_ssl=ssl_Classifier_DropoutRate_Bundle
		self.cls_dr_for_ssl.bool_remove_features=bool_remove_features #Add an attribute (to access in for loops)
		self.all_cls_drs=list_test_set_Classifier_DropoutRate_Bundles[:]
		for cls_dr in self.all_cls_drs:
			cls_dr.bool_remove_features=False #Add an attribute
		self.all_cls_drs.append(self.cls_dr_for_ssl) #So now this contains all the classifiers!

		# self.bool_remove_features=bool_remove_features
		assert self.cls_dr_for_ssl
		self.notice_for_feature_removal=notice_for_feature_removal
		self.imbalance_ratio_to_trigger_notice=imbalance_ratio_to_trigger_notice
		self.num_to_add_each_iteration=num_to_add_each_iteration #if this is a dictionary it will be interpreted as the number of each class
		self.max_labelled_frac=max_labelled_frac
		self.num_corruptions_per_data_point=num_corruptions_per_data_point

		#Default Values
		self.bool_warm_start=True
		self.str_labels_to_forget="none" #"none","originally unlabelled","all"

		"""TODO"""
		def check(): 
			pass

		self.label_results=dict(num_labelled=[],\
			num_added_to_labelled=[],\
			num_of_each_type=[],\
			labels=[]) #For storing labels if we want to keep track of the algorithm changing its mind.
		self.test_error_results={}
		for cls_dr in self.all_cls_drs:
			assert cls_dr.get_description() not in self.test_error_results
			self.test_error_results[cls_dr.get_description()]=[]
			cls_dr.get_classifier().reset()


	def get_error(self,classifier,list_of_sets_of_feats,list_of_labels):
		assert list_of_sets_of_feats.shape[0]==len(list_of_labels)
		labels_pred=classifier.predict_labels(list_of_sets_of_feats)
		n=len(list_of_labels)
		error=(1/n)*sum(labels_pred[d]!=list_of_labels[d] for d in range(n))
		return error


	def do_semi_supervised_learning(self):
		assert self.label_results["num_labelled"]==[] #Each semi-supervised learner is intended to be use only once
		dm=self.data_manager
		dm.reset_to_initial_condition()
		if self.bool_warm_start:			
			print "Training 3 times to warm up classifier"
			((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels))=\
					dm.__get_data__(percent_dropout=self.cls_dr_for_ssl.get_dr(),\
									num_dropout_corruptions_per_point=self.num_corruptions_per_data_point,\
									bool_targetted_dropout=self.cls_dr_for_ssl.bool_remove_features)
			for i in range(3):
				self.cls_dr_for_ssl.get_classifier().train(train_feats_labelled,train_labels,self.bool_warm_start)
		print "dm.get_fraction_labelled()",dm.get_fraction_labelled()
		while dm.get_fraction_labelled()\
					 < self.max_labelled_frac:
			dict_labelled_amts=dm.get_num_points_labelled()
			num_labels_orig=dict_labelled_amts["num_labels_orig"]
			num_labels_added=dict_labelled_amts["num_labels_added"]
			num_labels_tot=dict_labelled_amts["num_labels_tot"]
			print "--Start It--, #lab:", num_labels_tot,"  #unlab:", dict_labelled_amts["num_pts_unlabelled"]
			# def print_status(self):
			# 	num_train=dm.get_num_train()
			# 	print "num labels provided:",num_labels_orig,\
			# 		"num labels added:",num_train-num_labels_orig,\
			# 		"num unlabelled",num_train-num_labels_tot,\
			# 		". Starting next iteration "
			# self.print_status()
			def train_classifier(cls_dr):
				nc=self.num_corruptions_per_data_point
				if cls_dr.get_dr() == 0:
					nc = 1
				print "cls_dr.get_dr()",cls_dr.get_dr()
				print "nc",nc
				((train_feats_labelled,train_labels),train_feats_unlabelled,(test_feats,test_labels))=\
						dm.__get_data__(percent_dropout=cls_dr.get_dr(),\
										num_dropout_corruptions_per_point=nc,\
										bool_targetted_dropout=cls_dr.bool_remove_features)
				cls_dr.get_classifier().train(train_feats_labelled,train_labels,self.bool_warm_start)
				test_error=self.get_error(cls_dr.get_classifier(),test_feats,test_labels)
				self.test_error_results[cls_dr.get_description()].append(test_error)
				print "  ",cls_dr.get_description()+"-test err:",test_error,
			print "training:",
			for cls_dr in self.all_cls_drs:
				train_classifier(cls_dr)
			def record_label_counts():
				lab_counts=np.bincount(train_labels)
				print "  # each lab:",lab_counts[1:],
				self.label_results['num_labelled'].append(num_labels_tot)
				self.label_results['num_added_to_labelled'].append(num_labels_added)
				self.label_results['num_of_each_type'].append(lab_counts)
			record_label_counts()

			dm.label_top_n(self.num_to_add_each_iteration,self.cls_dr_for_ssl.get_classifier(),self.str_labels_to_forget)
			print "top",self.num_to_add_each_iteration,"labelled"

			#Remove features that no longer help with generalization
			if self.cls_dr_for_ssl.bool_remove_features:
				dm.decrement_notice()
				dm.give_notice(self.imbalance_ratio_to_trigger_notice, self.notice_for_feature_removal)
		return self.label_results, self.test_error_results
