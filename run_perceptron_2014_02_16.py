import dataImport
import DL
import misc
import Yarowsky
import sys
import pickle
import semi_supervised_learner

import numpy as np
import data_manager
from sklearn import linear_model
import classifier
# from sklearn.ensemble import RandomForestClassifier 

# from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt

LOAD_DATA=True
LOAD_DATA_MIN_FEATURE_COUNT=3 #in 1,2,3,5,7,10
PICKLE_DATA=not LOAD_DATA
USE_SYNTHETIC_DATA=False
DATA_FRACTION_TO_USE=1 #This should usually be set to 1, unless you want to run it quick to get preliminary results.
if USE_SYNTHETIC_DATA:
	NUM_BETWEEN_SAMPLES=10
else:
	NUM_BETWEEN_SAMPLES=1000
TEST_DATA_MANAGER=False
DO_SEMI_SUPERVISED_LEARNING=True
#SEMI_SUPERVISED_RUNTYPES={"lr_drs","per_drs","per_ave_drs"}
SEMI_SUPERVISED_RUNTYPES={"lr_drs"}
DO_BASIC_CLASSIFICATION=not DO_SEMI_SUPERVISED_LEARNING
LEARNERS_TO_USE={"Perceptron","perceptron_classifier","averaged_perceptron_classifier","LogisticRegression"}
#DROPOUT_RATES
#LEARNERS_TO_USE={"averaged_perceptron_classifier"}

if PICKLE_DATA:
	reload(Yarowsky)
	data = 'namedentity'

	# gold is labels for test data
	# test is the features for the test data
	# train: features, no labels
	train,test,gold,nLabels,rules = dataImport.getData(data)
	rules['X2_Incorporated'] = (0.9999000000000001,3) # Simulate Max's tie-breaking rule

	# Label training data based on initial seed rules
	# 0: can't decide which class is best
	# 1: person, 2: 3: corporation
	labels = DL.label(train,rules,nLabels)

	split_var='@'
	def CHECK_split_var(list_of_features):
		num_occurrences=0
		for feats in list_of_features:
			for feat in feats:
				if split_var in feat:
					print split_var, "is in", feat
					num_occurrences+=1
		print split_var, "occurs a total of", num_occurrences, "times."
		return num_occurrences
	# CHECK_split_var(train)
	# CHECK_split_var(test)

	def splitter(str):
		return str.split(split_var)
	train2 = [split_var.join(train[i]) for i in range(len(train))]
	test2 = [split_var.join(test[i]) for i in range(len(test))]

	# from sklearn.feature_extraction.text import TfidfVectorizer
	# vectorizer = TfidfVectorizer(min_df = 1, tokenizer=splitter)
	from sklearn.feature_extraction.text import CountVectorizer
	for df in {1,2,3,5,7,10}:
		vectorizer = CountVectorizer(min_df = df, tokenizer=splitter)
		X_train = vectorizer.fit_transform(train2).tocsr()
		X_test = vectorizer.transform(test2).tocsr()
		Y_train=np.array(labels)
		Y_test=np.array(gold)
		# train = normalize(train, copy=False)
		# test = normalize(test, copy=False)
		print "min_df",df
		print "train.shape",X_train.shape
		print "test.shape",X_test.shape
		filename="vectorized_data_df"+str(df)+".pickle"
		dict_to_pickle=dict(train=(X_train,Y_train),test=(X_test,Y_test))
		with open(filename,'wb') as f:
			pickle.dump(dict_to_pickle,f,pickle.HIGHEST_PROTOCOL)
	sys.exit()

if LOAD_DATA:
	if USE_SYNTHETIC_DATA:
		print '*'*50
		print 'USE_SYNTHETIC_DATA'
		(X_train,Y_train),(X_test,Y_test)=\
			misc.create_synthetic_data(	num_labels=3,\
										num_train=20,\
										num_feats=11,\
										frac_labelled=.7,\
										num_test=5,\
										sparsity=2,\
										skew=2,\
										rand_seed=1)
		print "Synthetic Data Created:"
		print "Train"
		misc.print_labels_1_feats(X_train,Y_train)
		print "test"
		misc.print_labels_1_feats(X_test,Y_test)
		print "-"*50
	else:
		def get_data(df):	
			filename="vectorized_data_df"+str(df)+".pickle"
			with open(filename,'rb') as f:
				dict_to_unpickle=pickle.load(f)
			return dict_to_unpickle
		d=get_data(LOAD_DATA_MIN_FEATURE_COUNT)
		X_train, Y_train = d["train"]
		X_test, Y_test = d["test"]
		print "min_df",LOAD_DATA_MIN_FEATURE_COUNT
		print "X train.shape", X_train.shape
		print "X test.shape", X_test.shape
		print "Y train.shape", Y_train.shape
		print "Y test.shape", Y_test.shape

if DATA_FRACTION_TO_USE != 1:
	n=len(Y_train)
	nf=int(n*DATA_FRACTION_TO_USE)
	perm=np.random.permutation(n)
	X_train=X_train[perm[:nf]]
	Y_train=Y_train[perm[:nf]]
	#At the moment we are not taking a subset of the Y data

if TEST_DATA_MANAGER:
	print '*'*50
	print 'TEST_DATA_MANAGER'
	drs=[.1,.3,.5,.7,.9]
	dm=data_manager.data_manager(	csr_train_feats=X_train,\
									train_labels___0_means_unlabelled=Y_train,\
									csr_test_feats=X_test,\
									test_labels=Y_test,\
									#dropout_rates=set(),\
									dropout_rates=set(drs),\
									max_num_dropout_corruptions_per_point=3\
								)
	print 1
	def print_data(dr):
		print "DROPOUT RATE",dr
		print "len(tr_X.nonzero()[0])", len(tr_X.nonzero()[0])
		# print "len(te_X.nonzero()[0])", len(te_X.nonzero()[0])
		# print "Train - Labelled"
		# misc.print_labels_1_feats(tr_X,tr_Y,max_examples=25,max_feats=15)
		# print "tr_XU\n",tr_XU.todense()
		# print "Test - Labelled"
		# misc.print_labels_1_feats(te_X,te_Y,max_examples=25,max_feats=15)
		# print "tr_X",tr_X.shape
		# print "tr_Y",tr_Y.shape
		# print "tr_XU",tr_XU.shape
		# print "te_X",te_X.shape
		# print "te_Y",te_Y.shape
	(tr_X,tr_Y),tr_XU,(te_X,te_Y)=dm.get_data_no_dropout()
	print_data(0)
	for dr in drs:
		(tr_X,tr_Y),tr_XU,(te_X,te_Y)=dm.get_data_only_random_dropout(dr,2)
		print_data(dr)

	print '-'*50
	sys.exit()

def print_train_and_test_error(prediction_fcn):
	pred = prediction_fcn(tr_X)
	print 'training score ', 1.0*np.sum(pred == tr_Y)/len(pred)
	pred = prediction_fcn(te_X)
	print 'test score ', 1.0*np.sum(pred == te_Y)/len(pred)
if DO_BASIC_CLASSIFICATION:
	dm=data_manager.data_manager(	csr_train_feats=X_train,\
									train_labels___0_means_unlabelled=Y_train,\
									csr_test_feats=X_test,\
									test_labels=Y_test,\
									dropout_rates={.2,.4,.6},\
									max_num_dropout_corruptions_per_point=2\
								)
	(tr_X,tr_Y),tr_XU,(te_X,te_Y)=dm.get_data_only_random_dropout(.4,2)
	if "Perceptron" in LEARNERS_TO_USE:
		print "\nPerceptron"
		p = linear_model.Perceptron() 
		p.fit(tr_X, tr_Y)
		print_train_and_test_error(p.predict)
	if "LogisticRegression" in LEARNERS_TO_USE:
		print "\nLogisticRegression"
		# p = linear_model.LogisticRegression(penalty='l1')
		p = linear_model.LogisticRegression()
		p.fit(tr_X, tr_Y)
		print_train_and_test_error(p.predict)
		for C in np.logspace(-1,2,7):			
			print "C =", C
			p.set_params(C=C)
			p.fit(tr_X, tr_Y)
			print_train_and_test_error(p.predict)
		print "\nLogisticRegression"
	if "perceptron_classifier" in LEARNERS_TO_USE:
		print "\nperceptron_classifier"
		p=classifier.perceptron_classifier(5)
		p.train(tr_X, tr_Y)
		print_train_and_test_error(p.predict_labels)
	if "averaged_perceptron_classifier" in LEARNERS_TO_USE:
		p=classifier.averaged_perceptron_classifier(5,NUM_BETWEEN_SAMPLES)
		print "\naveraged_perceptron_classifier"
		p.train(tr_X, tr_Y)
		print_train_and_test_error(p.predict_labels)
	sys.exit(0)

import matplotlib.pyplot as plt
def graph(list_of_tuples_of_Val_X_and_Y_arrays,Val_description,x_label,file_name_base):
	plt.figure()
	plt.ylabel('test_error')
	plt.xlabel(x_label)
	file_name=file_name_base
	min_x=1000000
	min_y=1000000
	max_x=0
	max_y=0
	for val, X, Y in list_of_tuples_of_Val_X_and_Y_arrays:
		print "In plot"
		print "val",val
		print "X",X
		print "Y",Y
		plt.plot(X,Y,label=Val_description+"="+str(val),linewidth=2)
		# file_name+=","+str(val)
		min_x=min(min_x,min(X))
		min_y=min(min_y,min(Y))		
		max_x=max(max_x,max(X))
		max_y=max(max_y,max(Y))
		print "plotted ",Val_description,val,file_name_base
	print "Num of lines to plot:", len(list_of_tuples_of_Val_X_and_Y_arrays)
	plt.xlim(min_x,max_x)
	plt.ylim(0,max_y)
	plt.legend(loc="lower right",prop={'size':6})
	plt.savefig(file_name+".png")
	#savefig(file_name,dpi=72)
	print "NEW PLOTS SAVED!"
	with open(file_name+".pickle","wb") as f:
		pickle.dump((list_of_tuples_of_Val_X_and_Y_arrays,Val_description,x_label,file_name_base),f,pickle.HIGHEST_PROTOCOL)
	print "RESULTS PICKLED!"

if DO_SEMI_SUPERVISED_LEARNING:
	import datetime
	file_name_base = str(datetime.datetime.now())[5:16].replace(":",",").replace(" ","_")
	if "per_drs" in SEMI_SUPERVISED_RUNTYPES:
		drs=[.1,.3,.5,.7,.9] #Dropout RateS
		print "Y_train[:10]",Y_train[:10], Y_train.dtype
		dm=data_manager.data_manager(	csr_train_feats=X_train,\
										train_labels___0_means_unlabelled=Y_train,\
										csr_test_feats=X_test,\
										test_labels=Y_test,\
										dropout_rates=set(drs),\
										max_num_dropout_corruptions_per_point=10\
									)
		print "\nperceptron_classifier"
		rtg=[] #Results To Graph

		for dr in drs:
			p=classifier.Perceptron_Classifier(1)
			rf,rfs=False,"rfF"
			na=4000
			mlf=.8
			dr=dr
			nc=10

			ssl=semi_supervised_learner.semi_supervised_learner(\
					data_manager=dm,\
					classifier=p,\
					bool_remove_features=rf,\
					# notice_for_feature_removal=None,\
					# imbalance_ratio_to_trigger_notice=10,\
					num_to_add_each_iteration=na,\
					max_labelled_frac=mlf,\
					random_drop_out_rate=dr,\
					num_corruptions_per_data_point=nc\
					)

			# import cProfile
			# results=cProfile.run("ssl.do_semi_supervised_learning()","ssl_stats.profile")
			# import pstats
			# p=pstats.Stats("ssl_stats.profile")
			# p.sort_stats('cumulative').print_stats(10)
			# p.sort_stats('tottime').print_stats(10)
			results=ssl.do_semi_supervised_learning()
			print "results",results
			rtg.append((dr,results['num_labelled'],results['test_error_with_dropout']))
			filename=file_name_base+"_"+p.short_description()+"_"+str(rfs)+"_na"+str(na)+"_UPTOdr"+str(dr)+"_nc"+str(nc)+"_mlf"+str(mlf)
			graph(	list_of_tuples_of_Val_X_and_Y_arrays=rtg,\
					Val_description="Dropout Rate",\
					x_label='num_labelled',\
					file_name_base=filename)
			with open(filename+".pickle", "wb") as f:
				pickle.dump(results,f,pickle.HIGHEST_PROTOCOL)
	

	if "per_ave_drs" in SEMI_SUPERVISED_RUNTYPES:
		drs=[.1,.3,.5,.7,.9] #Dropout RateS
		dm=data_manager.data_manager(	csr_train_feats=X_train,\
										train_labels___0_means_unlabelled=Y_train,\
										csr_test_feats=X_test,\
										test_labels=Y_test,\
										dropout_rates=set(drs),\
										max_num_dropout_corruptions_per_point=10\
									)
		print "\naveraged_perceptron_classifier"
		rtg=[] #Results To Graph

		for dr in drs:
			p=classifier.Averaged_Perceptron_Classifier(n_iter=1,sample_frequency_for_averaging=1000)
			rf,rfs=False,"rfF"
			na=4000
			mlf=.9
			dr=dr
			nc=10

			ssl=semi_supervised_learner.semi_supervised_learner(\
					data_manager=dm,\
					classifier=p,\
					bool_remove_features=rf,\
					# notice_for_feature_removal=None,\
					# imbalance_ratio_to_trigger_notice=10,\
					num_to_add_each_iteration=na,\
					max_labelled_frac=mlf,\
					random_drop_out_rate=dr,\
					num_corruptions_per_data_point=nc\
					)

			# import cProfile
			# results=cProfile.run("ssl.do_semi_supervised_learning()","ssl_stats.profile")
			# import pstats
			# p=pstats.Stats("ssl_stats.profile")
			# p.sort_stats('cumulative').print_stats(10)
			# p.sort_stats('tottime').print_stats(10)
			results=ssl.do_semi_supervised_learning()
			print "results",results
			rtg.append((dr,results['num_labelled'],results['test_error_with_dropout']))
			filename=file_name_base+"_"+p.short_description()+"_"+str(rfs)+"_na"+str(na)+"_UPTOdr"+str(dr)+"_nc"+str(nc)+"_mlf"+str(mlf)
			graph(	list_of_tuples_of_Val_X_and_Y_arrays=rtg,\
					Val_description="Dropout Rate",\
					x_label='num_labelled',\
					file_name_base=filename)
			with open(filename+".pickle", "wb") as f:
				pickle.dump(results,f,pickle.HIGHEST_PROTOCOL)

	
	if "lr_drs" in SEMI_SUPERVISED_RUNTYPES:
		drs=[.1,.3,.5,.7,.9] #Dropout RateS
		print "Y_train[:10]",Y_train[:10], Y_train.dtype
		dm=data_manager.data_manager(	csr_train_feats=X_train,\
										train_labels___0_means_unlabelled=Y_train,\
										csr_test_feats=X_test,\
										test_labels=Y_test,\
										dropout_rates=set(drs),\
										max_num_dropout_corruptions_per_point=10\
									)
		print "\nlogistic regression classifier"
		rtg=[] #Results To Graph

		for dr in drs:
			p=classifier.Logistic_Regression_Classifier()
			rf,rfs=False,"rfF"
			na=4000
			mlf=.8
			dr=dr
			nc=10

			ssl=semi_supervised_learner.semi_supervised_learner(\
					data_manager=dm,\
					classifier=p,\
					bool_remove_features=rf,\
					# notice_for_feature_removal=None,\
					# imbalance_ratio_to_trigger_notice=10,\
					num_to_add_each_iteration=na,\
					max_labelled_frac=mlf,\
					random_drop_out_rate=dr,\
					num_corruptions_per_data_point=nc\
					)

			# import cProfile
			# results=cProfile.run("ssl.do_semi_supervised_learning()","ssl_stats.profile")
			# import pstats
			# p=pstats.Stats("ssl_stats.profile")
			# p.sort_stats('cumulative').print_stats(10)
			# p.sort_stats('tottime').print_stats(10)
			results=ssl.do_semi_supervised_learning()
			print "results", results
			rtg.append((dr,results['num_labelled'],results['test_error_with_dropout']))
			filename=file_name_base+"_"+p.short_description()+"_"+str(rfs)+"_na"+str(na)+"_UPTOdr"+str(dr)+"_nc"+str(nc)+"_mlf"+str(mlf)
			graph(	list_of_tuples_of_Val_X_and_Y_arrays=rtg,\
					Val_description="Dropout Rate",\
					x_label='num_labelled',\
					file_name_base=filename)
			with open(filename+".pickle", "wb") as f:
				pickle.dump(results,f,pickle.HIGHEST_PROTOCOL)
	sys.exit(0)


