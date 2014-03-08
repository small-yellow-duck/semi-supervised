import dataImport
import DL
import misc
import Yarowsky
import sys
import pickle
import semi_supervised_learner
import matplotlib.pyplot as plt
import numpy as np
import data_manager
from sklearn import linear_model
import classifier
import shutil
import datetime
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
#SEMI_SUPERVISED_RUNTYPES={"lr_drs","per_drs","per_ave_drs","per_ave_drs_long","averaged_multilabel_perceptron","multilabel_perceptron"}
SEMI_SUPERVISED_RUNTYPES={"multilabel_perceptron"}
DO_BASIC_CLASSIFICATION=not DO_SEMI_SUPERVISED_LEARNING
#LEARNERS_TO_USE={"Perceptron","perceptron_classifier","averaged_perceptron_classifier","LogisticRegression", "multilabel_perceptron_classifier"}
LEARNERS_TO_USE={"perceptron_classifier", "multilabel_perceptron_classifier", "averaged_multilabel_perceptron_classifier"}

#DROPOUT_RATES
#LEARNERS_TO_USE={"averaged_perceptron_classifier"}

USUALLY_HIDE=True
if USUALLY_HIDE:
	if PICKLE_DATA:
		reload(Yarowsky)
		data = 'namedentity'

		# gold is labels for test data
		# test is the features for the test data
		# train: features, no labels
		train,test,gold,nLabels,rules = dataImport.getData(data)
		rules['X2_Incorporated'] = (0.9999000000000001,3) # Simulate Max's tie-breaking rule

		# Label training data based on initial seed rules
		# 0: can't decide which clLOAD_DATA_MIN_FEATURE_COUNTass is best
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
				filename="pickled_data/vectorized_data_df"+str(df)+".pickle"
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


		#(tr_X,tr_Y),tr_XU,(te_X,te_Y)=dm.get_data_only_random_dropout(.4,2)
		(tr_X,tr_Y),tr_XU,(te_X,te_Y) = dm.__get_data__(0,1,False)
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
			p=classifier.Perceptron_Classifier(5)
			print tr_X.shape, tr_Y.shape
			p.train(tr_X, tr_Y)
			print_train_and_test_error(p.predict_labels)
		if "averaged_perceptron_classifier" in LEARNERS_TO_USE:
			p=classifier.Averaged_Perceptron_Classifier(5,NUM_BETWEEN_SAMPLES)
			print "\naveraged_perceptron_classifier"
			p.train(tr_X, tr_Y)
			print_train_and_test_error(p.predict_labels)

		if "multilabel_perceptron_classifier" in LEARNERS_TO_USE:
			p=classifier.Perceptron_Multilabel_Classifier(5)
			print "\nmultilabel_perceptron_classifier"
			p.train(tr_X, tr_Y)
			print_train_and_test_error(p.predict_labels)

		if "averaged_multilabel_perceptron_classifier" in LEARNERS_TO_USE:
			p=classifier.Averaged_Perceptron_Multilabel_Classifier(5)
			print "\naveraged_multilabel_perceptron_classifier"
			p.train(tr_X, tr_Y)
			print_train_and_test_error(p.predict_labels)		
		sys.exit(0)

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
	file_name_fig=file_name_base+".png"
	plt.savefig(file_name_fig)
	#savefig(file_name,dpi=72)
	print "NEW PLOTS SAVED!"
	file_name_pickle=file_name+"_graph.pickle"
	with open(file_name_pickle,"wb") as f:
		pickle.dump((list_of_tuples_of_Val_X_and_Y_arrays,Val_description,x_label,file_name_base),f,pickle.HIGHEST_PROTOCOL)
	print "RESULTS PICKLED!"
	return [file_name_fig,file_name_pickle]

def run_then_graph_semi_supervised_learning_set(data_manager,dict_diff_from_defaults,list_of_dicts_of_run_specific_values,runs_description="DEFAULT"):
	"""Runs the semi_supervised_learner multiple times with different paramters

	There are a bunch of default paramter values. Also: 

	dict_diff_from_defaults: used to initialize some parameters differently from the defaults.
	list_of_dicts_of_run_specific_values: used to change some parameters for each run.

	"""
	str_date_time = str(datetime.datetime.now())[5:16].replace(":",",").replace(" ","_")

	ss_params={}
	def set_default_parameters():
		ss_params["ssl_Classifier_DropoutRate_Bundle"]=\
			classifier.Classifier_DropoutRate_Bundle(classifier.Averaged_Perceptron_Classifier(1,1000),.5,"apcSS")
		ss_params["list_test_set_Classifier_DropoutRate_Bundles"]=[]
		ss_params["bool_remove_features"]=False
		ss_params["notice_for_feature_removal"]=None
		ss_params["imbalance_ratio_to_trigger_notice"]=10
		ss_params["num_to_add_each_iteration"]=2000
		ss_params["max_labelled_frac"]=.9
		ss_params["num_corruptions_per_data_point"]=2
		ss_params["run_description"]="DEFAULT VALUES - THIS SHOULD HAVE BEEN RESET!"
	set_default_parameters()
	def change_params(dict_params):
		for key in dict_params:
			assert key in ss_params
			ss_params[key]=dict_params[key]
	change_params(dict_diff_from_defaults)
	rtg_dict={} #Results To Graph. a dict of lists of tuples
	results_files=[]
	for d in list_of_dicts_of_run_specific_values:
		change_params(d)
		ss_params["ssl_Classifier_DropoutRate_Bundle"].get_classifier().reset()
		for cdb in ss_params["list_test_set_Classifier_DropoutRate_Bundles"]:
			cdb.get_classifier().reset()
		ssl=semi_supervised_learner.semi_supervised_learner(\
			data_manager=data_manager,\
			ssl_Classifier_DropoutRate_Bundle=ss_params["ssl_Classifier_DropoutRate_Bundle"],\
			list_test_set_Classifier_DropoutRate_Bundles=ss_params["list_test_set_Classifier_DropoutRate_Bundles"],\
			bool_remove_features=ss_params["bool_remove_features"],\
			notice_for_feature_removal=ss_params["notice_for_feature_removal"],\
			imbalance_ratio_to_trigger_notice=ss_params["imbalance_ratio_to_trigger_notice"],\
			num_to_add_each_iteration=ss_params["num_to_add_each_iteration"],\
			max_labelled_frac=ss_params["max_labelled_frac"],\
			num_corruptions_per_data_point=ss_params["num_corruptions_per_data_point"]\
			)
		# results=ssl.do_semi_supervised_learning()
		label_results, test_error_results=ssl.do_semi_supervised_learning()
		# print "results", results
		for classifier_description, test_set_error in test_error_results.iteritems():
			if classifier_description not in rtg_dict: rtg_dict[classifier_description]=[]
			rtg_dict[classifier_description].append((ss_params["run_description"]+","+classifier_description,label_results['num_labelled'],test_set_error))
		brf=ss_params["bool_remove_features"]
		rfs="rf"+"T"*brf+"F"*(not brf) #rfT if true, else rfF
		for f in results_files:
			shutil.move('./'+f,"./Results")
		results_files=[]
		def graph_and_save_results(rtg, description):
			filename=	str_date_time+"_"+\
								runs_description+"_UPTO"+ss_params["run_description"]+"_"+description+"_"+\
								ss_params["ssl_Classifier_DropoutRate_Bundle"].get_classifier().short_description()+"_"+\
								str(rfs)+\
								"_na"+str(ss_params["num_to_add_each_iteration"])+\
								"_dr"+str(ss_params["ssl_Classifier_DropoutRate_Bundle"].get_dr())+\
								"_nc"+str(ss_params["num_corruptions_per_data_point"])+\
								"_mlf"+str(ss_params["max_labelled_frac"])
			
			graph_results_files = graph(list_of_tuples_of_Val_X_and_Y_arrays=rtg,\
																		Val_description="Dropout Rate",\
																		x_label='num_labelled',\
																		file_name_base=filename)
			pickle_results_file=filename+".pickle"
			with open(pickle_results_file, "wb") as f:
				pickle.dump(rtg,f,pickle.HIGHEST_PROTOCOL)
			'''Make sure only the latest results are left in the main folder:'''
			print "results_files",results_files
			results_files.extend(graph_results_files)
			results_files.append(pickle_results_file)
		for description, rtg in rtg_dict.iteritems():
			graph_and_save_results(rtg, description)

if DO_SEMI_SUPERVISED_LEARNING:
	drs=[.1,.3,.5,.7,.9] #Dropout RateS
	dm=data_manager.data_manager(	csr_train_feats=X_train,\
									train_labels___0_means_unlabelled=Y_train,\
									csr_test_feats=X_test,\
									test_labels=Y_test,\
									dropout_rates=set(drs),\
									max_num_dropout_corruptions_per_point=10\
								)
	for runtype in SEMI_SUPERVISED_RUNTYPES:
		dict_diff_from_defaults={}
		list_of_dicts_of_run_specific_values=[]

		'''Initialize parameter dicts differently for different run-types'''		
		# ss_params["ssl_Classifier_DropoutRate_Bundle"]=\
		# 	classifier.Classifier_DropoutRate_Bundle(classifier.Averaged_Perceptron_Classifier(1,1000),.5)
		# ss_params["list_test_set_Classifier_DropoutRate_Bundles"]={}
		# ss_params["bool_remove_features"]=False
		# ss_params["notice_for_feature_removal"]=None
		# ss_params["imbalance_ratio_to_trigger_notice"]=10
		# ss_params["num_to_add_each_iteration"]=2000
		# ss_params["max_labelled_frac"]=.95
		# ss_params["num_corruptions_per_data_point"]=2
		# ss_params["run_description"]="DEFAULT VALUES - THIS SHOULD HAVE BEEN RESET!"
		if runtype=="per_drs":
			p=classifier.Perceptron_Classifier(1)
			dict_diff_from_defaults["list_test_set_Classifier_DropoutRate_Bundles"]=\
				[classifier.Classifier_DropoutRate_Bundle(classifier.Perceptron_Classifier(1),0,"per0dr")]
			for dr in drs:
				d={}
				d["ssl_Classifier_DropoutRate_Bundle"]=classifier.Classifier_DropoutRate_Bundle(p,dr,"perSS")
				d["run_description"]="dr"+str(dr)
				list_of_dicts_of_run_specific_values.append(d)
		if runtype=="per_ave_drs":
			p=classifier.Averaged_Perceptron_Classifier(1,1000)
			for dr in drs:
				d={}
				d["ssl_Classifier_DropoutRate_Bundle"]=classifier.Classifier_DropoutRate_Bundle(p,dr,"per_ss")
				d["run_description"]="dr"+str(dr)
				list_of_dicts_of_run_specific_values.append(d)
		if runtype=="per_ave_drs_long":
			p=classifier.Averaged_Perceptron_Classifier(1,1000)
			for dr in drs:
				d={}
				d["ssl_Classifier_DropoutRate_Bundle"]=classifier.Classifier_DropoutRate_Bundle(p,dr,"per_ss")
				d["run_description"]="dr"+str(dr)
				list_of_dicts_of_run_specific_values.append(d)

		if runtype=="multilabel_perceptron":
			p=classifier.Perceptron_Multilabel_Classifier(1)
			for dr in drs:
				d={}
				d["ssl_Classifier_DropoutRate_Bundle"]=classifier.Classifier_DropoutRate_Bundle(p,dr,"per_ss")
				d["run_description"]="dr"+str(dr)
				list_of_dicts_of_run_specific_values.append(d)

			dict_diff_from_defaults["num_to_add_each_iteration"]=500
			dict_diff_from_defaults["num_corruptions_per_data_point"]=10

		if runtype=="averaged_multilabel_perceptron":
			p=classifier.Averaged_Perceptron_Multilabel_Classifier(1)
			for dr in drs:
				d={}
				d["ssl_Classifier_DropoutRate_Bundle"]=classifier.Classifier_DropoutRate_Bundle(p,dr,"per_ss")
				d["run_description"]="dr"+str(dr)
				list_of_dicts_of_run_specific_values.append(d)

			dict_diff_from_defaults["num_to_add_each_iteration"]=500
			dict_diff_from_defaults["num_corruptions_per_data_point"]=10			

		# run_then_graph_semi_supervised_learning_set(data_manager=dm,\
		# 																				dict_diff_from_defaults=dict_diff_from_defaults,\
		# 																				list_of_dicts_of_run_specific_values=list_of_dicts_of_run_specific_values,\
		# 																				runs_description=runtype)
		import cProfile
		cProfile.run("run_then_graph_semi_supervised_learning_set(data_manager=dm,\
																						dict_diff_from_defaults=dict_diff_from_defaults,\
																						list_of_dicts_of_run_specific_values=list_of_dicts_of_run_specific_values,\
																						runs_description=runtype)",\
													"ssl_stats.profile")
		import pstats
		p=pstats.Stats("ssl_stats.profile")
		p.sort_stats('cumulative').print_stats(10)
		p.sort_stats('tottime').print_stats(10)
	sys.exit(0)

			# import cProfile
			# results=cProfile.run("ssl.do_semi_supervised_learning()","ssl_stats.profile")
			# import pstats
			# p=pstats.Stats("ssl_stats.profile")
			# p.sort_stats('cumulative').print_stats(10)
			# p.sort_stats('tottime').print_stats(10)


