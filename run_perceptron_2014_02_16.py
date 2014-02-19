import dataImport
import DL
import misc
import Yarowsky
import sys
import pickle

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
if USE_SYNTHETIC_DATA:
	NUM_BETWEEN_SAMPLES=10
else:
	NUM_BETWEEN_SAMPLES=1000
TEST_DATA_MANAGER=False
PICKLE_DATA_MANAGER=False
DO_SEMI_SUPERVISED_LEARNING=False
DO_BASIC_CLASSIFICATION=not DO_SEMI_SUPERVISED_LEARNING
LEARNERS_TO_USE={"Perceptron","perceptron_classifier","averaged_perceptron_classifier","LogisticRegression"}
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
										num_train=10,\
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

if TEST_DATA_MANAGER:
	print '*'*50
	print 'TEST_DATA_MANAGER'
	drs={.05,.1,.2}
	dm=data_manager.data_manager(	csr_train_feats=X_train,\
									train_labels___0_means_unlabelled=Y_train,\
									csr_test_feats=X_test,\
									test_labels=Y_test,\
									#dropout_rates=set(),\
									dropout_rates=drs,\
									max_num_dropout_corruptions_per_point=2\
								)
	print 1
	def print_data(dr):
		print "DROPOUT RATE",dr
		print "len(tr_X.nonzero()[0])", len(tr_X.nonzero()[0])
		print "len(te_X.nonzero()[0])", len(te_X.nonzero()[0])
		print "Train - Labelled"
		misc.print_labels_1_feats(tr_X,tr_Y,max_examples=25,max_feats=15)
		print "tr_XU\n",tr_XU.todense()
		print "Test - Labelled"
		misc.print_labels_1_feats(te_X,te_Y,max_examples=25,max_feats=15)
		print "tr_X",tr_X.shape
		print "tr_Y",tr_Y.shape
		print "tr_XU",tr_XU.shape
		print "te_X",te_X.shape
		print "te_Y",te_Y.shape
	(tr_X,tr_Y),tr_XU,(te_X,te_Y)=dm.get_data_no_dropout()
	print 2
	print_data(0)
	print 3
	for dr in drs:
		print 4, dr
		(tr_X,tr_Y),tr_XU,(te_X,te_Y)=dm.get_data_only_random_dropout(dr,2)
		print_data(dr)

	print '-'*50
	sys.exit()

if PICKLE_DATA_MANAGER:
	print '*'*50
	print 'PICKLE_DATA_MANAGER'
	drs={.05,.1,.2,.3,.4,.5,.7,.8,.9}
	maxCr=10
	dm=data_manager.data_manager(	csr_train_feats=X_train,\
									train_labels___0_means_unlabelled=Y_train,\
									csr_test_feats=X_test,\
									test_labels=Y_test,\
									#dropout_rates=set(),\
									dropout_rates=drs,\
									max_num_dropout_corruptions_per_point=maxCr\
								)
	filename="dataMn_dr"+"".join(str(x) for x in drs)+"minCt"+str(LOAD_DATA_MIN_FEATURE_COUNT)+"maxCr"+maxCr+".pickle"
	with open(filename,'wb') as f:
		pickle.dump(dm,f,pickle.HIGHEST_PROTOCOL)
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





#idx0 = [i for i in range(len(labels)) if labels[i] == 0]

trainpredict0 = clf.predict(train[:])
p_trainpredict0 = clf.predict_proba(train[:])
print "the predicted probabilities start with"
print p_trainpredict0[:10]
p = np.max(p_trainpredict0, axis=1)
plt.hist(p, bins=100)

def do_semi_supervised_learning():
	idx0 = idx
	cutoff = 0.7 # cutoff certainty
	for j in [1,2]:
		idx1 = [i for i in range(len(trainpredict0)) if not (i in idx0) and p[i] > cutoff]
		clf.fit(train[idx0+idx1,:], labels[idx0+idx1])
		trainingscore = clf.score(train[idx0+idx1,:], labels[idx0+idx1])
		print 'SGD training score iter ', j, trainingscore
		testpredict = clf.predict(test)
		print 'SGD test score iter ', j, 1.0*np.sum(testpredict == gold)/len(testpredict)
		idx0 = idx0+idx1
		
		trainpredict0 = clf.predict(train[:])
		p_trainpredict0 = clf.predict_proba(train[:])
		p = np.max(p_trainpredict0, axis=1)
# do_semi_supervised_learning()


# probs = clf.predict_proba(test)
# p = np.max(probs, axis=1)
# 
# plt.hist(p, bins=100)

# predict some more labels for fitems in the training set
# pick the ones that have the highest confidence and add them to the labeled training set
# retrain

#co-training
#split the features... randomly? according to how predictive they are (_coef?)


# clf = linear_model.RidgeClassifier()
# clf.fit(train[idx,:], labels[idx])
# 
# trainingscore = clf.score(train[idx,:], labels[idx])
# print 'RidgeClassifier training score ', trainingscore
# testpredict = clf.predict(test)
# print 'RidgeClassifier test score ', 1.0*np.sum(testpredict == gold)/len(testpredict)
# 





# Compute accuracy based on seed rules
# errRate0 = DL.error(test,gold,rules,nLabels)
# print 'Accuracy with seed rules: ' + str(1-errRate0)
# 
# # Train classifier based on these labels
# rulesDL = DL.train(train,labels,nLabels,threshold=0)
# errRateDL = DL.error(test,gold,rulesDL,nLabels)
# print 'Accuracy with decision list: ' + str(1-errRateDL)
# 
# # Yarowsky algorithm
# rulesY = Yarowsky.train(train,rules,nLabels,test,gold)
# errRateY = DL.error(test,gold,rulesY,nLabels)
# print 'Accuracy with Yarowsky: ' + str(1-errRateY)

# Yarowsky-cautiaus algorithm
# rulesYC = Yarowsky.train(train,rules,nLabels,test,gold,cautiaus=5,useSmooth=0)
# errRateYC = DL.error(test,gold,rulesYC,nLabels)
# print 'Accuracy with Yarowsky-cautiaus: ' + str(1-errRateYC)
