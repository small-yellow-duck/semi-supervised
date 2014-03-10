import dataImport
import DL
import misc
import Yarowsky
import sys

import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier 

# from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt

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
vectorizer = CountVectorizer(min_df = 2, tokenizer=splitter)
train = vectorizer.fit_transform(train2).tocsr()
test = vectorizer.transform(test2).tocsr()
# train = normalize(train, copy=False)
# test = normalize(test, copy=False)


idx = [i for i in range(len(labels)) if labels[i] != 0]
labels = np.array(labels)
#train_labelled = [train[i,:] for i in range(len(labels)) if labels[i] != 0]
#train_labelled_labels = [labels[i] for i in range(len(labels)) if labels[i] != 0]

def FIGURE_out_sparse():
	print "BOO"
	print "train:",train
	print "train[0]:",train[0]
	print "train[:,0]:",train[:,0]
	print "train[1]:",train[1]
	print "train[:,1]:",train[:,1]
	print "train[2]:",train[2]
	print "train[:,2]:",train[:,2]
	print "train[0, 67869]:",train[0, 67869]
	print "train[234, 1]:",train[234, 1]
# FIGURE_out_sparse()
freqs=train.sum(axis=0).flat
print "Total number of features: ", len(freqs)
print "Number of 1s: ", sum(1 for i in freqs if i==1)
print "shape of train:", train.shape 
print "Total Number of all-zero Rows.", sum(train.sum(axis=1)==0)
plt.hist(freqs, bins=100)

print 'vectorizer finished for split_var',split_var,' train/test shapes are:'
print train.shape, test.shape

# sys.exit()

# clf = linear_model.SGDClassifier(loss='log') #worse: penalty="elasticnet"
# clf.fit(train[idx,:], labels[idx])

clf = linear_model.LogisticRegression() 
clf.fit(train[idx,:], labels[idx])

def print_train_and_test_error():
	trainingscore = clf.score(train[idx,:], labels[idx])
	print 'SGD training score ', trainingscore
	testpredict = clf.predict(test)
	print 'SGD test score ', 1.0*np.sum(testpredict == gold)/len(testpredict)
print_train_and_test_error()

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
