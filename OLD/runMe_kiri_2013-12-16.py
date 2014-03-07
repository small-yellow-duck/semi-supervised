import dataImport
import DL
import misc
import Yarowsky

import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt

def commasplitter(str):
	return str.split(',')


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

train2 = [','.join(train[i]) for i in range(len(train))]
test2 = [','.join(test[i]) for i in range(len(test))]
tfidf = TfidfVectorizer(min_df = 1, tokenizer=commasplitter)

train = tfidf.fit_transform(train2).tocsr()
test = tfidf.transform(test2).tocsr()
# train = normalize(train, copy=False)
# test = normalize(test, copy=False)

idx = [i for i in range(len(labels)) if labels[i] != 0]
labels = np.array(labels)
#train_labelled = [train[i,:] for i in range(len(labels)) if labels[i] != 0]
#train_labelled_labels = [labels[i] for i in range(len(labels)) if labels[i] != 0]


print 'vectorizer finished'
print train.shape, test.shape

clf = linear_model.SGDClassifier(loss='log') #worse: penalty="elasticnet"
clf.fit(train[idx,:], labels[idx])

# cutoff for selecting the most predictive features (1.0 => training score of 0.66)
c = 0.8

# select the most predictive features
best = list(set(range(len(clf.coef_[1,:]))*(((np.abs(clf.coef_[0,:]) > c) + (np.abs(clf.coef_[1,:]) > c) + (np.abs(clf.coef_[2,:]) > c)) > 0))  - set([0]))

# most of the features are not predictive of the label
fig = plt.figure(0)
plt.clf()
plt.hist(clf.coef_[0,:], bins=100, color='b')
plt.hist(clf.coef_[1,:], bins=100, color='r')
plt.hist(clf.coef_[2,:], bins=100, color='g')
#plt.legend([], loc='upper right', frameon=False)
plt.ylabel('# of features')
plt.xlabel('predictive coeff (large -> strongly predictive)')
plt.show()


clf = linear_model.SGDClassifier(loss='modified_huber') #worse: penalty="elasticnet"
clf.fit( train[idx,:][:,best], labels[idx])

trainingscore = clf.score(train[idx,:][:,best], labels[idx])
print 'SGD training score ', trainingscore
testpredict = clf.predict(test[:,best])
print 'SGD test score ', 1.0*np.sum(testpredict == gold)/len(testpredict)

#idx0 = [i for i in range(len(labels)) if labels[i] == 0]

trainpredict0 = clf.predict(train[:,best])
p_trainpredict0 = clf.predict_proba(train[:,best])
p = np.max(p_trainpredict0, axis=1)



fig = plt.figure(1)
plt.clf()
plt.hist(p[list(set(range(len(p))) -set(idx))], bins=100, color='b')
plt.hist(p[idx], bins=100, color='r')
plt.legend(['unlabeled training', 'labeled training'], loc='upper right', frameon=False)
plt.ylabel('# of predictions')
plt.xlabel('confidence in prediction')
plt.show()

idx0 = idx
cutoff = 0.66 # cutoff certainty
for j in [0]:
	idx1 = [i for i in range(len(trainpredict0)) if not (i in idx0) and p[i] > cutoff]
	labels[idx1] = clf.predict(train[idx1,:][:,best])
	
	#generate new fit for expanded training set (previous training subset idx0 and the new training subset idx1)
	clf.fit(train[idx0+idx1,:], labels[idx0+idx1])
	
	best = list(set(range(len(clf.coef_[1,:]))*(((np.abs(clf.coef_[0,:]) > c) + (np.abs(clf.coef_[1,:]) > c) + (np.abs(clf.coef_[2,:]) > c)) > 0))  - set([0]))
	clf.fit(train[idx0+idx1,:][:,best], labels[idx0+idx1])


	trainingscore = clf.score(train[idx0+idx1,:][:,best], labels[idx0+idx1])
	print 'SGD training score iter ', j, trainingscore
	testpredict = clf.predict(test[:,best])
	print 'SGD test score iter ', j, 1.0*np.sum(testpredict == gold)/len(testpredict)
	idx0 = idx0+idx1
	
	trainpredict0 = clf.predict(train[:,best])
	p_trainpredict0 = clf.predict_proba(train[:,best])
	p = np.max(p_trainpredict0, axis=1)



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
