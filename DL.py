from __future__ import division
from misc import argmax
from misc import sortedInd

def label(data,rules,nLabels,useSum=0):
	labels = []
	if useSum:
		for d in range(len(data)):
			labelScores = [0]*(nLabels+1)
			for feature in data[d]:
				if feature in rules:
					labelScores[rules[feature][1]] += rules[feature][0]
			labels.append(argmax(labelScores))
	else:
		for d in range(len(data)):
			label = 0
			bestRule = 0
			for feature in data[d]:
				if feature in rules:
					if rules[feature][0] > bestRule or (rules[feature][0] == bestRule and rules[feature][1] < label):
						bestRule = rules[feature][0]
						label = rules[feature][1]
			labels.append(label)
	return labels

def softLabel(data,rules,nLabels):
	labels = []
	for d in range(len(data)):
		label = 0
		bestRule = 0
		for feature in data[d]:
			if feature in rules:
				if rules[feature][0] > bestRule or (rules[feature][0] == bestRule and rules[feature][1] < label):
					bestRule = rules[feature][0]
					label = rules[feature][1]
		if bestRule != 0:
			labelVect = [(1-bestRule)/(nLabels-1)]*nLabels
			labelVect[label-1] = bestRule
		else:
			labelVect = [(1-bestRule)/nLabels]*nLabels
		labels.append(labelVect)
	return labels

def labelTop(data,rules,nLabels,seeds,cautiaus=1):
	labels = []
	labels2 = []
	bestRules = []
	for d in range(len(data)):
		label = 0
		label2 = 0
		bestRule = 0
		for feature in data[d]:
			if feature in seeds: # Always take labels from seed rules
				bestRule = 0
				label = rules[feature][1]
				label2 = rules[feature][1]
				break
			if feature in rules:
				if rules[feature][0] > bestRule or (rules[feature][0] == bestRule and rules[feature][1] < label):
					bestRule = rules[feature][0]
					label = rules[feature][1]
		labels.append(label)
		labels2.append(label2)
		bestRules.append(bestRule)

	index = sortedInd(bestRules,reverse=True)

	labelCount = [0]*nLabels
	labelsDone = 0
	for i in index:
		label = labels[i]
		if labelCount[label-1] < cautiaus:
			labels2[i] = labels[i]
			labelCount[label-1] += 1
			if labelCount[label-1] == labelCount:
				labelsDone += 1
				if labelsDone == nLabels:
					break
	return labels2

def error(data,gold,rules,nLabels,useSum=0):
	nSamples = len(data)
	labels = label(data,rules,nLabels,useSum)
	errs = sum([labels[d] != gold[d] for d in range(nSamples)])
	return errs/nSamples

def train(features,labels,nLabels,threshold=0,epsilon=.1,cautiaus=0,useSmooth=1):
	nSamples = len(features)
	assert(nSamples == len(labels))

	# Count the number of times each feature occurs in a labeled example,
	# 	and the number of such times it appears with each label
	featureCount = dict()
	featureLabelCount = dict()
	for d in range(nSamples):
		if labels[d] != 0:
			for feature in features[d]:
				if feature in featureCount:
					featureCount[feature] += 1
				else:
					featureCount[feature] = 1
				if (feature,labels[d]) in featureLabelCount:
					featureLabelCount[(feature,labels[d])] += 1
				else:
					featureLabelCount[(feature,labels[d])] = 1


	rules= dict()
	if cautiaus==0:
		# Find all rules above threshold
		for (feature,label) in featureLabelCount.keys():
			accuracy = (featureLabelCount[(feature,label)]+epsilon)/(featureCount[feature]+epsilon*nLabels)
			if useSmooth:
				testStat = accuracy
			else:
				testStat = featureLabelCount[(feature,label)]/featureCount[feature]
			if testStat > threshold:
				if feature not in rules or (feature in rules and accuracy > rules[feature][0]):
					rules[feature] = (accuracy,label)
		return rules
	else: # Cautiausness based on raw scores then number of examples supporting score
		candidates = []
		for (feature,label) in featureLabelCount.keys():
			accuracy = (featureLabelCount[(feature,label)]+epsilon)/(featureCount[feature]+epsilon*nLabels)
			accuracyRaw = featureLabelCount[(feature,label)]/featureCount[feature]
			if useSmooth:
				testStat = accuracy
			else:
				testStat = accuracyRaw
			if testStat > threshold:
				candidates.append([accuracyRaw,featureCount[feature],feature,accuracy,label])

		# Sort values
		sortedValues = sorted((candidate for candidate in candidates),key = lambda x:(-x[0],-x[1],x[2]))
		
		# Now take top rules for each feature
		labelCount = [0]*nLabels
		labelsDone = 0
		for candidate in sortedValues:
			label = candidate[4]
			if labelCount[label-1] < cautiaus:
				feature = candidate[2]
				accuracy = candidate[3]
				rules[feature] = (accuracy,label)
				labelCount[label-1] += 1
				if labelCount[label-1] == cautiaus:
					labelsDone += 1
					if labelsDone == nLabels:
						break
		return rules

def labelSum(data,rules,nLabels):
	labels = []
	for d in range(len(data)):
		labelScores = [0]*(nLabels+1)
		for feature in data[d]:
			if feature in rules:
				for (labelScore,labelVal) in rules[feature]:
					labelScores[labelVal] += labelScore
		labels.append(argmax(labelScores))
	return labels

def errorSum(data,gold,rules,nLabels):
	nSamples = len(data)
	labels = labelSum(data,rules,nLabels)
	errs = sum([labels[d] != gold[d] for d in range(nSamples)])
	return errs/nSamples

def trainSum(features,labels,nLabels,threshold=0,epsilon=.1,cautiaus=0):
	nSamples = len(features)
	assert(nSamples == len(labels))

	# Count the number of times each feature occurs in a labeled example,
	# 	and the number of such times it appears with each label
	featureCount = dict()
	featureLabelCount = dict()
	for d in range(nSamples):
		if labels[d] != 0:
			for feature in features[d]:
				if feature in featureCount:
					featureCount[feature] += 1
				else:
					featureCount[feature] = 1
				if (feature,labels[d]) in featureLabelCount:
					featureLabelCount[(feature,labels[d])] += 1
				else:
					featureLabelCount[(feature,labels[d])] = 1
	rules= dict()
	if cautiaus==0:
		# Find all rules above threshold
		for (feature,label) in featureLabelCount.keys():
			accuracy = (featureLabelCount[(feature,label)]+epsilon)/(featureCount[feature]+epsilon*nLabels)
			if accuracy > threshold:
				if feature not in rules:
					rules[feature] = [(accuracy,label)]
				else:
					#print "Feature in rules!"
					rules[feature].append((accuracy,label))
	else: # Cautiausness based on raw scores then number of examples supporting score
		candidates = []
		for (feature,label) in featureLabelCount.keys():
			accuracy = (featureLabelCount[(feature,label)]+epsilon)/(featureCount[feature]+epsilon*nLabels)
			accuracyRaw = featureLabelCount[(feature,label)]/featureCount[feature]
			if accuracyRaw > threshold:
				candidates.append([accuracyRaw,featureCount[feature],feature,accuracy,label])

		# Sort values
		sortedValues = sorted((candidate for candidate in candidates),key = lambda x:(1-x[0],nSamples-x[1],x[2],x[4]))
		
		# Now take top rules for each feature
		labelCount = [0]*nLabels
		labelsDone = 0
		for candidate in sortedValues:
			label = candidate[4]
			if labelCount[label-1] < cautiaus:
				feature = candidate[2]
				accuracy = candidate[3]
				if feature not in rules:
					rules[feature] = [(accuracy,label)]
				else:
					rules[feature].append((accuracy,label))
				labelCount[label-1] += 1
				if labelCount[label-1] == cautiaus:
					labelsDone += 1
					if labelsDone == nLabels:
						break
	return rules
