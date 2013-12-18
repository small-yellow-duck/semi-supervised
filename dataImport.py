from __future__ import division

def getDataOld(dataName,clean=1):

	# Set of possible labels
	f = open('data/' + dataName + '/labels')
	labels = [x.rstrip() for x in list(f)]
	f.close()

	# Set of possible features
	#f = open('data/' + dataName + '/features')
	#features = [x.rstrip() for x in list(f)]
	#f.close()

	# Set of seed rules
	f = open('data/' + dataName + '/seeds')
	seeds = [x.rstrip() for x in list(f)]
	f.close()

	# Training Data
	f = open('data/' + dataName + '/train')
	train = [x.rstrip() for x in list(f)]
	f.close()

	# Testing Data
	f = open('data/' + dataName + '/test')
	test = [x.rstrip() for x in list(f)]
	f.close()

	# Labels for testing data
	f = open('data/' + dataName + '/gold')
	gold = [x.rstrip() for x in list(f)]
	f.close()

	if clean:
		# Remove examples where test label isn't among choices
		for i in range(len(gold)-1,-1,-1):
			if gold[i] not in labels:
				test[i:i+1] = []
				gold[i:i+1] = []

	return {'labelVals':labels,
		#'featureVals':features,
		'seeds':seeds,
		'train':train,
		'test':test,
		'gold':gold}

def getData(dataName,clean=1):
	data = getDataOld(dataName,clean=clean)

	train = [set(x.split()) for x in data['train']] # Converting to set removes duplicate features
	test = [set(x.split()) for x in data['test']]
	seeds = [x.split() for x in data['seeds']]
	seeds = dict((x[1],(float(x[0]),int(x[2]))) for x in seeds)
	#featureVals = dict((data['featureVals'][i],i) for i in range(len(data['featureVals'])))
	nLabels = len(data['labelVals'])
	gold = [int(x) for x in data['gold']]

	if 'X2_Incorporated' in seeds:
		seeds['X2_Incorporated'] = (0.9999000000000001,3)

	return train,test,gold,nLabels,seeds

	
def compareData(dataName):

	f = open('data/' + dataName + '/train')
	train = [x.rstrip() for x in list(f)]
	train = [set(x.split()) for x in train]
	f.close()

	f = open('data/' + dataName + '2/train')
	train2 = [x.rstrip() for x in list(f)]
	train2 = [set(x.split()) for x in train2]
	f.close()

	for i in range(len(train)):
		diff = train2[i]-train[i]
		if diff != set([]):
			print diff

def evaluateSeeds(dataName):
	data = getData(dataName)

	train = [set(x.split()) for x in data['train']] # Converting to set removes duplicate features
	test = [set(x.split()) for x in data['test']]
	seeds = [x.split() for x in data['seeds']]
	rules = dict((x[1],(float(x[0]),int(x[2]))) for x in seeds)
	nLabels = len(data['labelVals'])
	gold = [int(x) for x in data['gold']]
	nTest = len(gold)

	nOccurences = dict()
	nCorrect = dict()
	for feature in rules:
		nOccurences[feature] = 0
		nCorrect[feature] = 0
	for d in range(len(test)):
		for feature in rules:
			if feature in test[d]:
				nOccurences[feature] += 1
				if rules[feature][1] == gold[d]:
					nCorrect[feature] += 1
	print rules
	print nCorrect
	print nOccurences

def generateSeeds(dataName,smooth=0):
	data = getData(dataName)

	train = [set(x.split()) for x in data['train']] # Converting to set removes duplicate features
	test = [set(x.split()) for x in data['test']]
	seeds = [x.split() for x in data['seeds']]
	rules = dict((x[1],(float(x[0]),int(x[2]))) for x in seeds)
	nLabels = len(data['labelVals'])
	gold = [int(x) for x in data['gold']]
	nTest = len(gold)

	nOccurences = dict()
	nCorrect = dict()
	for d in range(len(test)):
		for feature in test[d]:
			if feature not in nOccurences:
				nOccurences[feature] = 0
				nCorrect[feature] = [0]*nLabels
			nOccurences[feature] += 1
			nCorrect[feature][gold[d]-1] += 1
	for feature in nOccurences:
		for label in range(nLabels):
			if smooth:
				nCorrect[feature][label] = (nCorrect[feature][label]+1)/(nOccurences[feature]+nLabels)
			else:
				if nCorrect[feature][label] != nOccurences[feature]:
					nCorrect[feature][label] = 0
	for label in range(nLabels):
		potentialRules = sorted(((feature,nCorrect[feature][label]) for feature in nCorrect),key = lambda x:x[1])
		print "Label = "+str(label+1)
		print potentialRules[-10:]
