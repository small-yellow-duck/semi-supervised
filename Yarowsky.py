from __future__ import division
from misc import argmax
import DL
from random import random
from math import floor

def train(features,seeds,nLabels,testFeatures=[],testLabels=[],epsilon=0.1,useSum=0,threshold=0.95,maxIter = 500,cautiaus=0,useSmooth=1):
	
	labels = DL.label(features,seeds,nLabels,useSum=useSum)
	oldLabels = labels
	oldLen = len(seeds)

	rules = DL.train(features,labels,nLabels,epsilon=epsilon,threshold=threshold,cautiaus=cautiaus,useSmooth=useSmooth)
	rules.update(seeds)
	labels = DL.label(features,rules,nLabels,useSum=useSum)

	labels = DL.label(features,rules,nLabels,useSum=useSum)

	changes = sum(i!=j for i,j in zip(labels,oldLabels))

	for i in range(maxIter):

		if testFeatures != []:
			print "iter = "+str(i)+", testErr = "+str(1-DL.error(testFeatures,testLabels,rules,nLabels))+", nRules = "+str(len(rules))+", changes = "+str(changes)
		elif i%10 == 0:
			print i

		rules = DL.train(features,labels,nLabels,epsilon=epsilon,threshold=threshold,cautiaus=cautiaus*(i+2),useSmooth=useSmooth)
		rules.update(seeds)
		labels = DL.label(features,rules,nLabels,useSum=useSum)

		labels = DL.label(features,rules,nLabels,useSum=useSum)

		changes = sum(i!=j for i,j in zip(labels,oldLabels))
		if (changes == 0) and (cautiaus == 0 or oldLen == len(rules)):
			break
		oldLabels = labels
		oldLen = len(rules)

	print "Finished on iteration "+str(i)+ ", re-training..."
	rules = DL.train(features,labels,nLabels,threshold=0,cautiaus=0)
	rules.update(seeds)
	
	return rules


