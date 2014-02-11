import operator

def argmax(values):
	max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
	return max_index

def pause():
    raw_input("Press enter to continue")

def sortedInd(x,reverse=False):
	return [i[0] for i in sorted(enumerate(x),reverse=reverse,key=lambda x:x[1])]

def create_synthetic_data(num_labels,num_train,num_feats,frac_labelled,num_test):
	feat_probs=np.random.rand(num_labels,num_feats)**2*np.random.rand(num_feats)**2
	assert 0<frac_labelled<1
	def create_data(num_points):
		labels=np.random.randint(1,num_labels+1,num_points)
		assert len(np.unique(labels))==num_labels
		feats=(np.random.rand(num_points,num_feats)<feat_probs[labels])*1
		return feats,labels
	train_X,train_Y=create_data(num_train)
	test_X,test_Y=create_data(num_test)
	train_Y*=np.random.rand(num_train)<frac_labelled
	print "train labels | features"
	print np.column_stack(train_Y,np.ones(num_train),train_X)
	print np.column_stack(test_Y,np.ones(num_test),test_X)	
	return ((train_X,train_Y),(test_X,test_Y))



