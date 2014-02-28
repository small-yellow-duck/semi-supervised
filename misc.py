import operator
import numpy as np


def argmax(values):
	max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
	return max_index

def pause():
   raw_input("Press enter to continue")

def sortedInd(x,reverse=False):
	return [i[0] for i in sorted(enumerate(x),reverse=reverse,key=lambda x:x[1])]

def create_synthetic_data(num_labels,num_train,num_feats,frac_labelled,num_test,sparsity=3,skew=2,rand_seed=None):
	from scipy import sparse
	assert sparsity>=skew
	if rand_seed != None:
		np.random.seed(rand_seed)
	# np.set_printoptions(precision=3,suppress=True)
	feat_probs=np.random.rand(num_labels,num_feats)**skew*np.random.rand(num_feats)**(sparsity-skew)
	# print "feat_probs",feat_probs
	assert 0<frac_labelled<=1
	def create_data(num_points):
		labels=np.random.randint(1,num_labels+1,num_points)
		# print "labels:",labels, "unique:",np.unique(labels)
		#assert len(np.unique(labels))==num_labels
		feats=(np.random.rand(num_points,num_feats)<feat_probs[labels-1])*1
		feats=sparse.csr_matrix(feats)
		return feats, labels
	train_X,train_Y=create_data(num_train)
	test_X,test_Y=create_data(num_test)
	train_Y*=np.random.rand(num_train)<frac_labelled
	return ((train_X,train_Y),(test_X,test_Y))

def print_labels_1_feats(X,Y,max_examples=20,max_feats=15):
	"""Concatenates the labels and feats horizontally with a column of 1s in-between"""
	assert X.shape[0]==Y.shape[0]
	max_examples=min(Y.shape[0],max_examples)
	max_feats=min(max_feats,X.shape[1])
	print "labels | features"
	# print X.shape, Y.shape
	# print "max_examples",max_examples
	# print Y[:max_examples].shape
	# print np.ones(max_examples).shape
	# print X.shape
	# print X[:max_examples].shape
	# print X[:max_examples][:,:max_feats].shape
	# print X[:max_examples][:,:max_feats].todense().shape
	matrix_to_print=np.column_stack((Y[:max_examples],np.ones(max_examples),X[:max_examples][:,:max_feats].todense()))
	print matrix_to_print
	# for m in range(0,max_examples-1,5):
	# 	print matrix_to_print[m:m+5]
	# 	print '_'*50
