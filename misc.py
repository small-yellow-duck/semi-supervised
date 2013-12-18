import operator

def argmax(values):
	max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
	return max_index

def pause():
    raw_input("Press enter to continue")

def sortedInd(x,reverse=False):
	return [i[0] for i in sorted(enumerate(x),reverse=reverse,key=lambda x:x[1])]
