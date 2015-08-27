"""
This is a hacky little script I wrote a long time ago to parse the trees
"""

def find_index_of_close_bracket(rep, index_of_open_bracket=0):
	'''
	finds the index of the close bracket that matches the open bracket found at rep[index_of_open_bracket]def
	'''
	if rep[index_of_open_bracket] != '(':
		print index_of_open_bracket
		raise Exception("find_index_of_close_bracket didn't get a valid index of open bracket")
	obrackets = 0
	cbrackets = 0
	found = 0
	for j in range(len(rep[index_of_open_bracket:])):
		i = j+index_of_open_bracket
		char = rep[i]
		if char == '(':
			obrackets += 1
		elif char == ')':
			cbrackets += 1
		if obrackets == cbrackets:
			found = -1 # break when we find equality
			break
	if found == 0:
		raise Exception("find_index_of_close_bracket never got to equality of brackets")
	else:
		return i


def is_well_formed_PTB_rep(rep):
	'''
	checks some things regarding well-formed PTB representation
	'''
	# check that the first char is a (
	if rep[0] != '(':
		print "representation didn't start with ("
		return False

	# check that the second char is a number
	if not rep[1].isdigit():
		print "representation didn't have a label"
		return False

	# check that this is the representation of only one node; i.e, make sure the close bracket for the first open bracket is the last close bracket
	# this also makes sure that the rep ends with )
	index_of_close_bracket = find_index_of_close_bracket(rep)
	if index_of_close_bracket != (len(rep) - 1):
		print index_of_close_bracket
		print "representation wasn't of only one node"
		return False

	return True


def is_leaf(rep):
	'''
	checks if the representation is of a leaf node or not
	'''
	return rep.count('(') == 1

def find_left_right_reps(rep):
	'''
	assumes rep is the full representation of a node with two children
	'''
	# get first child's open bracket index (1: strips first bracket)
	first_child_open_bracket = rep[1:].find('(') + 1
	first_child_close_bracket = find_index_of_close_bracket(rep, first_child_open_bracket)

	second_child_open_bracket = rep[first_child_close_bracket+1:].find('(') + first_child_close_bracket+1
	second_child_close_bracket = find_index_of_close_bracket(rep, second_child_open_bracket)

	return rep[first_child_open_bracket:first_child_close_bracket+1], rep[second_child_open_bracket:second_child_close_bracket+1]

class PTBNode(object):
	'''
	a node of a PTB tree
	'''
	def __init__(self, rep, parent=None):
		'''
		string is expected to be a full string representation of a PTB node; i.e, is of the format (# (...)(...)) where # is the label and (...)(...) represents the two child nodes (if the exist)

		this will be called recursively from the top of the tree.
		'''
		rep = rep.strip() #strip newlines
		assert is_well_formed_PTB_rep(rep), "got a malformed PTB representation"
		self.parent=parent # none means root
		self.label = int(rep[1])
		self.leaf = is_leaf(rep)
		
		self.rntnparams = {'wIndex':None, 'vec':None, 'fprop':False} #rntnparams will be used by the RNTN to store info
		
		
		if not self.leaf:
			left_rep, right_rep = find_left_right_reps(rep)
			self.left = PTBNode(left_rep, self)
			self.right = PTBNode(right_rep, self)
			self.word = None
		else:
			self.left = None
			self.right = None
			self.word = rep.split()[1][:-1] # hacky but works

		

			

class PTBTree(object):
	'''
	whole PTB tree representation
	'''
	def __init__(self,rep):
		self.root = PTBNode(rep)
		self.leaves = []
		self.nodes = []
		self.traverse_tree_find_leaves()

	def traverse_tree_find_leaves(self):
		'''
		recurse down tree, add leaves to self.leaves and all nodes to self.nodes

		doesn't consider edge cases really (maybe add if needed)
		'''
		start = self.root
		self.recurse_find_leaves(start)

	def recurse_find_leaves(self,node):
		if node.leaf:
			self.leaves.append(node)
			self.nodes.append(node)
		else:
			self.nodes.append(node)
			self.recurse_find_leaves(node.left) # go left
			self.recurse_find_leaves(node.right) # go right

	def clear_rntn(self):
		for node in self.nodes:
			node.rntnparams['vec'] = None


class PTBDataset(object):
	'''
	whole dataset representation
	'''
	def __init__(self, filepath):
		self.trees = []
		f = open(filepath)
		for rep in f:
			self.trees.append(PTBTree(rep.strip().lower())) #lowercase tokens




if __name__ == '__main__':
	train = PTBDataset('trees/train.txt')
        test = PTBDataset('trees/test.txt')
        dev = PTBDataset('trees/dev.txt')

        # we follow the dataset strategy in Yoon Kim's paper (http://arxiv.org/pdf/1408.5882v2.pdf)

        # now dump these into the source/target form we want
        # note that we're making each of these 1 word just bc it fits the decoder's output form better (just a softmax basically)
        # if you make these with spaces, it should actually work just as well (it can easily memorize the structure), but it will
        # be slower because of the batching, and the learning will be a little wierd because the 'very *' would always be batched 
        # together.
        mapping = ['verynegative', 'negative', 'neutral', 'positive', 'verypositive']
	def recurse_find_leaves(leaves, node):
		if node.leaf:
			leaves.append(node)
		else:
			recurse_find_leaves(leaves, node.left) # go left
			recurse_find_leaves(leaves, node.right) # go right


        ### finegraind
        sf = open('finegrained/train/source.txt', 'w')
        tf = open('finegrained/train/target.txt', 'w')

        ntrainfine = 0
        for tree in train.trees:
            for node in tree.nodes:
                leaves = []
                recurse_find_leaves(leaves, node)
                nodestr = ' '.join([leaf.word for leaf in leaves])
                sf.write(nodestr + '\n')
                tf.write(mapping[node.label] + '\n')
                ntrainfine += 1
        sf.close()
        tf.close()

        ntestfine=0
        sf = open('finegrained/test/source.txt', 'w')
        tf = open('finegrained/test/target.txt', 'w')
        for tree in test.trees:
            treestr = ' '.join([leaf.word for leaf in tree.leaves])
            sf.write(treestr+'\n')
            tf.write(mapping[tree.root.label] + '\n')
            ntestfine+=1
        sf.close()
        tf.close()

        ndevfine =0
        sf = open('finegrained/dev/source.txt', 'w')
        tf = open('finegrained/dev/target.txt', 'w')
        for tree in dev.trees:
            treestr = ' '.join([leaf.word for leaf in tree.leaves])
            sf.write(treestr+'\n')
            tf.write(mapping[tree.root.label] + '\n')
            ndevfine += 1
        sf.close()
        tf.close()

        #mapping=neutral should never be hit.
        mapping = ['negative', 'negative', '__PLACEHOLDER__NEVER__SEE__', 'positive', 'positive']

        sf = open('binary/train/source.txt', 'w')
        tf = open('binary/train/target.txt', 'w')

        ntrainbin = 0
        for tree in train.trees:
            for node in tree.nodes:
                if node.label is 2:
                    continue
                leaves = []
                recurse_find_leaves(leaves, node)
                nodestr = ' '.join([leaf.word for leaf in leaves])
                sf.write(nodestr + '\n')
                tf.write(mapping[node.label] + '\n')
                ntrainbin+=1
        sf.close()
        tf.close()

        sf = open('binary/test/source.txt', 'w')
        tf = open('binary/test/target.txt', 'w')
        ntestbin = 0
        for tree in test.trees:
            if tree.root.label is 2:
                continue
            treestr = ' '.join([leaf.word for leaf in tree.leaves])
            sf.write(treestr+'\n')
            tf.write(mapping[tree.root.label] + '\n')
            ntestbin += 1
        sf.close()
        tf.close()

        sf = open('binary/dev/source.txt', 'w')
        tf = open('binary/dev/target.txt', 'w')
        ndevbin = 0
        for tree in dev.trees:
            if tree.root.label is 2:
                continue
            treestr = ' '.join([leaf.word for leaf in tree.leaves])
            sf.write(treestr+'\n')
            tf.write(mapping[tree.root.label] + '\n')
            ndevbin += 1
        sf.close()
        tf.close()

        # which gives us the following dataset sizes:
        # FINE GRAINED:
        # train: 318582 (all subphrases including leaves)
        # test: 2210 
        # dev: 1101
        # BINARY:
        # train: 98794 (all non-neutral subphrases)
        # test: 1821 (all non-neutral sentences)
        # dev: 872 (all non-neutral sentences)
        # we'll just make these assertions now to make sure all went well.
        assert(len(train.trees) == 8544)
        assert(ntestfine == 2210)
        assert(ndevfine == 1101)
        assert(ntestbin == 1821)
        assert(ndevbin == 872)
        assert(ntrainfine == 318582)
        assert(ntrainbin== 98794)













