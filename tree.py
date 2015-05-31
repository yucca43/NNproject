import collections
UNK = 'UNK'
# This file contains the dataset in a useful way. We populate a list of Trees to train/test our Neural Nets such that each Tree contains any number of Node objects.

# The best way to get a feel for how these objects are used in the program is to drop pdb.set_trace() in a few places throughout the codebase
# to see how the trees are used.. look where loadtrees() is called etc..


class Node: # a node in the tree
    def __init__(self,label,word=None):
        self.label = label 
        self.word = word # NOT a word vector, but index into L.. i.e. wvec = L[:,node.word]
        self.parent = None # reference to parent
        self.left = None # reference to left child
        self.right = None # reference to right child
        self.isLeaf = False # true if I am a leaf (could have probably derived this from if I have a word)
        self.fprop = False # true if we have finished performing fowardprop on this node (note, there are many ways to implement the recursion.. some might not require this flag)
        self.hActs1 = None # h1 from the handout
        self.hActs2 = None # h2 from the handout (only used for RNN2)
        self.probs = None # yhat

class Tree:

    def __init__(self,treeString,openChar='(',closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        # for toks in treeString.strip().split():
        #     tokens += list(toks)
        # self.root = self.parse(tokens)
        print "*********"
        print treeString
        self.root = self.parse(treeString.strip().split())

    def parse(self,tokens, parent=None):
        print "---------"
        print tokens
        print ''.join(tokens)
        if len(tokens)<=0:
            return None
        assert tokens[0][0] == '(', "Malformed tree"
        assert tokens[-1][-1] == ')', "Malformed tree"
        split = 1 # position after open and label
        countOpen = countClose = 0
        if tokens[1][0] == '(': 
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose and split < len(tokens):
            cur = tokens[split]
            for c in cur:
                if c == '(':
                    countOpen += 1
                if c == ')':
                    countClose += 1
            split += 1 
        # New node
        print "tokens len ", len(tokens)
        print "split ", split
        if split<len(tokens):
            print "tokens[split]",tokens[split]
        if parent is None:
            # The root!
            node = Node(int(tokens[0][1])) # zero index labels
        else:
            node = Node(tokens[0][1:])

        node.parent = parent 
        # leaf Node
        if countOpen == 0:
            assert len(tokens) == 2
            closeAt = tokens[1].index(self.close)
            tokens[1][:closeAt]
            node.word = tokens[1][:closeAt].lower() # lower case?
            node.isLeaf = True
            return node
        node.left = self.parse(tokens[1:split],parent=node)
        tokens[-1] = tokens[-1][:-1]
        node.right = self.parse(tokens[split:],parent=node)
        return node

        

def leftTraverse(root,nodeFn=None,args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    nodeFn(root,args)
    if root.left is not None:
        leftTraverse(root.left,nodeFn,args)
    if root.right is not None:
        leftTraverse(root.right,nodeFn,args)


def countWords(node,words):
    if node.isLeaf:
        words[node.word] += 1

def clearFprop(node,words):
    node.fprop = False

def mapWords(node,wordMap):
    if node.isLeaf:
        if node.word not in wordMap:
            node.word = wordMap[UNK]
        else:
            node.word = wordMap[node.word]
    

def loadWordMap():
    import cPickle as pickle
    
    with open('wordMap.bin','r') as fid:
        return pickle.load(fid)

def buildWordMap():
    """
    Builds map of all words in training set
    to integer values.
    """

    import cPickle as pickle
    file = 'trees/train.txt'
    print "Reading trees to build word map.."
    with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]

    print "Counting words to give each word an index.."
    
    words = collections.defaultdict(int)
    for tree in trees:
        leftTraverse(tree.root,nodeFn=countWords,args=words)
    
    wordMap = dict(zip(words.iterkeys(),xrange(len(words))))
    wordMap[UNK] = len(words) # Add unknown as word
    
    print "Saving wordMap to wordMap.bin"
    with open('wordMap.bin','w') as fid:
        pickle.dump(wordMap,fid)

def loadTrees(dataSet='train'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    wordMap = loadWordMap()
    file = 'trees/%s.txt'%dataSet
    print "Loading %sing trees.."%dataSet
    with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]
    for tree in trees:
        leftTraverse(tree.root,nodeFn=mapWords,args=wordMap)
    return trees
      
if __name__=='__main__':
    buildWordMap()
    
    train = loadTrees()

    print "Now you can do something with this list of trees!"

