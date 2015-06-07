import numpy as np
import collections
import pdb



class RNN3:

    def __init__(self,wvecDim, middleDim, outputDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.middleDim = middleDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho
        self.pretrain = pretrain

    def initParams(self):
        np.random.seed(12341)

        # Word vectors
        if self.pretrain:
            import cPickle as pickle
            with open('wordvectors/wordvectors.'+str(self.wvecDim)+'d.bin','r') as fid:
                print "Loading from pretrained wordvectors"
                self.L = pickle.load(fid)
        else:       
            self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights for layer 1
        self.W1 = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim+1)
        self.b1 = np.zeros((self.wvecDim))

        # Hidden activation weights for layer 2
        self.W2 = 0.01*np.random.randn(self.middleDim,self.wvecDim)
        self.b2 = np.zeros((self.middleDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.middleDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs]

        # Gradients
        self.dW1 = np.empty(self.W1.shape)
        self.db1 = np.empty((self.wvecDim))
        
        self.dW2 = np.empty(self.W2.shape)
        self.db2 = np.empty((self.middleDim))

        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))


    def costAndGrad(self,mbdata,test=False): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W1, W2, Ws, b1, b2, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns 
           cost, correctArray, guessArray, total
        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs = self.stack
        # Zero gradients
        self.dW1[:] = 0
        self.db1[:] = 0
        
        self.dW2[:] = 0
        self.db2[:] = 0

        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

            
        if test:
            for tree in mbdata: 
                c,tot = self.forwardProp(tree.root,correct,guess,0,True)
                cost += c
                total += tot
            return (1./len(mbdata))*cost,correct, guess, total
        else:
            # Forward prop each tree in minibatch
            for tree in mbdata: 
                c,tot = self.forwardProp(tree.root,correct,guess)
                cost += c
                total += tot

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.W1**2)
        cost += (self.rho/2)*np.sum(self.W2**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dW1 + self.rho*self.W1),scale*self.db1,
                                   scale*(self.dW2 + self.rho*self.W2),scale*self.db2,
                                   scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]


    def forwardProp(self,node, correct=[], guess=[],level=0,testTime=False):
        cost  =  total = 0.0
        # this is exactly the same setup as forwardProp in rnn.py
        
        if(node.isLeaf):
        # if(node.left is None or node.right is None):
            # assert node.left is None and node.right is None
            # do nothing?
            node.hActs1 = self.L[:,node.word]
        else:
            Lcost,Ltotal = self.forwardProp(node.left,correct,guess,level+1)  
            Rcost,Rtotal = self.forwardProp(node.right,correct,guess,level+1) 
            total = Ltotal + Rtotal
            cost = Lcost + Rcost
            # Hidden Activation 1
            node.hActs1 = np.dot(self.W1, np.hstack([node.left.hActs1, node.right.hActs1,level])) + self.b1
            # ReLu
            node.hActs1[node.hActs1<0] = 0;

        # Hidden Activation 2
        node.hActs2 = np.dot(self.W2, node.hActs1) + self.b2
        # ReLu 2
        node.hActs2[node.hActs2<0] = 0;
        # softmax
        if testTime:
            node.probs = (np.dot(self.Ws,node.hActs2) + self.bs)*0.5
        else:
            #dropouts
            # nums = np.random.random(node.hActs2.shape)
            # node.mask = nums>0.5
            # can be turned on to use GradCheck, remember to disable the previous 2 lines at the same time
            if not hasattr(node,'mask'):
                nums = np.random.random(node.hActs2.shape)
                node.mask = nums>0.5
            h2 = node.hActs2*node.mask
            # perform softmax
            node.probs = np.dot(self.Ws,h2) + self.bs
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs/np.sum(node.probs)

        cost -= np.log(node.probs[node.label])
        correct.append(node.label)
        guess.append(np.argmax(node.probs))
        return cost, total + 1

    def backProp(self,node,error=None,level=0):

        # Clear nodes
        node.fprop = False

        # this is exactly the same setup as backProp in rnn.py
        # delta 3
        deltas = node.probs
        deltas[node.label] -= 1.0
        # U and bs
        h2 = node.hActs2*node.mask
        self.dWs += np.outer(deltas,h2)
        self.dbs += deltas

        # delta_2^(2)
        deltas = np.dot(self.Ws.T, deltas)
        deltas *= (h2 != 0)
        # W2 and b2
        self.dW2 += np.outer(deltas,node.hActs1)
        self.db2 += deltas

        # delta_2^(1) w/0 ReLu
        deltas = np.dot(self.W2.T, deltas)

        if error is not None:
            deltas += error

        deltas *= (node.hActs1 != 0)

        if node.isLeaf:
            self.dL[node.word] += deltas
            return

        if not node.isLeaf:
            self.dW1 += np.outer(deltas,np.hstack([node.left.hActs1, node.right.hActs1,level]))
            self.db1 += deltas
            deltas = np.dot(self.W1.T, deltas)
            self.backProp(node.left, deltas[:self.wvecDim],level+1)
            self.backProp(node.right, deltas[self.wvecDim:-1],level+1)
       
        
        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)

        err1 = 0.0
        count = 0.0
        print "Checking dWs, dW1 and dW2..."
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None] # add dimension since bias is flat
            dW = dW[...,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    W[i,j] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j] - numGrad)
                    err1+=err
                    count+=1
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)
        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                err2+=err
                count+=1

        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)

if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    middleDim = 10
    outputDim = 5

    rnn = RNN3(wvecDim,middleDim,outputDim,numW,mbSize=4)
    rnn.initParams()

    mbData = train[:4]
    
    print "Numerical gradient check..."
    rnn.check_grad(mbData)






