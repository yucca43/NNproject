import numpy as np
import collections
import sys
np.seterr(over='raise',under='raise')

class RNTN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,pretrain=False,dropout=False,rho=1e-6):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho
        self.pretrain = pretrain
        self.dropout = dropout

    def initParams(self):
        
        # Word vectors
        if self.pretrain:
            import cPickle as pickle
            with open('wordvectors/wordvectors.'+str(self.wvecDim)+'d.bin','r') as fid:
                print "Loading from pretrained wordvectors"
                self.L = pickle.load(fid)
        else:       
            self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights
        self.V = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = 0.01*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W, Ws, b, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns 
           cost, correctArray, guessArray, total
           
        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L, self.V, self.W, self.b, self.Ws, self.bs = self.stack
        # self.L,self.W,self.b,self.Ws,self.bs = self.stack

        # Zero gradients
        self.dV[:] = 0
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        if test:
            for tree in mbdata: 
                c,tot = self.forwardProp(tree.root,correct,guess,True)
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
        cost += (self.rho/2)*np.sum(self.V ** 2)
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)
        # cost,,,
        return scale*cost,[self.dL,scale*(self.dV+self.rho*self.V),scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]


    def forwardProp(self,node,correct=[], guess=[],testTime=False):
        node.fprop = True
        cost  =  total = 0.0 # cost should be a running number and total is the total examples we have seen used in accuracy reporting later

        if(node.isLeaf):
        # if(node.left is None or node.right is None):
            # assert node.left is None and node.right is None
            # do nothing?
            node.hActs1 = self.L[:,node.word]
        else:
            if node.right is not None:
                Lcost,Ltotal = self.forwardProp(node.left,correct,guess)  
                Rcost,Rtotal = self.forwardProp(node.right,correct,guess) 
                total = Ltotal + Rtotal
                cost = Lcost + Rcost
                # Hidden Activation
                h_combined = np.hstack([node.left.hActs1, node.right.hActs1])
                node.hActs1 = np.dot(h_combined,self.V).dot(h_combined) + np.dot(self.W, h_combined) + self.b
                # tanh
                node.hActs1 = np.tanh(node.hActs1)
            else:
                self.forwardProp(node.left,correct,guess)
                node.hActs1 = node.left.hActs1
                # ReLu
                node.hActs1[node.hActs1<0] = 0
        if node.parent is None:
            # softmax
            if self.dropout:
                if testTime:
                    node.probs = (np.dot(self.Ws,node.hActs1) + self.bs)*0.5
                else:
                    #dropouts
                    nums = np.random.random(node.hActs1.shape)
                    node.mask = nums>0.5
                    h1 = node.hActs1*node.mask
                    node.probs = np.dot(self.Ws,h1) + self.bs
            else:
                node.probs = np.dot(self.Ws,node.hActs1) + self.bs
            node.probs -= np.max(node.probs)
            node.probs = np.exp(node.probs)
            node.probs = node.probs/np.sum(node.probs)
            cost -= np.log(node.probs[node.label])
            correct.append(node.label)
            guess.append(np.argmax(node.probs))
            return cost, total + 1
        return 0,total


    def backProp(self,node,error=None):
        if not node.isLeaf and node.right is None:
            self.backProp(node.left, error)
            return
        # Clear nodes
        node.fprop = False
        deltas = 0

        if node.parent is None:
            # delta 3
            deltas = node.probs
            deltas[node.label] -= 1.0
            # U and b
            if self.dropout:
                h1 =  node.hActs1*node.mask
                self.dWs += np.outer(deltas,h1)
            else:
                self.dWs += np.outer(deltas,node.hActs1)
            self.dbs += deltas
            # delta_2^(1) w/0 relu'
            deltas = np.dot(self.Ws.T, deltas)

        if error is not None:
            deltas += error

        # delta_2^(1)
        # tanh
        tanhx = -np.tanh(node.hActs1)
        deltas *= (-tanhx*tanhx +1)

        if node.isLeaf:
            self.dL[node.word] += deltas
            return

        if not node.isLeaf:
            h_combined = np.hstack([node.left.hActs1, node.right.hActs1])
            hh = np.outer(h_combined, h_combined)
            self.dV += np.dstack([d*hh for d in deltas]).T
            self.dW += np.outer(deltas,h_combined)
            self.db += deltas
            S = sum([(delta*(v+v.T)).dot(h_combined) for delta,v in zip(deltas,self.V)])
            deltas = np.dot(self.W.T, deltas)+S
            self.backProp(node.left, deltas[:self.wvecDim])
            self.backProp(node.right, deltas[self.wvecDim:])
      

        
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

        print "Checking dW... (might take a while)"
        # for W,dW in zip(self.stack[1:],grad[1:]):
        for iii,(W,dW) in enumerate(zip(self.stack[1:],grad[1:])):
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        # print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
                        err1+=err
                        count+=1
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "err1: ",err1
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
                #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
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
    outputDim = 5

    nn = RNTN(wvecDim,outputDim,numW,mbSize=4)
    nn.initParams()

    mbData = train[:1]
    #cost, grad = nn.costAndGrad(mbData)

    print "Numerical gradient check..."
    nn.check_grad(mbData)






