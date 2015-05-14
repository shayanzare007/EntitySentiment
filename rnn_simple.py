from numpy import *
import itertools
import time
import sys

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid, make_onehot
from nn.math import MultinomialSampler, multinomial_sample

N_ASPECTS = 5
SENT_DIM = 11
class RNNLM(NNBase):
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)])
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """

    def __init__(self, L0, Dy, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        self.ydim = Dy
        param_dims = dict(H = (self.hdim, self.hdim),
                          U = (self.ydim, self.hdim))
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####
        var = .1
        sigma = sqrt(var)
        from misc import random_weight_matrix
        random.seed(rseed)
        # Initialize word vectors
        self.bptt = bptt
        self.alpha = alpha
        self.params.H=random_weight_matrix(*self.params.H.shape)
        if U0 is not None:
            self.params.U= U0.copy()
        else:
            self.params.U= random_weight_matrix(*self.params.U.shape)
        self.sparams.L = L0.copy()
        #### END YOUR CODE ####


    def _acc_grads(self, xs, ys):
        """
        Accumulate gradients, given a pair of training sequences:
        xs = [<indices>] # input words
        ys = [<indices>] # output words (to predict)

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.H += (your gradient dJ/dH)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # update row

        Per the handout, you should:
            - make predictions by running forward in time
                through the entire input sequence
            - for *each* output word in ys, compute the
                gradients with respect to the cross-entropy
                loss for that output word
            - run backpropagation-through-time for self.bptt
                timesteps, storing grads in self.grads (for H)
                and self.sgrads (for L,U)

        You'll want to store your predictions \hat{y}(t)
        and the hidden layer values h(t) as you run forward,
        so that you can access them during backpropagation.

        At time 0, you should initialize the hidden layer to
        be a vector of zeros.
        """

        # Expect xs as list of indices
        ns = len(xs)
        n_aspect = N_ASPECTS
        sent_dim = SENT_DIM
        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))
        # y_hat

        #### YOUR CODE HERE ####

        ##
        # Forward propagation
        for t in range(ns):
            hs[t] = sigmoid(self.params.H.dot(hs[t-1]) + self.sparams.L[xs[t]])
        h_final = hs[ns-1]
        z = self.params.U.dot(h_final) 
        y_hat = []
        for i in range(n_aspect):
            current = z[sent_dim*i:sent_dim*(i+1)]
            y_hat.extend(softmax(current))

        ##
        # Backward propagation through time
        delta = (y_hat -ys).reshape(len(ys),1)
        ht = h_final.reshape(len(hs[t]),1)
        self.grads.U+=delta.dot(ht.T)
        #for t in range(ns):
            # U
            #ht = hs[t].reshape(len(hs[t]),1)
            #dt = delta[t].reshape(len(delta[t]),1)
            #self.grads.U += dt.dot(ht.T)
            
            # H and L
        t = ns-1 # last t
        current = self.params.U.T.dot(dt) * ht * (1-ht) # the common part
        prev_ht = hs[t-1].reshape(len(hs[t-1]),1)
        self.grads.H += current.dot(prev_ht.T)
        xt = make_onehot(xs[t],self.vdim).reshape(self.vdim,1)
        self.sgrads.L[xs[t]] = xt.dot(current.T)[xs[t]]
        for i in range(1,self.bptt):
            if t<i: # so that h[-2] doesn't return anything
                continue
            ht_i = hs[t-i].reshape(len(hs[t-i]),1)
            prev_ht_i = hs[t-i-1].reshape(len(hs[t-i-1]),1)
            current = self.params.H.T.dot(current)*ht_i*(1-ht_i)
            self.grads.H += current.dot(prev_ht_i.T)
                
            prev_xt = make_onehot(xs[t-i],self.vdim).reshape(self.vdim,1)
            self.sgrads.L[xs[t-i]] = prev_xt.dot(current.T)[xs[t-i]]
                 

        #### END YOUR CODE ####



    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        J = 0
        #### YOUR CODE HERE ####
        # hs[-1] = initial hidden state (zeros)
        ns = len(ys)
        hs = zeros((ns+1, self.hdim))

        for t in range(ns):
            hs[t] = sigmoid(self.params.H.dot(hs[t-1]) + self.sparams.L[xs[t]])
            #ps[t] = softmax(self.params.U.dot(hs[t]))
            #J -= log(ps[t][ys[t]])
        h_final = hs[ns-1]
        z = self.params.U.dot(h_final) 
        y_hat = []
        for i in range(n_aspect):
            current = z[sent_dim*i:sent_dim*(i+1)]
            y_hat.extend(softmax(current))
        J =- sum(ys.reshape(len(ys),1)*log(array(y_hat).reshape(len(y_hat),1)))

        #### END YOUR CODE ####
        return J


    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)



