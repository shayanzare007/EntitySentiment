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
    Implements an RNN of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)] + b1)
    y_hat = softmax(U * h(t_final) + b_2)
    where y_hat is the sentiment vector associated with the sentence

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

    def __init__(self, L0, Dy=N_ASPECTS*SENT_DIM, U0=None,
                 alpha=0.005, rseed=10, bptt=5):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        self.ydim = Dy
        param_dims = dict(H_f = (self.hdim, self.hdim),
                          U = (self.ydim, 2*self.hdim),
                          H_b = (self.hdim, self.hdim),
                          b1_f = (self.hdim,),
                          b1_b = (self.hdim,),
                          b2 =(self.ydim,))
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
        self.params.H_f=random_weight_matrix(*self.params.H_f.shape)
        self.params.H_b=random_weight_matrix(*self.params.H_b.shape)
        if U0 is not None:
            self.params.U= U0.copy()
        else:
            self.params.U= random_weight_matrix(*self.params.U.shape)
        self.sparams.L = L0.copy()
        self.params.b1_f = zeros((self.hdim,))
        self.params.b1_b = zeros((self.hdim,))
        self.params.b2 = zeros((self.ydim,))


    def _acc_grads(self, xs, ys):
        # Forward propagation
        hs_f,hs_b,y_hat = self.forward_propagation(xs)       
        # backprop
        self.backprop(xs,ys,hs,y_hat)
                 
    def forward_propagation(self,xs):
        n_aspect = N_ASPECTS
        sent_dim = SENT_DIM
        ns = len(xs)
        hs_f = zeros((ns+1, self.hdim))
        hs_b = zeros((ns+1, self.hdim))
        for t in range(ns):
            hs_f[t] = sigmoid(self.params.H_f.dot(hs_f[t-1]) + self.sparams.L[xs[t]] + self.params.b1_f)
        h_f_final = hs_f[ns-1]
        inverted_xs = list(reversed(xs))
        for t in range(ns):
            hs_b[t] = sigmoid(self.params.H_b.dot(hs_b[t-1]) + self.sparams.L[inverted_xs[t]] + self.params.b1_b)
        h_b_final = hs_b[ns-1]

        z = self.params.U.dot(hstack([h_f_final,h_b_final])) + self.params.b2
        y_hat = []
        for i in range(n_aspect):
            current = z[sent_dim*i:sent_dim*(i+1)]
            y_hat.extend(softmax(current))
        return hs_f,hs_b,y_hat

    def backprop(self,xs,ys,hs_f,hs_b,y_hat):
        inverted_xs = list(reversed(xs))
        ns = len(xs)
        ht_f = hs_f[ns-1].reshape(len(h_final),1)
        ht_b = hs_b[ns-1].reshape(len(h_final),1)
        delta = (y_hat -ys)
        self.grads.b2 += delta
        delta = delta.reshape(len(ys),1)
        self.grads.U += delta.dot(ht.T)
         
        # H and L
        t = ns-1 # last t
        current_f = self.params.U.T.dot(delta)[:self.hdim]* ht_f * (1-ht_f)
        current_b = self.params.U.T.dot(delta)[self.hdim:]* ht_b * (1-ht_b) # the common part
        # update initial Hs
        prev_ht_f = hs_f[t-1].reshape(len(hs_f[t-1]),1)
        self.grads.H_f += current_f.dot(prev_ht_f.T)
        self.grads.b1_f += current_f.reshape((len(current_f),))

        prev_ht_b = hs_b[t-1].reshape(len(hs_b[t-1]),1)
        self.grads.H_b += current_b.dot(prev_ht_b.T)
        self.grads.b1_b += current_b.reshape((len(current_b),))

        # update initial L
        xt = make_onehot(xs[t],self.vdim).reshape(self.vdim,1)
        self.sgrads.L[xs[t]] = xt.dot(current_f.T)[xs[t]]
        inv_xt = make_onehot(inverted_xs[t],self.vdim).reshape(self.vdim,1)
        self.sgrads.L[inverted_xs[t]] = inv_xt.dot(current_b.T)[inverted_xs[t]]

        # update the rest
        for i in range(1,self.bptt):
            if t<i: # so that h[-2] doesn't return anything
                continue
            ht_f_i = hs_f[t-i].reshape(len(hs_f[t-i]),1)
            prev_ht_f_i = hs_f[t-i-1].reshape(len(hs_f[t-i-1]),1)
            current_f = self.params.H_f.T.dot(current_f)*ht_f_i*(1-ht_f_i)
            self.grads.H_f += current_f.dot(prev_ht_i.T)
            self.grads.b1_f += current_f.reshape((len(current_b),))

            ht_b_i = hs_b[t-i].reshape(len(hs_b[t-i]),1)
            prev_ht_b_i = hs_b[t-i-1].reshape(len(hs_b[t-i-1]),1)
            current_b = self.params.H_b.T.dot(current_b)*ht_b_i*(1-ht_b_i)
            self.grads.H_b += current_b.dot(prev_ht_b_i.T)
            self.grads.b1_b += current_b.reshape((len(current_b),))

            prev_xt = make_onehot(xs[t-i],self.vdim).reshape(self.vdim,1)
            self.sgrads.L[xs[t-i]] = prev_xt.dot(current_f.T)[xs[t-i]]
            prev_inv_xt = make_onehot(inverted_xs[t-i],self.vdim).reshape(self.vdim,1)
            self.sgrads.L[inverted_xs[t-i]] = prev_inv_xt.dot(current_b.T)[inverted_xs[t-i]]


    def compute_seq_loss(self, xs, ys):
        J = 0
        y_hat = self.predict(xs)
        J =- sum(array(ys).reshape(len(ys),1)*log(array(y_hat).reshape(len(y_hat),1)))

        #### END YOUR CODE ####
        return J

    def predict(self, xs):
        hs_f,hs_b,y_hat = self.forward_propagation(xs)
        return y_hat

    def compute_loss(self, X, Y):
        #print X.shape
        if len(X[0])==1: # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])


    def compute_mean_loss(self, X, Y):
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)






