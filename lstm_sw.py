import numpy as np
import theano
import theano.tensor as T
from nn.math import softmax, sigmoid, make_onehot
from theano import pp
import pandas as pd
from data_utils import utils as du
from theano import ProfileMode
import theano.tensor.nnet as nn
from misc import *
from nn.base import NNBase

profmode = theano.ProfileMode(optimizer='fast_compile')

dtype=theano.config.floatX
N_ASPECTS = 5
N_SENTIMENTS = 3


# squashing of the gates should result in values between 0 and 1
# therefore we use the logistic function
sigma = lambda x: 1 / (1 + T.exp(-x))

# for the other activation function we use the tanh
act = T.tanh


def one_lstm_step(seq,h_tm1, c_tm1, coeff,W_ci,W_cf,W_co,W_xi, U_hi, b_i, W_xf, U_hf, b_f, W_xc, U_hc, b_c, W_xo, U_ho, b_o, W_hy, b_y, L0):

    a = N_ASPECTS
    s = N_SENTIMENTS
    n_y = N_ASPECTS*N_SENTIMENTS

    x_t = L0[seq,:]
    i_t = sigma(theano.dot(x_t, W_xi) + theano.dot(h_tm1, U_hi) + theano.dot(c_tm1, W_ci)+b_i) #+ sigma(theano.dot(L0[seq,:],W_xi))
    f_t = sigma(theano.dot(x_t, W_xf) + theano.dot(h_tm1, U_hf) + theano.dot(c_tm1, W_cf)+b_f)
    c_t = f_t * c_tm1 + i_t * act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, U_hc) + b_c) 
    o_t = sigma(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, U_ho) + theano.dot(c_t, W_co) + b_o)
    h_t = o_t * act(c_t)

    #y_t = theano.shared(np.zeros(15, dtype=dtype))

    def sm(x):
        xt = T.exp(x - T.max(x))
        return xt / T.sum(xt)

    y_t = theano.shared(np.zeros(0,dtype=dtype))

    w_temp = theano.dot(h_t, W_hy) + b_y
    
    for i in range(a):
        mult = np.zeros((n_y,3))
        mult[3*i,0]=1
        mult[3*i+1,1]=1
        mult[3*i+2,2]=1
        temp = theano.dot(w_temp,mult)
        temp = sm(temp)
        y_t = T.concatenate([y_t, temp],axis=0)

    return [h_t, c_t, y_t]


#TODO: Use a more appropriate initialization method
def sample_weights(sizeX, sizeY):
    values = np.ndarray([sizeX, sizeY], dtype=dtype)
    for dx in xrange(sizeX):
        vals = np.random.uniform(low=-1., high=1.,  size=(sizeY,))
        #vals_norm = np.sqrt((vals**2).sum())
        #vals = vals / vals_norm
        values[dx,:] = vals
    _,svs,_ = np.linalg.svd(values)
    #svs[0] is the largest singular value                      
    values = values / svs[0]
    return values  


class LSTM:
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

    def __init__(self, L,n_hidden,w=None, U0=None,
                 alpha=0.005, rseed=10, bptt=None):

    	Dy=N_ASPECTS*N_SENTIMENTS
    	self.n_y = Dy
    	if w==None:
    		self.w = theano.shared(np.ones(self.n_y, dtype=dtype))
    	else:
    		self.w = theano.shared(np.cast[dtype](np.array(w)))

    	self.n_in = L.shape[1] # wordvec_dim
        self.n_hidden = self.n_i = self.n_c = self.n_o = self.n_f = n_hidden
        
       # Initialize shared variables
        self.W_ci = theano.shared(sample_weights(self.n_c, self.n_i)) 
        self.W_cf = theano.shared(sample_weights(self.n_c, self.n_f))
        self.W_co = theano.shared(sample_weights(self.n_c, self.n_o))
        self.W_xi = theano.shared(sample_weights(self.n_in, self.n_i))
        self.U_hi = theano.shared(sample_weights(self.n_hidden, self.n_i))  
        self.b_i = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = self.n_i)))
        self.W_xf = theano.shared(sample_weights(self.n_in, self.n_f)) 
        self.U_hf = theano.shared(sample_weights(self.n_hidden, self.n_f))
        self.b_f = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = self.n_f)))
        self.W_xc = theano.shared(sample_weights(self.n_in, self.n_c))  
        self.U_hc = theano.shared(sample_weights(self.n_hidden, self.n_c))
        self.b_c = theano.shared(np.zeros(self.n_c, dtype=dtype)) 
        self.W_xo = theano.shared(sample_weights(self.n_in, self.n_o))
        self.U_ho = theano.shared(sample_weights(self.n_hidden, self.n_o))
        self.b_o = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = self.n_o)))
        self.W_hy = theano.shared(sample_weights(self.n_hidden, self.n_y))
        self.b_y = theano.shared(np.zeros(self.n_y, dtype=dtype))
        self.c0 = theano.shared(np.zeros(self.n_hidden, dtype=dtype))
        self.h0 = T.tanh(self.c0)

        self.L0 = theano.shared(L) #np.cast[dtype](np.random.uniform(-0.5,.5,size = ((vocabsize, n_i))))) # replace with random init, 
		                              # or do in RNNLM.__init__()

        self.params = [self.W_ci,self.W_cf,self.W_co,self.W_xi, self.U_hi, self.b_i, self.W_xf, self.U_hf, self.b_f, self.W_xc, self.U_hc, self.b_c, self.W_xo, self.U_ho, self.b_o, self.W_hy, self.b_y, self.c0, self.L0]

        print "Params initilized"


        self.coeff = T.vector(dtype=dtype)

        self.target = T.vector(dtype=dtype)

        [self.h_vals, _, self.y_vals], _ = theano.scan(fn=one_lstm_step, 
                                  sequences = [T.arange(self.coeff.shape[0])],#coeff,dict(input=coeff,taps=[0]),#],#dict(input=v,taps=[0 #dict(input=v, taps=[0]), 
                                  outputs_info = [self.h0, self.c0, None ], # corresponds to return type of fn
                                  non_sequences = [self.coeff,self.W_ci,self.W_cf,self.W_co,self.W_xi, self.U_hi, self.b_i, self.W_xf, self.U_hf, self.b_f, self.W_xc, self.U_hc, self.b_c, self.W_xo, self.U_ho, self.b_o, self.W_hy, self.b_y, self.L0] )

        self.cost = -(self.w * self.target * T.log(self.y_vals[-1])).sum()
       # self.cost = -(np.array(self.w).reshape(self.n_y,1)*T.log(self.y_vals[-1].reshape(self.n_y,1))).sum()
       # J =- sum(array(ys).reshape(len(ys),1)*log(array(y_hat).reshape(len(y_hat),1)))

        #self.cost = -(self.w * self.target * T.log(self.y_vals[-1])+self.w * (1.-self.target)*T.log(1.-self.y_vals[-1])).sum()

        lr = np.cast[dtype](.05)
        learning_rate = theano.shared(lr)
        self.gparams = []
        i = 1

        for param in self.params:
            gparam = T.grad(self.cost, param)
            print "Gradient building in progress, parameter#: ",i," / 19"
            i +=1
            self.gparams.append(gparam)

        self.updates=[]
        for param, gparam in zip(self.params, self.gparams):
            self.updates.append((param, param - gparam * learning_rate))

        self.learn_lstm = theano.function(inputs=[self.coeff, self.target],outputs=self.cost,updates=self.updates,mode=profmode)
        self.predictions = theano.function(inputs=[self.coeff], outputs=self.y_vals[-1])
        self.devcost = theano.function(inputs=[self.coeff,self.target], outputs=self.cost)
        print "Succesfully Finished Initialization, ignore warnings"

        

    def train_sgd(self,X_train,Y_train,idx_random=None,printevery=None,costevery=400):
        t =0
        train_errors = np.ndarray(len(idx_random))
        for x in idx_random:
            error = 0.
            i = X_train[x]
            o = Y_train[x]
            try:
                error = self.learn_lstm(i,o)
            except:
                print "ERROR during training"
            train_errors[t] = error
            if t % costevery==0:
                if t!=0:
                    print "Iteration:",t,", cost:",np.mean(train_errors[(t+1-costevery):(t+1)])
                else:
                    print "Initial iteration:,",t,", cost:",error
            t +=1
        return train_errors

    '''


    def make_sentiment_idx(self,y_hat):
	    """
	    Transforms one hot vectors of ys and y_hat into sentiments between -5 and 5
	    """
	    sentiments = []
	    SENT_DIM = N_SENTIMENTS
	    for i in range(N_ASPECTS):
	        current_sentiment = np.argmax(y_hat[i*SENT_DIM:(i+1)*SENT_DIM])-floor(SENT_DIM/2)
	        sentiments.append(current_sentiment)
	    return sentiments

    def build_confusion_matrix(self,X,Y):
        conf_arr = zeros((N_SENTIMENTS,N_SENTIMENTS))
        for i,xs in enumerate(X):
            y = make_sentiment_idx(Y[i])
            try:
                #print "trying prediction"
                pred = self.predictions(xs)
                print pred
                y_hat = make_sentiment_idx(pred)
                print y
                print y_hat
                print "\n \n"
                for t in range(len(y)):
                    true_label=y[t]+floor(N_SENTIMENTS/2)
                    guessed_label=y_hat[t]+floor(N_SENTIMENTS/2)
                    conf_arr[true_label,guessed_label]+=1
            except:
                print "error during prediction,",i  
        print conf_arr

    '''
    def predict(self,xs):
        try:
            p = self.predictions(xs)
        except:
            p = [0]*15
            print "errorstate"
        return p

    def devloss(self,X_dev,Y_dev):
        error = []
        for i in range(len(X_dev)):
            xs = X_dev[i]
            ys = Y_dev[i]
            try:
                error.append(self.devcost(xs,ys))
            except:
                continue
        return np.mean(error)