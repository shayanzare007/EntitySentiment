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

#n_hidden = 50

# squashing of the gates should result in values between 0 and 1
# therefore we use the logistic function
sigma = lambda x: 1 / (1 + T.exp(-x))

# for the other activation function we use the tanh
act = T.tanh


def one_lstm_step(seq,seq2,h_tm1,h_tm2,c_tm1,c_tm2, coeff,coeff_rev,W_hy,b_y,W_xi,W_xi2, U_hi,U_hi2, b_i, b_i2, W_xf,W_xf2, U_hf,U_hf2, b_f,b_f2, W_xc,W_xc2, U_hc,U_hc2, b_c,b_c2, W_xo,W_xo2, U_ho,U_ho2, b_o,b_o2, L0):
                                                               
    a = N_ASPECTS
    s = N_SENTIMENTS
    n_y = N_ASPECTS*N_SENTIMENTS

    x_t = L0[seq,:]
    i_t = sigma(theano.dot(x_t, W_xi) + theano.dot(h_tm1, U_hi) + b_i) #+ sigma(theano.dot(L0[seq,:],W_xi))
    f_t = sigma(theano.dot(x_t, W_xf) + theano.dot(h_tm1, U_hf) + b_f)
    c_t = f_t * c_tm1 + i_t * act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, U_hc) + b_c) 
    o_t = sigma(theano.dot(x_t, W_xo)+ theano.dot(h_tm1, U_ho)  + b_o)
    h_t = o_t * act(c_t)

    x_t2 = L0[seq2,:]
    i_t2 = sigma(theano.dot(x_t2, W_xi2) + theano.dot(h_tm2, U_hi2) + b_i2) #+ sigma(theano.dot(L0[seq,:],W_xi))
    f_t2 = sigma(theano.dot(x_t2, W_xf2) + theano.dot(h_tm2, U_hf2) + b_f2)
    c_t2 = f_t2 * c_tm2 + i_t2 * act(theano.dot(x_t2, W_xc2) + theano.dot(h_tm2, U_hc2) + b_c2) 
    o_t2 = sigma(theano.dot(x_t2, W_xo2)+ theano.dot(h_tm2, U_ho2)  + b_o2)
    h_t2 = o_t2 * act(c_t2)

    h_conc = T.concatenate([h_t,h_t2],axis=0)

    def sm(x):
        xt = T.exp(x - T.max(x))
        return xt / T.sum(xt)

    y_t = theano.shared(np.zeros(0,dtype=dtype))

    w_temp = theano.dot(h_conc, W_hy) + b_y
    
    for i in range(a):
        mult = np.zeros((n_y,3))
        mult[3*i,0]=1
        mult[3*i+1,1]=1
        mult[3*i+2,2]=1
        temp = theano.dot(w_temp,mult)
        temp = sm(temp)
        y_t = T.concatenate([y_t, temp],axis=0)
    
    return [h_t,h_t2, c_t,c_t2,y_t]


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


class BLSTM:
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

    def __init__(self, L,hdim,w=None, U0=None,
                 alpha=0.005, rseed=10, bptt=None):

         
    	Dy=N_ASPECTS*N_SENTIMENTS
    	self.n_y = Dy
    	if w==None:
    		self.w = theano.shared(np.ones(self.n_y, dtype=dtype))
    	else:
    		self.w = theano.shared(np.cast[dtype](np.array(w)))

    	self.n_in = L.shape[1] # wordvec_dim
        self.n_hidden = self.n_i = self.n_c = self.n_o = self.n_f = hdim
        
       # Initialize shared variables
        self.W_xi_f = theano.shared(sample_weights(self.n_in, self.n_i))
        self.W_xi_b = theano.shared(sample_weights(self.n_in, self.n_i))
        self.U_hi_f = theano.shared(sample_weights(self.n_hidden, self.n_i)) 
        self.U_hi_b = theano.shared(sample_weights(self.n_hidden, self.n_i))  
        self.b_i_f = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = self.n_i)))
        self.b_i_b = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = self.n_i)))
        self.W_xf_f = theano.shared(sample_weights(self.n_in, self.n_f)) 
        self.W_xf_b = theano.shared(sample_weights(self.n_in, self.n_f)) 
        self.U_hf_f = theano.shared(sample_weights(self.n_hidden, self.n_f))
        self.U_hf_b = theano.shared(sample_weights(self.n_hidden, self.n_f))
        self.b_f_f = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = self.n_f)))
        self.b_f_b = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = self.n_f)))
        self.W_xc_f = theano.shared(sample_weights(self.n_in, self.n_c))  
        self.W_xc_b = theano.shared(sample_weights(self.n_in, self.n_c)) 
        self.U_hc_f = theano.shared(sample_weights(self.n_hidden, self.n_c))
        self.U_hc_b = theano.shared(sample_weights(self.n_hidden, self.n_c))
        self.b_c_f = theano.shared(np.zeros(self.n_c, dtype=dtype)) 
        self.b_c_b = theano.shared(np.zeros(self.n_c, dtype=dtype))
        self.W_xo_f = theano.shared(sample_weights(self.n_in, self.n_o))
        self.W_xo_b = theano.shared(sample_weights(self.n_in, self.n_o))
        self.U_ho_f = theano.shared(sample_weights(self.n_hidden, self.n_o))
        self.U_ho_b = theano.shared(sample_weights(self.n_hidden, self.n_o))
        self.b_o_f = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = self.n_o)))
        self.b_o_b = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = self.n_o)))
        self.W_hy = theano.shared(sample_weights(2*self.n_hidden, self.n_y))
        self.b_y = theano.shared(np.zeros(self.n_y, dtype=dtype))
        self.c0_f = theano.shared(np.zeros(self.n_hidden, dtype=dtype))
        self.c0_b = theano.shared(np.zeros(self.n_hidden, dtype=dtype))
        self.h0_f = T.tanh(self.c0_f)
        self.h0_b = T.tanh(self.c0_b)

        self.L0 = theano.shared(L) #np.cast[dtype](np.random.uniform(-0.5,.5,size = ((vocabsize, n_i))))) # replace with random init, 
		                              # or do in RNNLM.__init__()

        self.params = [self.W_xi_f, self.W_xi_b,self.U_hi_f, self.U_hi_b,self.b_i_f,self.b_i_b, self.W_xf_f,self.W_xf_b, self.U_hf_f,self.U_hf_b, self.b_f_f,self.b_f_b, self.W_xc_f, self.W_xc_b,self.U_hc_f,self.U_hc_b, self.b_c_f,self.b_c_b, self.W_xo_f,self.W_xo_b, self.U_ho_f, self.U_ho_b,self.b_o_f,self.b_o_b, self.W_hy, self.b_y, self.c0_f,self.c0_b, self.L0]

        print "Params initilized"

        self.coeff = T.vector(dtype=dtype)
        self.coeff_rev = T.vector(dtype=dtype)

        self.target = T.vector(dtype=dtype)

        [self.h_vals_for,self.h_vals_back,_, _,self.y_vals], _ = theano.scan(fn=one_lstm_step, 
                                  sequences = [T.arange(self.coeff.shape[0]),T.arange(self.coeff_rev.shape[0])],#coeff,dict(input=coeff,taps=[0]),#],#dict(input=v,taps=[0 #dict(input=v, taps=[0]), 
                                  outputs_info = [self.h0_f,self.h0_b, self.c0_f,self.c0_b,None], # corresponds to return type of fn
                                  non_sequences = [self.coeff,self.coeff_rev,self.W_hy,self.b_y,self.W_xi_f,self.W_xi_b, self.U_hi_f,self.U_hi_b, self.b_i_f,self.b_i_b, self.W_xf_f,  self.W_xf_b,self.U_hf_f,self.U_hf_b, self.b_f_f,self.b_f_b, self.W_xc_f, self.W_xc_b,self.U_hc_f,self.U_hc_b, self.b_c_f,self.b_c_b, self.W_xo_f, self.W_xo_b, self.U_ho_f,self.U_ho_b, self.b_o_f,self.b_o_b, self.L0] )

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
            print "Gradient building in progress, parameter#: ",i," / 29"
            i +=1
            self.gparams.append(gparam)

        self.updates=[]
        for param, gparam in zip(self.params, self.gparams):
            self.updates.append((param, param - gparam * learning_rate))

        self.learn_lstm = theano.function(inputs=[self.coeff,self.coeff_rev, self.target],outputs=self.cost,updates=self.updates,mode=profmode)
        #self.learn_lstm_back = theano.function(inputs=[self.coeff_rev, self.target],outputs=self.cost,updates=self.updates,mode=profmode)
        self.predictions = theano.function(inputs = [self.coeff,self.coeff_rev], outputs = self.y_vals[-1])
        self.devcost = theano.function(inputs=[self.coeff,self.coeff_rev,self.target], outputs=self.cost)
        print "Succesfully Finished Initialization, ignore warnings"        

    '''
    def concat(self,i,i_reverse,o):
        h1 = theano.function(inputs=[self.coeff, self.target],outputs=self.cost,updates=self.updates,mode=profmode)
        h2 = theano.function(inputs=[self.coeff_rev, self.target],outputs=self.cost,updates=self.updates,mode=profmode)

        def sm(x):
            xt = T.exp(x - T.max(x))
            return xt / T.sum(xt)

        y_t = theano.shared(np.zeros(0,dtype=dtype))

        h_conc = T.concatenate([self.h_vals_for[-1],self.h_vals_back[-1]],axis=0)
        w_temp = theano.dot(h_conc, self.W_hy) + self.b_y

        #z = self.params.U.dot(hstack([h_f_final,h_b_final])) + self.params.b2
        y_hat = []

        for i in range(N_ASPECTS):
            mult = np.zeros((N_SENTIMENTS*N_ASPECTS,3))
            mult[3*i,0]=1
            mult[3*i+1,1]=1
            mult[3*i+2,2]=1
            temp = theano.dot(w_temp,mult)
            temp = sm(temp)
            y_t = T.concatenate([y_t, temp],axis=0)
    
        self.cost = -(self.w * self.target * T.log(y_t)).sum()
    '''


    def train_sgd(self,X_train,Y_train,idx_random=None,printevery=None,costevery=400):
        t =0
        train_errors = np.ndarray(len(idx_random))
        print len(idx_random)
        for x in idx_random:
            error = 0.
            i = X_train[x]
            i_reverse = i[::-1]
            o = Y_train[x]
            try:
                error = self.learn_lstm(i,i_reverse,o)
            except:
                print "ERROR during training"
            train_errors[t] = error
            if t % costevery==0:
                if t!=0:
                    print "Iteration:",t,", cost:",np.mean(train_errors[(t+1-costevery):(t+1)])
                else:
                    print "Initial iteration:,",t,", cost:",error
            t +=1
        print "train sgd finished"
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
                pred = predict(xs)
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
            xs_reversed=xs[::-1]
            p = self.predictions(xs,xs_reversed)
        except:
            p = [0]*15
        return p
    
    def devloss(self,X_dev,Y_dev):
        error = []
        for i in range(len(X_dev)):
            xs = X_dev[i]
            xs_reversed = xs[::-1]
            ys = Y_dev[i]
            try:
                error.append(self.devcost(xs,xs_reversed,ys))
            except: 
                continue
        return np.mean(error)



