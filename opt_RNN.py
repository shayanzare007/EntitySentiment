import sys, os
from numpy import *
import matplotlib.pyplot as plt
import pdb
from rnn_simple import RNN_SIMPLE
from brnn import BRNN
from brnn_weighted import BRNN_WEIGHTED
from rnn_weighted import RNN_WEIGHTED
from data_utils import utils as du
import pandas as pd
from misc import *
N_ASPECTS = 5
SENT_DIM = 3

# Load the vocabulary

vocab = pd.read_table("worddic.txt",header=None,sep="\s+",index_col=0)

n2w = dict(enumerate(vocab.index))
w2n = du.invert_dict(n2w)

vocabsize = len(w2n)

num2word =dict(enumerate(w2n))
word2num = du.invert_dict(num2word)
print "Number of unique words:",len(num2word)

##############

filename_train = 'x_train.txt'#'reviews_plain.txt'
filename_dev = 'x_dev.txt'
X_train = read_data(filename_train,word2num)
X_dev = read_data(filename_dev,word2num)


hdim = 100 # dimension of hidden layer = dimension of word vectors
random.seed(10)
L0 = random_weight_matrix(vocabsize, hdim) # replace with random init, 
                              # or do in RNNLM.__init__()
# create weight vectors                             
w1 = [1.2,.6,1.2] # sum up to 3
w = []
for i in range(N_ASPECTS):
    w.extend(w1)

Y_train = read_labels('y_train.csv')#'train_recu.csv'
Y_dev = read_labels('y_dev.csv')
print "Number of training samples",len(Y_train)

if len(X_dev)!= len(Y_dev):
  print "Sanity Check failed, len(X_dev)=",len(X_dev),"len(Y_dev)=",len(Y_dev)

epoch = 50
scores_with_rep = zeros((5,5))
scores_without_rep = zeros((5,5))
for size_i in range(5):
    size = 5*(size_i+1)
    max_nb = len(Y_train)/size
    for number_i in range(5):
        number = 100*(number_i+1)
        print "Parameters: size: %f, nb batches %f " % (size, number_i)
        idx_mini_rep = create_minibatches(Y_train,number,size_batches=size,replacement = True)
        idx__minibatch_rep = idx_mini_rep
        for e in range(epoch):
            idx__minibatch_rep.append(idx_mini_rep)
        model = RNN_SIMPLE(L0, U0=None, alpha=0.05, rseed=10, bptt=20)
        curve = model.train_sgd(array(X_train),array(Y_train),idx__minibatch_rep,None,100,100)
        scores_with_rep[size_i,number_i,cand] = model.compute_mean_loss(X_dev, Y_dev)

        idx_mini_no_rep = create_minibatches(Y_train,number,size_batches=size,replacement = False)
        if idx_mini_no_rep == None: continue
        idx__minibatch_norep = idx_mini_no_rep
        for e in range(epoch):
            idx__minibatch_norep.append(idx_mini_no_rep)
        model = RNN_SIMPLE(L0, U0=None, alpha=0.05, rseed=10, bptt=20)
        curve = model.train_sgd(array(X_train),array(Y_train),idx__minibatch_norep,None,100,100)
        scores_without_rep[size_i,number_i,cand] = model.compute_mean_loss(X_dev, Y_dev)

print scores_with_rep
print scores_without_rep

print 'WITH REP, ARGMIN INDEX IS'
print argmin(scores_with_rep)
print'\n \n \n'
print 'WITHOUT REP, ARGMIN INDEX IS'
print argmin(scores_without_rep)


## Evaluate cross-entropy loss on the dev set,
## then convert to perplexity for your writeup
dev_loss = model.compute_mean_loss(X_dev, Y_dev)
print dev_loss

#build_confusion_matrix(X_dev,Y_dev,model)
