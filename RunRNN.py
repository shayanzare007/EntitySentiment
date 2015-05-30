import sys, os
from numpy import *
import matplotlib.pyplot as plt

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

model = RNN_WEIGHTED(L0,w, U0=None, alpha=0.08, rseed=10, bptt=10)

Y_train = read_labels('y_train.csv')#'train_recu.csv'
Y_dev = read_labels('y_dev.csv')
print "Number of training samples",len(Y_train)

if len(X_dev)!= len(Y_dev):
  print "Sanity Check failed, len(X_dev)=",len(X_dev),"len(Y_dev)=",len(Y_dev)

##
# Pare down to a smaller dataset, for speed (optional)
ntrain = len(Y_train)
X = X_train[:ntrain]
Y = Y_train[:ntrain]
nepoch = 10
X = array(X)
Y = array(Y)
# ADD DUPLICATES
X,Y = preprocess_duplicates(X,Y,SENT_DIM,N_ASPECTS)

idxiter_random = random.permutation(len(Y))
for i in range(2,nepoch):
    permut = random.permutation(len(Y)) 
    idxiter_random = concatenate((idxiter_random,permut),axis=0)

idx_normal = range(len(Y))
curve = model.train_sgd(X,Y,idxiter_random,None,400,400) 

## Evaluate cross-entropy loss on the dev set,
## then convert to perplexity for your writeup
dev_loss = model.compute_mean_loss(X_dev, Y_dev)
print dev_loss

build_confusion_matrix(X_dev,Y_dev,model)
