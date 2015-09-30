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

vocab = pd.read_table("example_data/worddic.txt",header=None,sep="\s+",index_col=0)

n2w = dict(enumerate(vocab.index))
w2n = du.invert_dict(n2w)

vocabsize = len(w2n)

num2word =dict(enumerate(w2n))
word2num = du.invert_dict(num2word)
print "Number of unique words:",len(num2word)

##############

filename_train = 'x_train.txt'#'reviews_plain.txt'
filename_dev = 'x_dev.txt'
filename_test = 'x_test.txt'
X_train = read_data(filename_train,word2num)
X_dev = read_data(filename_dev,word2num)
X_test = read_data(filename_test,word2num)


hdim = 100 # dimension of hidden layer = dimension of word vectors
random.seed(10)
L0 = random_weight_matrix(vocabsize, hdim) # replace with random init, 
                              # or do in RNNLM.__init__()

Y_train = read_labels('example_data/y_train.csv')#'train_recu.csv'
Y_dev = read_labels('example_data/y_dev.csv')
Y_test = read_labels('example_data/y_test.csv')
print "Number of training samples",len(Y_train)

if len(X_dev)!= len(Y_dev):
  print "Sanity Check failed, len(X_dev)=",len(X_dev),"len(Y_dev)=",len(Y_dev)

##
# Pare down to a smaller dataset, for speed (optional)
ntrain = len(Y_train)
X = X_train[:ntrain]
Y = Y_train[:ntrain]
nepoch = 8
X = array(X)
Y = array(Y)
# ADD DUPLICATES
X,Y = preprocess_duplicates(X,Y,SENT_DIM,N_ASPECTS)

size = 10
number = 500
idxiter_random = create_minibatches(Y_train,number,size_batches=size,replacement = True)#random.permutation(len(Y))
for i in range(2,nepoch):
    permut = create_minibatches(Y_train,number,size_batches=size,replacement = True)#random.permutation(len(Y)) 
    idxiter_random = concatenate((idxiter_random,permut),axis=0)

idx_normal = range(len(Y))

score1 =[]

for i in range(1):
# create weight vectors  
    #wi = 1.35   
    #w2 =  3-(2*wi)                       
    w1 = [1.1, .8, 1.1]#[wi,w2,wi] # sum up to 3
    w = []
    for i in range(N_ASPECTS):
        w.extend(w1)

    model = BRNN(L0, U0=None, alpha=0.05, rseed=10, bptt=20)
    #print w
    curve = model.train_sgd(X,Y,idxiter_random,None,400,400) 
    score1.append(build_confusion_matrix(X_dev,Y_dev,model))
print score1

#model = RNN_SIMPLE(L0, U0=None, alpha=0.05, rseed=10, bptt=20)
#curve = model.train_sgd(X,Y,idxiter_random,None,400,400) 
## Evaluate cross-entropy loss on the dev set,
## then convert to perplexity for your writeup
dev_loss = model.compute_mean_loss(X_dev, Y_dev)
print dev_loss
test_loss = model.compute_mean_loss(X_test, Y_test)
print test_loss
build_confusion_matrix(X_test,Y_test,model)
