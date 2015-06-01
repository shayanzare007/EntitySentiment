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
nepoch = 50
X = array(X)
Y = array(Y)
# ADD DUPLICATES
X,Y = preprocess_duplicates(X,Y,SENT_DIM,N_ASPECTS)

model = RNN_SIMPLE(L0, U0=None, alpha=0.05, rseed=10, bptt=10)
idxiter_random = random.permutation(len(Y))
dev_costs_RNN = zeros(50)
train_costs_RNN = zeros(50)
for i in range(nepoch):
    curve = model.train_sgd(X,Y,idxiter_random,None,400,400) 
    dev_costs_RNN[i] = model.compute_mean_loss(X_dev, Y_dev)
    train_costs_RNN[i] = curve[-1][1]

print train_costs_RNN
print dev_costs_RNN

model = BRNN(L0, U0=None, alpha=0.05, rseed=10, bptt=10)
idxiter_random = random.permutation(len(Y))
dev_costs_BRNN = zeros(50)
train_costs_BRNN = zeros(50)
for i in range(nepoch):
    curve = model.train_sgd(X,Y,idxiter_random,None,400,400) 
    dev_costs_BRNN[i] = model.compute_mean_loss(X_dev, Y_dev)
    train_costs_BRNN[i] = curve[-1][1]

print train_costs_BRNN
print dev_costs_BRNN

plt.figure(1)
plt.plot(range(50),train_costs_RNN,label='RNN Train cost')
plt.plot(range(50),dev_costs_RNN,label='RNN Dev cost')
plt.plot(range(50),train_costs_BRNN,label='BRNN Train cost')
plt.plot(range(50),dev_costs_BRNN,label='BRNN dev cost')
plt.legend()
plt.show()
