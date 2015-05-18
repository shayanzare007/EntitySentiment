import sys, os
from numpy import *
from matplotlib.pyplot import *

from rnn_simple import RNNLM

# Gradient check on toy data, for speed
random.seed(10)
wv_dummy = random.randn(55,50)
model = RNNLM(L0 = wv_dummy, U0 = wv_dummy,
              alpha=0.005, rseed=10, bptt=4)
model.grad_check(array([1,2,3]), array([2,3,4]))

from data_utils import utils as du
import pandas as pd

# Load the vocabulary

vocab = pd.read_table("worddic.txt",header=None,sep="\s+",index_col=0)

# Choose how many top words to keep
#vocabsize = 2000
vocabsize = 58868 #remove for implemenation


num_to_word = dict(enumerate(vocab.index[:vocabsize]))

#word_to_num = du.invert_dict(num_to_word)
word_to_num = du.invert_dict(num_to_word)

#print word_to_num2

##############
filename = 'reviews_plain.txt'
print "Opening the file..."

X_train = []

f = open(filename,'r')
count = 0

for line in f.readlines():
    sentence = []
    line = line.strip()
    if not line: continue
    ret = line.split()
    for word in ret:
        word = word.strip()
        try:
            if word_to_num.get(word) is not None:
                sentence.append(word_to_num.get(word))
        except:
            count +=1
    X_train.append(sentence)

print "And finally, we close the file."
f.close()


hdim = 100 # dimension of hidden layer = dimension of word vectors
random.seed(10)
L0 = zeros((vocabsize, hdim)) # replace with random init, 
                              # or do in RNNLM.__init__()
model = RNNLM(L0, U0 = L0, alpha=0.08, rseed=10, bptt=2)

print len(Y_train)
# Gradient check is going to take a *long* time here
# since it's quadratic-time in the number of parameters.
# run at your own risk...
# model.grad_check(array([1,2,3]), array([2,3,4]))
#### YOUR CODE HERE ####

##
# Pare down to a smaller dataset, for speed (optional)
ntrain = len(Y_train)
ntrain = 5000
X = X_train[:ntrain]
Y = Y_train[:ntrain]
nepoch = 5

idxiter_random = random.permutation(len(Y))
for i in range(2,nepoch):
    permut = random.permutation(len(Y)) 
    idxiter_random = concatenate((idxiter_random,permut),axis=0)

curve = model.train_sgd(X,Y,idxiter_random,None,400,1000) #minibatch_sgd(X,Y,0.1)
#clf3.train_sgd(X_train,y_train,idxiter,None,400,1000)

## Evaluate cross-entropy loss on the dev set,
## then convert to perplexity for your writeup
dev_loss = model.compute_mean_loss(X_dev, Y_dev)
print dev_loss

## DO NOT CHANGE THIS CELL ##
# Report your numbers, after computing dev_loss above.
def adjust_loss(loss, funk):
    return (loss + funk * log(funk))/(1 - funk)
print "Unadjusted: %.03f" % exp(dev_loss)
print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost))

