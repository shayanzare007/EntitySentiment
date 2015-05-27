import sys, os
from numpy import *
from matplotlib.pyplot import *

from rnn_simple import RNN_SIMPLE
from brnn import BRNN
from data_utils import utils as du
import pandas as pd

N_ASPECTS = 5
SENT_DIM = 5

def read_labels(filename):
    training_set = []
    with open(filename,'rU') as f:
        for i,line in enumerate(f):
            line=line.rstrip().split(',')
            try:
                current=[int(x) for x in line]
                training_set.append(current)
            except:
                print "Error, number",i
      
    return training_set

def read_data(filename):
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
        X_train.append(array(sentence))

    print "File successfully read"
    f.close()
    return X_train

def make_sentiment_idx(y_hat):
    """
    Transforms one hot vectors of ys and y_hat into sentiments between -5 and 5
    """
    sentiments = []
    sent_dim = SENT_DIM
    for i in range(N_ASPECTS):
        current_sentiment = argmax(y_hat[i*SENT_DIM:(i+1)*SENT_DIM])-floor(SENT_DIM/2)
        sentiments.append(current_sentiment)
    return sentiments


def build_confusion_matrix(X,Y,model):
    conf_arr = np.zeros((SENT_DIM,SENT_DIM))
    for i,xs in enumerate(X):
        y = make_sentiment_idx(Y[i])
        y_hat = make_sentiment_idx(model.predict(xs))
        print y
        print y_hat
        print "\n \n"
        for t in range(len(y)):
            true_label=y[t]
            guessed_label=y_hat[t]
        
            conf_arr[true_label,guessed_label]+=1
    print conf_arr
    makeconf(conf_arr)

def makeconf(conf_arr):
    # makes a confusion matrix plot when provided a matrix conf_arr
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if a!=0:
                tmp_arr.append(float(j)/float(a))
            else:
                tmp_arr.append(0)
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    indexs = '0123456789'
    plt.xticks(range(width), indexs[:width])
    plt.yticks(range(height), indexs[:height])
    # you can save the figure here with:
    # plt.savefig("pathname/image.png")

    plt.show()

# Load the vocabulary

vocab = pd.read_table("worddic.txt",header=None,sep="\s+",index_col=0)

# Choose how many top words to keep
vocabsize = 60000
#vocabsize = 58868 #remove for implemenation
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

##############

filename_train = 'x_train.txt'#'reviews_plain.txt'
filename_dev = 'x_dev.txt'
X_train = read_data(filename_train)
X_dev = read_data(filename_dev)


hdim = 100 # dimension of hidden layer = dimension of word vectors
random.seed(10)
L0 = zeros((vocabsize, hdim)) # replace with random init, 
                              # or do in RNNLM.__init__()
model = RNN_SIMPLE(L0, U0=None, alpha=0.08, rseed=10, bptt=5)

Y_train = read_labels('y_train.csv')#'train_recu.csv'
Y_dev = read_labels('y_dev.csv')
print len(Y_train)

if len(X_dev)!= len(Y_dev):
  print "Sanity Check failed, len(X_dev)=",len(X_dev),"len(Y_dev)=",len(Y_dev)

##
# Pare down to a smaller dataset, for speed (optional)
ntrain = len(Y_train)
X = X_train[:ntrain]
Y = Y_train[:ntrain]
nepoch = 5
X = array(X)
Y = array(Y)

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
