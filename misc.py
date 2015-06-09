# Miscellaneous functions for main
from __future__ import division
from numpy import *
from random import sample
import pdb
import copy
import sys, os
import matplotlib.pyplot as plt

N_ASPECTS = 5
SENT_DIM = 3

def random_weight_matrix(m, n):
    epsilon = sqrt(6)/(sqrt(m+n))
    A0 = random.uniform(-epsilon,epsilon,size=(m,n))
    #assert(A0.shape == (m,n))
    return A0

def compute_entropy(temp_counter,dim_sent):
    # if one of the entries is zero we return 0
    if min(temp_counter) <= 0:
        return 0
    all_sent = sum(temp_counter)
    probs = [x/all_sent for x in temp_counter]
    ent = [-p*log2(p) for p in probs]
    return sum(ent)

def choose_best(Y,current_counters,cand,dim_sent,n_aspect):
    best = cand[0]
    entropies = zeros(len(cand))
    for i,candidate in enumerate(cand):
        #print candidate
        current_y_count = count_current(Y[candidate],dim_sent,n_aspect)
        temp_counter = copy.deepcopy(current_counters)+current_y_count
        entropies[i] = compute_entropy(temp_counter,dim_sent)
    best = cand[argmax(entropies)]
    return best

def preprocess_duplicates(X,Y,dim_sent,n_aspect):
    processed_Y = list(copy.deepcopy(Y))
    processed_X = list(copy.deepcopy(X))
    for i,example in enumerate(Y):
        sent_count = count_current(example,dim_sent,n_aspect)
        n_dupl = sum(sent_count) - sent_count[int(floor(dim_sent/2))]
        for j in range(n_dupl):
            processed_X.append(X[i])
            processed_Y.append(Y[i])
    return array(processed_X),array(processed_Y)


def create_minibatches(Y,n_batches,size_batches=10,n_candidates = 5,replacement = False, dim_sent =3,n_aspect=5):
    if replacement==False and n_batches*size_batches>len(Y):
        print 'Error: cannot create minibatches larger than data'
        return None

    Y_tr = copy.deepcopy(Y)
    
    batches = []
    for i in range(n_batches):
        current_batch = []
        current_counters = zeros(dim_sent)
        for j in range(size_batches):
            cand = sample(range(0,len(Y_tr)),n_candidates)
            best_cand = choose_best(Y_tr,current_counters,cand,dim_sent,n_aspect)
            current_batch.append(best_cand)
            current_counters = current_counters + count_current(Y_tr[best_cand],dim_sent,n_aspect)
            if replacement == False:
                del Y_tr[best_cand]
        batches.extend(current_batch)
    return batches

def count_current(y,dim_sent,n_aspect):
    y = array(y).reshape((n_aspect,dim_sent))
    return sum(y,axis=0)

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

def read_data(filename,word_to_num):
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
    conf_arr = zeros((SENT_DIM,SENT_DIM))
    for i,xs in enumerate(X):
        y = make_sentiment_idx(Y[i])
        y_hat = make_sentiment_idx(model.predict(xs))
<<<<<<< HEAD
=======

        #print y
        #print y_hat
        #print "\n \n"
>>>>>>> fc44c6ed5cd7354852b4e6888885fc5914ffb92f
        for t in range(len(y)):
            true_label=y[t]+floor(SENT_DIM/2)
            guessed_label=y_hat[t]+floor(SENT_DIM/2)
        
            conf_arr[true_label,guessed_label]+=1
    print conf_arr
<<<<<<< HEAD
    return calc_accuracy(conf_arr)
=======
    calc_accuracy(conf_arr)
>>>>>>> fc44c6ed5cd7354852b4e6888885fc5914ffb92f
    #makeconf(conf_arr)

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
    res = ax.imshow(array(norm_conf), cmap=plt.cm.jet, 
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


def calc_accuracy(conf_arr):
    dim = shape(conf_arr)[0]
    acc = []
    rec = []
    pi_z = 0
    pi_n = 0
    rec_n = 0
    f1_mac = []
    for i in range(dim):
        acc_i = conf_arr[i,i]/sum(conf_arr[:,i])
        rec_i = conf_arr[i,i]/sum(conf_arr[i,:])
        acc.append(acc_i)
        rec.append(rec_i)
        f1_mac.append(2*acc_i*rec_i/(acc_i+rec_i))
        pi_z += conf_arr[i,i]
        pi_n += sum(conf_arr[:,i])
        rec_n += sum(conf_arr[i,:])
    acc_micro = pi_z/pi_n
    recall_micro = pi_z/rec_n
    f1_mic = 2*acc_micro*recall_micro/(acc_micro+recall_micro)
    f1_macro = sum(f1_mac)/dim
    print "Accucary single classes:",acc
    print "Recalls single classes:",rec
    print "F1 single classes:",f1_mac
    print "Accuracy averaged:",acc_micro
    print "Recall averaged",recall_micro
    print "F1 Score (macro):",f1_macro,"F1 Score (micro):",f1_mic

    return (acc_micro,recall_micro,f1_macro,f1_mic)


