##
# Miscellaneous helper functions
##
from __future__ import division
from numpy import *
import random
import copy

def random_weight_matrix(m, n):
    epsilon = sqrt(6)/(sqrt(m+n))
    A0 = random.uniform(-epsilon,epsilon,size=(m,n))
    assert(A0.shape == (m,n))
    return A0

def compute_entropy(temp_counter,dim_sent):
    # if one of the entries is zero we return 0
    if min(temp_counter) <= 0:
        return 0
    all_sent = sum(temp_counter)
    probs = [x/all_sent for x in temp_counter]
    ent = [-p*log2(p) for p in probs]
    return sum(ent)

def choose_best(Y,current_counters,cand,dim_sent):
    best = cand[0]
    entropies = zeros(len(cand))
    for i,candidate in enumerate(cand):
        print candidate
        current_y = Y[candidate]
        temp_counter = copy.deepcopy(current_counters)
        temp_counter[current_y+int(floor(dim_sent/2))] += 1
        entropies[i] = compute_entropy(temp_counter,dim_sent)
    best = cand[argmax(entropies)]
    return best

def create_minibatches(Y,n_batches,size_batches,n_candidates = 3,replacement = False, dim_sent =3,n_aspect=5):
    if replacement==False and n_batches*size_batches>len(Y):
        print 'Error: cannot create minibatches larger than data'
        return None

    Y_tr = copy.deepcopy(Y)
    y_len = len(Y[0])
    batches = []
    for i in range(n_batches):
        current_batch = []
        current_counters = [0]*dim_sent
        for j in range(size_batches):
            cand = random.sample(range(0,len(Y_tr)),n_candidates)
            best_cand = choose_best(Y_tr,current_counters,cand,dim_sent)
            current_batch.append(best_cand)
            for k in range(y_len):
                best_cand_y = Y[best_cand]
                current_counters[best_cand_y[k]+int(floor(dim_sent/2))]+=1
            if replacement == False:
                del Y_tr[best_cand]
        batches.append(current_batch)
    return batches

