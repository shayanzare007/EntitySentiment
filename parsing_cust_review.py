# Parsing customer review data

import re
import csv

ASPECT = ['player','sound','battery','price','software']

def parse(filename,output_filename):
    #open(output_filename, "w")
    features = dict()
    with open(filename) as f:
        for i,line in enumerate(f):
            if line[:3] == '[t]':
                # This is the title of the review
                continue
            try:
                line = line.split('##')
                entity_sentiment = line[0]
                sentence = line[1]
                ent_sent_dict = extract_entity_sentiment(entity_sentiment)
                for ent in ent_sent_dict:
                    if ent in features:
                        features[ent]=features[ent]+1
                    else:
                        features[ent]=1
                write_training(ent_sent_dict,sentence,output_filename)
            except:
                continue
    print features

            
def extract_entity_sentiment(text):
    ent_sent = {}
    if len(text)>0:
        entity_sent = text.split(',')
        n_ent = len(entity_sent)
        for j,current in enumerate(entity_sent):
            try:
                entity = current.partition('[')[0]
                score = current.split('[')[1][:-1]
                names=entity.split(' ')
                ent_sent[entity] = score
            except:
                continue 
    return ent_sent

def write_training(ent_sent_dict,sentence,output_filename):
    with open(output_filename, "a") as f:
        for i,word in enumerate(sentence.rstrip().split(' ')):
            if word in ent_sent_dict:
                f.write(word+',1,'+ent_sent_dict[word]+'\n')
            else:
                f.write(word+',0,0\n')


def format_recurrent(filename,output_filename_text='text_recu',output_feat = 'train_recu'):
    open(output_filename_text, "w")
    open(output_feat,"w")
    features = dict()
    with open(filename) as f:
        for i,line in enumerate(f):
            if line[:3] == '[t]':
                # This is the title of the review
                continue
            #try:
            line = line.split('##')
            entity_sentiment = line[0]
            sentence = line[1]
            ent_sent_dict = extract_entity_sentiment(entity_sentiment)
            sent_vect = build_sent(ent_sent_dict)
            print sentence
            #print ent_sent_dict
            write_sent(sent_vect,output_feat)
            write_sentence(sentence,output_filename_text)

            #except:
            #    print 'error'
            #    continue
    print features

def write_sentence(sentence, output_filename_text):
    with open(output_filename_text,"a") as f:
        f.write(sentence.rstrip() + '\n')

def build_sent(ent_sent_dict):
    n_asp = len(ASPECT)
    sent_vector=[]
    for i,aspect in enumerate(ASPECT):
        current = zerolistmaker(11)
        if aspect in ent_sent_dict:
            current[convert_to_int(ent_sent_dict[aspect])+5] = 1
        else:
            current[5] = 1
        sent_vector.extend(current)
    return sent_vector

def write_sent(sent_vect,output_feat):
    n = len(sent_vect)
    with open(output_feat,"a") as output:
        for i in range(0,n-1):
            output.write(str(sent_vect[i])+',')
        output.write(str(sent_vect[n-1])+'\n')

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def convert_to_int(sentiment):
    print sentiment
    if sentiment[0] == '+':
        return int(sentiment[1])
    elif sentiment[0] == '-':
        return -int(sentiment[1])
    else:
        print error
        return int(sentiment)