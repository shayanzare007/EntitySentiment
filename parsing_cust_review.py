# Parsing customer review data

import re
import csv
import math

#ASPECT = ['player','sound','battery','price','software'] MUSIC PLAYER
ASPECT = ['picture','price','battery','portability','features']
DIM_SENT = 3

vocabulary ={'software':'features','optical zoom':'features','use':'features','photos':'picture','size':'portability','ease of use':'features','pictures':'picture',
'zoom':'features','controls':'features','picture detail':'picture','buttons':'features','panorama mode':'features','TFT':'features',
'TFT screen':'features','picture quality':'picture','Small':'portability','compact':'portability','light':'portability',
'auto mode':'features','manual modes':'features','LCD':'features','macro mode':'features','dynamic range':'features',
'housing':'portability','small':'portability','images':'picture','pics':'picture','sharp':'picture','color balance':'picture',
'pictures of people':'picture','focus':'features','usability':'features','looks':'portability','weight':'portability','weighs':'portability',
'image quality':'picture','tiny':'portability','resolution':'picture','smaller':'portability','flash':'features',
'panorama setting':'features','fits':'portability','viewfinder':'features','screen':'features','functionality':'features',
'lcd':'features','autofocus':'features','shots':'picture', 'batteries':'battery','memory':'features',
'battery life':'battery'}
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
                entity = current.partition('[')[0].lower()
                if entity in vocabulary:
                    entity = vocabulary[entity]
                score = current.split('[')[1][:-1]
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
        current = zerolistmaker(DIM_SENT)
        print ent_sent_dict
        if aspect in ent_sent_dict:
            se = ent_sent_dict[aspect]
            idx = comp_sent(convert_to_int(se))+int(math.floor(DIM_SENT/2))
            current[idx] = 1
        else:
            current[int(math.floor(DIM_SENT/2))] = 1
        sent_vector.extend(current)
    print sent_vector
    return sent_vector

def write_sent(sent_vect,output_feat):
    n = len(sent_vect)
    with open(output_feat,"a") as output:
        for i in range(0,n-1):
            output.write(str(sent_vect[i])+',')
        output.write(str(sent_vect[n-1])+'\n')

def comp_sent(sent):
    if DIM_SENT == 5:
        if sent < -2:
            return -2
        elif sent == -2 or sent == -1 : 
            return -1
        elif sent == 1 or sent == 2:
            return 1
        elif sent > 2:
            return 2
        elif sent == 0:
            return 0
    elif DIM_SENT == 3:
        if sent<0:
            return -1
        elif sent > 0:
            return 1
        else:
            return 0
    else:
        return sent
    return 0
        
def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def convert_to_int(sentiment):
    if sentiment[0] == '+':
        return int(sentiment[1])
    elif sentiment[0] == '-':
        return -int(sentiment[1])
    else:
        return int(sentiment)
