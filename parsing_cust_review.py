# Parsing customer review data

import re

def parse(filename,output_filename):
    #open(output_filename, "w")
    with open(filename) as f:
        for i,line in enumerate(f):
            if line[:3] == '[t]':
                # This is the title of the review
                continue
            line = line.split('##')
            entity_sentiment = line[0]
            sentence = line[1]
            ent_sent_dict = extract_entity_sentiment(entity_sentiment)
            write_training(ent_sent_dict,sentence,output_filename)

            



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
                for i,name in enumerate(names):
                    ent_sent[name] = score
                #ent_sent[entity] = score
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
