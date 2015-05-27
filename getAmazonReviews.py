import json
import re
from pprint import pprint

#ASPECT = ['player','sound','battery','price','software']

ASPECT = ['picture','price','battery','portability','features']

INPUT_PATH = 'data/cameras_mined/SonyA3000.txt'
OUTPUT_PATH = 'data/cameras_mined/SonyA3000_labeled.txt'


reviews = []
with open(INPUT_PATH) as json_data:
    for i, json_obj in enumerate(json_data):
        d = json.loads(json_obj)
        reviews.append(d['summary'])
        reviews.append(d['reviewText'])

open(OUTPUT_PATH,"w").close() # EMPTYING FILE
for i,review in enumerate(reviews):
    review = re.split('[.|!]',review)
    for k,sentence in enumerate(review):
        if len(sentence) < 2 : continue
        with open(OUTPUT_PATH,"a") as out:
            for j,asp in enumerate(ASPECT):
                if j < (len(ASPECT)-1):
                    out.write(str(asp)+'[0],')
                else:
                    out.write(str(asp)+'[0]##'+sentence.strip()+'\n')

