from sys import argv
import re
import csv
import parsing_cust_review as parser
import numpy as np

#script, filename = argv
label_total_filename = 'example_data/total_data.csv'
label_train_filename = 'example_data/y_train.csv'
label_dev_filename = 'example_data/y_dev.csv'
label_test_filename = 'example_data/y_test.csv'

data_filename = 'example_data/total_sentences.csv'
x_train_filename = 'example_data/x_train.txt'
x_dev_filename = 'example_data/x_dev.txt'
x_test_filename = 'example_data/x_test.txt'

#f = open(filename1,'r')
print "Opening the file..."
#target = open(filename, 'w')

#label_total = open(label_total_filename)
label_train = open(label_train_filename,'a')
label_dev = open(label_dev_filename,'a')
label_test = open(label_test_filename,'a')

#data=open(data_filename)
x_train = open(x_train_filename,'a')
x_dev = open (x_dev_filename,'a')
x_test = open(x_test_filename,'a')

total = 7149
#total = 10

#tot_arr = np.arange(total)

tot_train=int(0.7*total)
tot_dev=int(0.5*(total-tot_train))
tot_test=total-tot_train-tot_dev

print "Split set into ",tot_train," training sampples, ",tot_dev," dev samples and ",tot_test," test samples."

#train_choice = np.random.choice(tot_arr,tot_train,replace=True)
#shuffle = np.random.shuffle(tot_arr)

#permut = np.random.permutation(total)

#print permut

with open(data_filename,'rb') as f:
	data = f.read().split('\n')
with open(label_total_filename,'rb') as f1:
	labels = f1.read().split('\n')

z = zip(data,labels)
np.random.shuffle(z)

dat,lab = zip(*z)

print "Sanity check",len(dat)
print "Sanity check",len(lab)

count=1
for line in dat:
	if count<=tot_train:
		x_train.write(line)
		x_train.write('\n')

	elif count<=tot_train+tot_dev:
		x_dev.write(line)
		x_dev.write('\n')
	else:
		x_test.write(line)
		x_test.write('\n')

	count+=1

count=1
for line in lab:
	if count<=tot_train:
		label_train.write(line)
		label_train.write('\n')
	elif count<=tot_train+tot_dev:
		label_dev.write(line)
		label_dev.write('\n')
	else:
		label_test.write(line)
		label_test.write('\n')
	count+=1

#label_total.close()
label_train.close()
label_dev.close()
label_test.close()

#data.close()
x_train.close()
x_dev.close()
x_test.close()