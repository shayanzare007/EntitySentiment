from sys import argv

#script, filename = argv
filename = 'text_new.txt'
filename1 = 'text_recu.txt'

#f = open(filename1,'r')
print "Opening the file..."
#target = open(filename, 'w')


f = open(filename1)
f1 = open(filename,'a')

count = 11856
for line in f.readlines():
	f1.write(str(count))
	f1.write('\t')
	f1.write(line)
	count +=1



print "And finally, we close it."
f1.close()
f.close()
