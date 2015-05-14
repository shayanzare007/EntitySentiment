from sys import argv

#script, filename = argv
filename = 'worddic.txt'
filename1 = 'text_recu.txt'

#f = open(filename1,'r')
print "Opening the file..."
#target = open(filename, 'w')


f = open(filename1)
f1 = open(filename,'a')

for line in f.readlines():
	line = line.strip()
	if not line: continue
	ret = line.split()
	for word in ret:
		f1.write(word)
		f1.write('\n')
	#f1.write(str(ret).strip())

print "And finally, we close it."
f1.close()
f.close()
