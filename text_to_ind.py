

def text_to_ind(input_file,output_file):
    vocab = {}
    count = 0
    with open(input_file) as f:
        for i,line in enumerate(f):
            line = line.split(' ')
            for j,word in enumerate(line):
                word = word.rstrip()
                if word not in vocab:
                    vocab[word] = count
                    count += 1
    output = open(output_file,"w")
    with open(input_file) as f:
        for i,line in enumerate(f):
            line = line.split(' ')
            for j,word in enumerate(line):
                word = word.rstrip()
                if j==len(line)-1:
                    output.write(str(vocab[word]))
                else:
                    output.write(str(vocab[word])+',')
            output.write('\n')

text_to_ind('example_data/text_recu.txt','example_data/testing_parser')
