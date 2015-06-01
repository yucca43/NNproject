import numpy as np
import csv
def transformToRNN(filename = 'tweeti.b.dist.parsed'):
	outfile = 'trees/temp.txt'
	if filename == 'data/b.train.preprocessed.utf8.parsed':
		outfile = 'trees/train.txt'
	elif filename == 'data/b.dev.preprocessed.utf8.parsed':
		outfile = 'trees/dev.txt'
	elif filename == 'data/b.test.parsed':
		outfile = 'trees/test.txt'
	labelMap = {"neutral":1,"positive":2,"negative":0,"objective":1,"objective-OR-neutral":1,
	"neutral\"\"":1,"positive\"\"":2,"negative\"\"":0,"objective\"\"":1,"objective-OR-neutral\"\"":1}
	reader = csv.reader(open(filename),delimiter="\t")
	writer = open(outfile,'w')
	for line in reader:
		treeString = line[3][6:-1]
		index = treeString.index(' ')
		writer.write("("+str(labelMap[line[2]])+treeString[index:]+'\n')
	writer.close()

def seperateTrainDev():
	total = 6092
	percent = 0.2
	devIndices = set(np.random.choice(total,size = percent*total,replace=False))
	indicesWriter = open('data/dev_indices','w')
	indicesWriter.write(str(devIndices))
	indicesWriter.close()
	filename = 'data/tweeti.b.dist'
	reader = open(filename)
	trainWriter = open('data/b.train','w')
	devWriter = open('data/b.dev','w')
	for i,line in enumerate(reader):
		if i in devIndices:
			devWriter.write(line)
		else:
			trainWriter.write(line)
	devWriter.close()
	trainWriter.close()


transformToRNN(filename = 'data/b.dev.preprocessed.utf8.parsed')
transformToRNN(filename = 'data/b.train.preprocessed.utf8.parsed')

