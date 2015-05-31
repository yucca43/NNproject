
import csv
def transformToRNN(filename = 'tweeti.b.dist.parsed'):

	outfile = 'trees/temp.txt'
	if filename == 'tweeti.b.dist.parsed':
		outfile = 'trees/train.txt'
	else filename == 'tweeti.b.dev.parsed':
		outfile = 'trees/dev.txt'
	labelMap = {"neutral":1,"positive":2,"negative":0,"objective":1,"objective-OR-neutral":1,
	"neutral\"\"":1,"positive\"\"":2,"negative\"\"":0,"objective\"\"":1,"objective-OR-neutral\"\"":1}
	reader = csv.reader(open(filename),delimiter="\t")
	writer = open(outfile,'w')
	for line in reader:
		treeString = line[3][6:-1]
		index = treeString.index(' ')
		writer.write("("+str(labelMap[line[2]])+treeString[index:]+'\n')
	writer.close()

# transformToRNN(filename = 'tweeti.b.dev.parsed')
# transformToRNN(filename = 'tweeti.b.dist.parsed')

