import pandas as pd
import numpy as np
import csv





path = '/Users/niktat/Desktop/EggNog/outCSV/'
csv_data = '/Users/niktat/Desktop/EggNog/non_redundunt_v2.csv'
test_set = ['COG0065@2.csv', 'COG3631@2.csv', 'COG4319@2.csv', 'COG0039@2.csv', 'COG2032@2.csv', 'COG0441@2.csv']


cols = ['id', 'Dataset', 'Temperature', 'Sequence']
training_set = pd.read_csv(csv_data, sep=';', usecols = cols)


tempDfList = []
for i in range (len(test_set)):
	tmp = pd.read_csv(path+test_set[i], sep=';', usecols = cols)
	print (len(tmp))
	tempDfList.append(tmp)

test_set = pd.concat(tempDfList)
tr_len = (len(training_set))

cond = training_set['id'].isin(test_set['id'])
training_set.drop(training_set[cond].index, inplace = True)
assert (len(training_set) == tr_len - len(test_set))

#save cvs
training_set.to_csv('training_set.csv', sep=';', index=False, header=True, quoting=csv.QUOTE_NONE, escapechar = ' ')
test_set.to_csv('test_set.csv', sep=';', index=False, header=True, quoting=csv.QUOTE_NONE, escapechar = ' ')

# fasta file modifications
training_set['id'] = '>' + training_set['id'].astype(str)
test_set['id'] = '>' + test_set['id'].astype(str)
# add special character
training_set['Sequence'] = '\n' + training_set['Sequence'].astype(str)
test_set['Sequence'] = '\n' + test_set['Sequence'].astype(str)
# save it
training_set.to_csv('training_set.fasta', sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar = ' ')
test_set.to_csv('test_set.fasta', sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar = ' ')

print ('done')


