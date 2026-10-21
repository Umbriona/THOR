import pandas as pd
import numpy as np
import os
import glob
import re
import csv
import sys
from Bio import SeqIO

def temp_class4(input):

	if input < 20:
		return "Psychrophile"
	elif input < 45:
		return 'Mesophile'
	else:
		return "Thermophile"


def convertWithAttributes(input):
    """
    todo: doc
    """
    if not os.path.exists(input):
        raise IOError(errno.ENOENT, 'No such file', input)


    df_fasta = { "id":[], "Temperature": [], "Sequence":[] }


    for rec in SeqIO.parse(input, "fasta"):
        df_fasta["id"].append(rec.id)
        df_fasta["Temperature"].append(rec.description.split()[1])
        df_fasta["Sequence"].append(str(rec.seq))


    df_fasta = pd.DataFrame(df_fasta)

    return df_fasta


def split_data(frame, tr_set, tst_set, ext): #TODO make this pretty


	psyStrBuild = tr_set+'/train_psychro_'+ext+'.fasta'
	thermoStrBuild = tr_set+'/train_thermo_'+ext+'.fasta'
	mesoStrBuild = tr_set+'/train_meso_'+ext+'.fasta'

	mesoTstStrBuild = tst_set+'/test_meso_'+ext+'.fasta'

	psychro = frame[frame['tmp_class'].str.contains('Psychrophile')]
	if len(psychro) != 0:
		psychro = psychro[['id', 'Temperature', 'tmp_class', 'Sequence']]
		# fasta file modifications
		psychro['id'] = '>' + psychro['id'].astype(str)
		# add special character
		psychro['Sequence'] = '\n' + psychro['Sequence'].astype(str)
		psychro.to_csv(psyStrBuild, sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar = ' ')
	else:
		print ('zero psychro')

	thermo = frame[frame['tmp_class'].str.contains('Thermophile')]
	if len(thermo) != 0:
		thermo = thermo[['id', 'Temperature', 'tmp_class', 'Sequence']]
		# fasta file modifications
		thermo['id'] = '>' + thermo['id'].astype(str)
		# add special character
		thermo['Sequence'] = '\n' + thermo['Sequence'].astype(str)
		thermo.to_csv(thermoStrBuild, sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar = ' ')
	else:
		print ('zero thermo')

	meso = frame[frame['tmp_class'].str.contains('Mesophile')]
	if len(meso) != 0:
		meso = meso[['id', 'Temperature', 'tmp_class', 'Sequence']]
		# fasta file modifications
		meso['id'] = '>' + meso['id'].astype(str)
		# add special character
		meso['Sequence'] = '\n' + meso['Sequence'].astype(str)

		mask = np.random.rand(len(meso)) <= 0.5
		train_meso = meso[mask]
		test_meso = meso[~mask]

		train_meso.to_csv(mesoStrBuild, sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar = ' ')
		test_meso.to_csv(mesoTstStrBuild, sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar = ' ')
	else:
		print ('zero meso')




# contents40 = (glob.glob('/Users/niktat/Desktop/outC/clusters/clu_40/top_100_40/*'))
# contents50 = (glob.glob('/Users/niktat/Desktop/outC/clusters/clu_50/top_100_50/*'))
# contents60 = (glob.glob('/Users/niktat/Desktop/outC/clusters/clu_60/top_100_60/*'))
# contents70 = (glob.glob('/Users/niktat/Desktop/outC/clusters/clu_70/top_100_70/*'))

contents40 = (glob.glob('/cephyr/NOBACKUP/groups/snic2019-35-21/Nikos/clusters/clu_40/top_100_40/*'))
contents50 = (glob.glob('/cephyr/NOBACKUP/groups/snic2019-35-21/Nikos/clusters/clu_50/top_100_50/*'))
contents60 = (glob.glob('/cephyr/NOBACKUP/groups/snic2019-35-21/Nikos/clusters/clu_60/top_100_60/*'))
contents70 = (glob.glob('/cephyr/NOBACKUP/groups/snic2019-35-21/Nikos/clusters/clu_70/top_100_70/*'))


all_contents = [contents40, contents50, contents60, contents70]


# Remove clutter

for i in range (len(all_contents)):
	for content in all_contents[i]:
		if (content.endswith('.tsv')):
			os.remove(content)


for i in range (len(all_contents)):
	for content in all_contents[i]:
		print (content)
		ext = content.split('/')[-1].split('_')[1].split('.')[0]
		fasta_file = content+'/outFasta_'+ext+'.fasta'
		df_fasta = convertWithAttributes(fasta_file)
		df_fasta['Temperature'] = df_fasta['Temperature'].astype(float)
		df_fasta['tmp_class'] = df_fasta.apply (lambda row: temp_class4(row['Temperature']), axis=1)


		tr_set = content+'/TrainingSet'
		tst_set = content+'/TestSet'
		os.mkdir(tr_set)
		os.mkdir(tst_set)

		split_data(df_fasta, tr_set, tst_set, ext)



exit()






