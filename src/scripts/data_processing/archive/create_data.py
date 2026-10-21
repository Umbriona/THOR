import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import argparse
import sys, os
import csv
import re

from argparse import RawTextHelpFormatter
from collections import Counter
from Bio import SeqIO

# TODO: OVERHAUL

pd.set_option('display.max_rows', 100)

# parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

# parser.add_argument('-i', '--input_i', type=str, required=True,
#                     help="Input requirement: Fasta File.")

# parser.add_argument('-j', '--input_j', type=str, required=True,
#                     help="Input requirement: Emapper Annotations File.\nUsage: python create_data.py -i non_redundant_v2.fasta -j Emapper.emapper.annotations")

# parser.add_argument('-o', '--input_o', type=bool, required=True,
#                     help="Input requirement: True or False")

# parser.add_argument('-e', '--input_e', type=bool, required=True,
#                     help="Input requirement: True or False")



def make_hist(region, alphabet):

	region = Counter(region)
	lst = [0] * len(alphabet)
	k = 0 
	for letter in alphabet:
		lst[k] = region[letter]
		k +=1
	return lst


def normalize_feat (sequence):
	return [float(i)/sum(sequence) for i in sequence]


def sum_h (hist):
	return sum(hist)


def convertWithAttributes(input):
    """
    todo: doc
    """
    if not os.path.exists(input):
        raise IOError(errno.ENOENT, 'No such file', input)


    df_fasta = { "id":[], "Dataset":[], "Temperature": [], "Sequence":[] }


    for rec in SeqIO.parse(input, "fasta"):
        df_fasta["id"].append(rec.id)
        df_fasta["Dataset"].append(rec.description.split()[1])
        df_fasta["Temperature"].append(rec.description.split()[2])
        df_fasta["Sequence"].append(str(rec.seq))


    df_fasta = pd.DataFrame(df_fasta)

    return df_fasta



def write_files(cogs, fasta_format, csv_format, dataFrame, fastaPath, csvPath):


	fastExt = '.fasta'
	csvExt = '.csv'

	if csv_format == True or fasta_format == True:
		print ('writing files')
		for i in range (len(cogs)):
			print ('Processed line {} out of {}'.format(i, len(cogs)))
			tmp = dataFrame.loc[dataFrame.OG_clean==cogs[i]]
			if csv_format == True:
				tmp.to_csv(csvPath+cogs[i]+csvExt, sep=';', index=False, header=True, quoting=csv.QUOTE_NONE, escapechar = ' ')
			if fasta_format == True:
				tmp = tmp[['id', 'Dataset', 'Temperature', 'Sequence']]
				# fasta file modifications
				tmp['id'] = '>' + tmp['id'].astype(str)
				# add special character
				tmp['Sequence'] = '\n' + tmp['Sequence'].astype(str)
				tmp.to_csv(fastaPath+cogs[i]+fastExt, sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar = ' ')

	return 0


def make_data (csv_format, fasta_format, thres, threshold):
	aa_alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

	csv_format = True
	fasta_format = True
	thres = True
	threshold = 100

	eggnog_filename = "Emapper.emapper.annotations"
	# fasta_filename = 'non_redundent_OGT_IMG.fasta'
	fasta_filename = 'definitely_non_redundunt.fasta'

	col = ['eggNOG_OGs', '#query', 'EC']

	print ('calc')

	#TODO skip footer on eggnog output.
	df = pd.read_csv(eggnog_filename, dtype={"#query": "string"}, skiprows=4, sep='\t', usecols=col)

	print (df.isna().any())
	print (df.isnull().sum().sum())
	print (len(df.index))

	print ('calc2')
	df1 = convertWithAttributes(fasta_filename)

	print (df1.isna().any())
	print (df1.isnull().sum().sum())
	print (len(df1.index))

	df['OG_root'] = df['eggNOG_OGs'].str.split('|').str[0]
	df['OG_clean'] = df['eggNOG_OGs'].str.split('root,' , 1).str[1]
	df['OG_clean'] = df['OG_clean'].str.split(',').str[0]
	df[['OG_clean', 'domain']] = df.OG_clean.str.split('|', expand=True)

	print (df.head(40))

	df = df.drop('eggNOG_OGs', axis=1)
	print (df[['OG_clean', '#query']].head(40))

	print (df.head(40))
	print (df['domain'].value_counts(dropna = False))

	if thres: 
		df['count']=df.groupby('OG_clean')['#query'].transform('count')
		df = df[df['count'].ge(threshold)]
		# df = df.sort_values(['count'], ascending=[False])
		# print (df.head(10))

	print (df.isna().any())
	print (df.isnull().sum().sum())


	# left join on smaller dataset
	if (len(df1.index) <= len(df.index)):
		ddf = pd.merge(df1, df, how='left', left_on=['id'], right_on=['#query'])
	else:
		ddf = pd.merge(df, df1, how='left', left_on=['#query'], right_on=['id']) 

	# print (ddf.isna().any())
	# print (ddf.isnull().sum().sum())

	if csv_format == True:

		ddf = ddf[['id', 'Dataset', 'Temperature', 'domain', 'OG_root', 'OG_clean', 'EC',  'Sequence']]
		ddf['feature'] = ddf.apply (lambda row: make_hist(row['Sequence'], aa_alphabet), axis=1)
		ddf['seq_len'] = ddf.apply (lambda row: sum_h(row['feature']), axis=1)
		ddf['normalized_feat'] = ddf.apply (lambda row: normalize_feat(row['feature']), axis=1)
	
	# sort the dataframe
	ddf.sort_values(by='OG_clean', inplace=True)

	print ('done1')

	if thres:
		# set the index to be this and don't drop
		ddf.set_index(keys=['OG_clean'], drop=False,inplace=True)
		# get a list of COG names
		cogs=ddf['OG_clean'].unique().tolist()
		print (len(cogs))


	fastaFolder = 'outFasta/'
	csvFolder = 'outCSV/'
	write_files(cogs, fasta_format, csv_format, ddf, fastaFolder, csvFolder)


	return 0 

make_data()
# if __name__ == "__main__":
#     args = parser.parse_args()
#     main(args)