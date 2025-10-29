import pandas as pd
# import modin.pandas as pd
import argparse
import sys, os
import ray
import csv
import re

from argparse import RawTextHelpFormatter
from io import StringIO
# from modin.config import Engine
from Bio import SeqIO


# Usage: python remove_virus.py -i non_redundent_OGT_IMG.fasta -j Emapper.emapper.annotations 
# ray.init()
# Engine.put("ray")

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

parser.add_argument('-i', '--input_i', type=str, required=True,
                    help="Input requirement: Fasta File.")

parser.add_argument('-j', '--input_j', type=str, required=True,
                    help="Input requirement: Emapper Annotations File.")

parser.add_argument('-k', '--input_k', type=str, required=True,
                    help="Input requirement: If we also need a csv filtered by EC for kcats predictions.")


parser.add_argument('-l', '--input_l', type=str, required=True,
                    help="Input requirement: If fasta file is broken, set to True.\nUsage: python remove_virus.py -i non_redundent_OGT_IMG.fasta -j Emapper.emapper.annotations -k False -l False")


def convertWithAttributes(input):
    """
    todo: doc
    """

    # input = StringIO(input)

    df_fasta = { "id":[], "Dataset":[], "Temperature": [], "Sequence":[] }


    for rec in SeqIO.parse(input, "fasta"):
        df_fasta["id"].append(rec.id)
        df_fasta["Dataset"].append(rec.description.split()[1])
        df_fasta["Temperature"].append(rec.description.split()[2])
        df_fasta["Sequence"].append(str(rec.seq))


    df_fasta = pd.DataFrame(df_fasta)

    return df_fasta



def ec_filtering(df):
	"""
	Keeps sequences that have an EC number (needed for Kcats pred) and it filters 
	out sequences such that their length is between 100 and 512 AA long.
	"""

	ddf = df[df['EC'].str.contains('-') == False]
	msk = (ddf['Sequence'].str.len() >= 100) & (ddf['Sequence'].str.len() <= 512)
	ddf = ddf.loc[msk]

	ddf.to_csv('kcats.tsv', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE, escapechar = ' ')

	return ddf


def check_fasta_consistency(fasta_file_name):

	output = StringIO()

	with open(fasta_file_name,'r') as fi:
		for ln in fi:
			if ln.startswith('>'):
				k = ln.split(' ')
				if len(k) == 2:
					k.append(k[1])
					k[1] = 'Gang'
					output.write(' '.join(str(x) for x in k))
					continue
			output.write(ln)

	with open('intermediate.fasta', mode='w') as f:
		print(output.getvalue(), file=f)

	return output

def remove_viruses(fasta_file_name, eggnog_file_name, ec_filter, val):

	print (ec_filter)
	print (val)
	"""
	todo: doc
	"""
	col = ['eggNOG_OGs', '#query', 'EC']
	df = pd.read_csv(eggnog_file_name, dtype={"#query": "string"}, skiprows=4, skipfooter=3, sep='\t', usecols=col)


	# Just keep Bacteria, Eukaryota & Archaea
	df = df[(df['eggNOG_OGs'].str.contains('Bacteria')) | (df['eggNOG_OGs'].str.contains('Eukaryota')) | (df['eggNOG_OGs'].str.contains('Archaea'))]
	# df = df[~df['eggNOG_OGs'].str.contains("Viruses")]


	if val == True:
		fasta_fixed = check_fasta_consistency(fasta_file_name)
		df1 = convertWithAttributes('intermediate.fasta')
	else:
		df1 = convertWithAttributes(fasta_file_name)

	ddf = pd.merge(df, df1, how='left', left_on=['#query'], right_on=['id'])
	ddf = ddf[['id', 'Dataset', 'Temperature', 'EC', 'Sequence']]


	if ec_filter == True:
		ec_filtering(ddf)

	# fasta file modifications
	ddf['id'] = '>' + ddf['id'].astype(str)
	# add special character
	ddf['Sequence'] = '\n' + ddf['Sequence'].astype(str)


	ddf.to_csv('meltome_non_redundunt.fasta', sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar = ' ')

	return 0


def main(args):

	if (args.input_k == 'False'):
		args.input_k = False
	else:
		args.input_k = True


	if (args.input_l == 'False'):
		args.input_l = False
	else:
		args.input_l = True


	remove_viruses(args.input_i, args.input_j, args.input_k, args.input_l)

	return 0


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)