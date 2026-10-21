from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import Bio.SeqIO as SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
import glob, subprocess
from io import StringIO
import pandas as pd
import csv
import os


contents40 = (glob.glob('/cephyr/NOBACKUP/groups/snic2019-35-21/Nikos/clusters/clu_40/top_100_40/*'))
contents50 = (glob.glob('/cephyr/NOBACKUP/groups/snic2019-35-21/Nikos/clusters/clu_50/top_100_50/*'))
contents60 = (glob.glob('/cephyr/NOBACKUP/groups/snic2019-35-21/Nikos/clusters/clu_60/top_100_60/*'))
contents70 = (glob.glob('/cephyr/NOBACKUP/groups/snic2019-35-21/Nikos/clusters/clu_70/top_100_70/*'))


all_contents = [contents40, contents50, contents60, contents70]


def create_structure(fromContents):

	mystr = fromContents.split('/')[-1].split('_')[1].split('.')[0]
	folder_name = fromContents.split('.')[0]

	os.mkdir(folder_name)

	in_file = folder_name+'/outFasta_'+mystr+'.fasta'
	out_file = folder_name+'/aligned_'+mystr+'.fasta'
	perc_out = folder_name+'/out_mat'
	kimura_out = folder_name+'/out_mat_kimura'

	return in_file, out_file, perc_out, kimura_out, mystr, folder_name


cols = ['id', 'Temperature', 'Sequence']
cols_small = ['col2']
df1 = pd.read_csv('all_data0.csv', sep=';', usecols = cols)



for i in range (len(all_contents)):
	for j in range (len(all_contents[i])):

		in_file, out_file, perc_out, kimura_out, mystr, folder_name = create_structure(all_contents[i][j])
		df = pd.read_csv(all_contents[i][j], sep='\t', usecols = cols_small)

		ddf = pd.merge(df, df1, how='left', left_on=['col2'], right_on=['id'])
		ddf = ddf[['id', 'Temperature', 'Sequence']]

		assert (df.shape[0] == ddf.shape[0])
		print (ddf.head())

		# fasta file modifications
		ddf['id'] = '>' + ddf['id'].astype(str)
		# add special character
		ddf['Sequence'] = '\n' + ddf['Sequence'].astype(str)
		ddf.to_csv(in_file, sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar = ' ')


		print ('calculating for: '+mystr+' path= '+ folder_name)

		clustalomega_cline = ClustalOmegaCommandline(infile=in_file, outfile=out_file, verbose=True, distmat_full = True,  percentid = True, distmat_out = perc_out, threads = 32)
		clustalomega_cline()

		clustalomega_cline = ClustalOmegaCommandline(infile=in_file, profile1 = out_file, distmat_full = True, distmat_out = kimura_out, usekimura = 'True', threads = 32)
		clustalomega_cline()






