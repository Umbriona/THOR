'''
import pandas as pd
import numpy as np
import glob, subprocess
import shutil

allCogs = (glob.glob('/Users/niktat/Desktop/EggNog/outCSV/*'))
cols = ['id']
path = '/Users/niktat/Desktop/EggNog/outCSV/'
fasta_path = '/Users/niktat/Desktop/EggNog/outFasta/'
destination = '/Users/niktat/Desktop/top100cogs'

test_set = [path+'COG0065@2.csv', path+'COG3631@2.csv', path+'COG4319@2.csv', path+'COG0039@2.csv', path+'COG2032@2.csv', path+'COG0441@2.csv']


hashmap = {}

for i in range (len(allCogs)):
	if allCogs[i] in test_set:
		print (allCogs[i])
		continue

	df = pd.read_csv(allCogs[i], sep=';', usecols = cols)
	dfLen = len(df)
	hashmap[allCogs[i]] = dfLen


sorted_hash = {k: v for k, v in sorted(hashmap.items(), reverse=True, key=lambda item: item[1])}

cnt = 0
sufix = '.fasta'
for key in sorted_hash:
	thisFasta = fasta_path+(key.split('/')[-1].split('.')[0])+sufix
	print (thisFasta)
	shutil.copy2(thisFasta, destination)
	cnt +=1
	if (cnt == 100):
		break
'''



from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import Bio.SeqIO as SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
import glob, subprocess
from io import StringIO
import pandas as pd
import csv
import os
import shutil


contentsCOG = (glob.glob('/cephyr/NOBACKUP/groups/snic2019-35-21/Nikos/top100cogs/*'))


all_contents = [contentsCOG]


def create_structure(fromContents):

        mystr = fromContents.split('/')[-1].split('.')[0]
        folder_name = fromContents.split('.')[0]

        os.mkdir(folder_name)
        in_file = fromContents

        out_file = folder_name+'/aligned_'+mystr+'.fasta'
        perc_out = folder_name+'/out_mat'
        kimura_out = folder_name+'/out_mat_kimura'

        return in_file, out_file, perc_out, kimura_out, mystr, folder_name


for i in range (len(all_contents)): #TODO outer loop not really needed
        for j in range (len(all_contents[i])):

                in_file, out_file, perc_out, kimura_out, mystr, folder_name = create_structure(all_contents[i][j])

                print ('calculating for: '+mystr+' path= '+ folder_name)

                shutil.move(in_file, folder_name)
                clustalomega_cline = ClustalOmegaCommandline(infile=in_file, outfile=out_file, verbose=True, distmat_full = True,  percentid = True, distmat_out = perc_out, threads = 32)
                clustalomega_cline()

                clustalomega_cline = ClustalOmegaCommandline(infile=in_file, profile1 = out_file, distmat_full = True, distmat_out = kimura_out, usekimura = 'True', threads = 32)
                clustalomega_cline()

                shutil.move(in_file, folder_name)