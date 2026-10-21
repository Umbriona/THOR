#! /usr/bin/env python
# write fasta temp
import os
from multiprocessing import Pool
import argparse
from itertools import repeat

import numpy as np 
from Bio import SeqIO

STR = "******************\nClass description\n******************\n\n\nClass    temperature-range\n"

parser = argparse.ArgumentParser(""" """)

parser.add_argument('-i', '--input', type=str, default = '../../data/clean/OGT_sequences.fasta',
                   help = "")
parser.add_argument('-o', '--output', nargs='+', default = '../../data/OGT_Classes',
                   help = "")
parser.add_argument('-n', '--n_bins', type=int, default = 5,
                   help = "")
parser.add_argument('-T', '--threads', type=int, default = 4,
                   help = "")
parser.add_argument('--max_temp', type=int, default = 80,
                   help = "")
parser.add_argument('--min_temp', type= int, default = 0,
                   help = "")

parser.add_argument('-v', '--verbose', type=int, choices=[0,1,2], default = 1)

def load_data(file):
    seq_dict = {'id':[], 'ogt':[], 'seq':[]}
    for rec in SeqIO.parse(file, 'fasta'):
        seq_dict['id'].append(rec.id)
        seq_dict['ogt'].append(float(rec.description.split()[-1]))
        seq_dict['seq'].append(rec.seq)
    return seq_dict

def get_temperature_diff(seq_dict, n_bins = 10, args = None, output_files=[" "," "]):
    raw_temp = seq_dict['ogt']
    raw_temp = [ i  for i in raw_temp if i <=args.max_temp]
    hist_temp, bin_edge_temp = np.histogram(raw_temp, bins = n_bins)
    temp_bins = [(bin_edge_temp[i],bin_edge_temp[i+1]) for i in range(n_bins)]
    min_bin = np.min(hist_temp)
    
    if args.verbose>0:
        with open(os.path.dirname(output_files[0])+"/README", 'w') as f:
            f.write(STR)
            for i, name in enumerate(output_files):
                temp1 = temp_bins[i][0]
                temp2 = temp_bins[i][1]
                f.write('%s %6.2f%6.2f\n' % (name.split('/')[-1], temp1, temp2))
                
            f.write("\n\nTotal number of samles: %i" % (min_bin*args.n_bins))
            
    return temp_bins, min_bin

def filter_temp(temp, min_bin, seq_dict, file):

    f = open(file,'w')
  
    count= 0
    for id_, seq, ogt in zip(seq_dict['id'], seq_dict['seq'], seq_dict['ogt']):
        if ogt>=temp[0] and ogt<temp[1]:
            f.write('>{} {}\n{}\n'.format(id_,ogt,seq))
            count += 1
        else:
            continue
        if count >= min_bin:
            break            
    f.close()
    return 0

def main(args):

    seq_dict = load_data(args.input)
    temp_bins, min_bin = get_temperature_diff(seq_dict, n_bins = args.n_bins, args = args, output_files = args.output)
    
    
    print("Temperature classes selected", temp_bins)
    print("Number of samples in each class", min_bin)
    print("Total number of samples", min_bin*args.n_bins)
    p = Pool(min(args.threads, args.n_bins))
    p.starmap(filter_temp, zip(temp_bins,
                               repeat(min_bin, args.n_bins),
                               repeat(seq_dict,args.n_bins),
                               args.output))
    
    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)