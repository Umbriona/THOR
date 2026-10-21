#!/usr/bin/env python

import os
import numpy as np
import random
import re
from Bio import SeqIO
import argparse

parser = argparse.ArgumentParser(""" """)

parser.add_argument('-o', '--output', type=str, default = '../../data/clean/cleaned_brenda_sequences.fasta')

parser.add_argument( '--max_length', type=int, default = 2000,
                   help = '')
parser.add_argument('--min_length', type=int, default = 100,
                   help = '')
parser.add_argument('seq', type=str, default = '../../data/raw/brenda_sequences_20180109.fasta',
                   help = '')
parser.add_argument('tsv', type=str, default = '../../data/raw/enzyme_ogt_topt.tsv',
                   help = '')

def load_seq_info(infile):
    seq_info = dict()
    for line in open(infile):
        if line.startswith('ec'): continue
        cont = line.split()
        seq_info[cont[1]] = '{0} {1} {2}'.format(cont[3],cont[2].lower(),cont[4]) 
    print('Number of sequences with info:',len(seq_info))
    return seq_info

def is_valid_seq(seq, max_len, min_len):
    """
    True if seq is valid, False otherwise.
    """
    l = len(seq)
    valid_aas = "ACDEFGHIKLMNPQRSTVWY"
    if (l <= max_len) and set(seq) <= set(valid_aas) and l>=min_len: return True
    else: return False
    
def clean_sequences(seqfile, max_len, min_len):
    cleaned_seqs = dict()
    k = 0
    for rec in SeqIO.parse(seqfile,'fasta'):
        k += 1
        if not is_valid_seq(str(rec.seq), max_len, min_len): continue
        if len(rec.id) == 0: continue
        cleaned_seqs[rec.id] = rec
    print('Number of sequences after clean',len(cleaned_seqs))
    print('Number of removed sequences:',k-len(cleaned_seqs))
    return cleaned_seqs

def save_cleaned_seqs(seq_info,cleaned_seqs,outfile):
    fhand = open(outfile,'w')
    for seqid, rec in cleaned_seqs.items():
        try:
            rec.description = seq_info[seqid]
            SeqIO.write([rec],fhand,'fasta')
        except: pass
    fhand.close()
    return 0
    
def main(args):
    
    seq_info = load_seq_info(args.tsv)
    cleaned_seqs = clean_sequences(args.seq, args.max_length, args.min_length)
    save_cleaned_seqs(seq_info,cleaned_seqs, args.output)
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    
