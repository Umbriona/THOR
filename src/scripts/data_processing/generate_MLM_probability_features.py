#! /usr/bin/env python
import os
from bert_class import BertPatcher, BertLikelihood
from Bio import SeqIO
import pandas as pd
import shutil
from time import time
import argparse
import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool

############################# Create records


# DEFINES
STR = "******************\nClass description\n******************\n\n\nClass    temperature-range\n"
AAS = 'XACDEFGHIKLMNPQRSTVWY'
# Create lookup table
TABLE = {aa: i-1 for i, aa in enumerate(AAS)}
TABLE['U'] = -1
TABLE['B'] = -1
TABLE['Z'] = -1
TABLE['<cls>'] = -1
TABLE['<eos>'] = -1
TABLE['<unk>'] = -1
TABLE['<pad>'] = -1
TABLE['<mask>'] = -1

LOWEST_TEMP=-20
HIGHEST_TEMP=200

PARTITION_SIZE = 20 # 15%


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-i', '--input', type=str,  required = True,
                   help = "input fasta files that have the sequences you want to extract features from")
parser.add_argument('-o', '--output', required=True,
                   help = "Directory where the features will be stored.")
parser.add_argument( '--max_entries', type=int, default=50000,
                  help = "maximum entries in one pickle dictionary file")

parser.add_argument('--gpu', type=str, default="0")

parser.add_argument('--model', type=str, default="ESM")

parser.add_argument('--name',  type=str, default="sequence_mlm_features",
                  help = "name of output files")



parser.add_argument('-v', '--verbose', action="store_true")

def load_model(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    model_name= args.model #"BertRost" #"ESM" #BertRost
    model = BertLikelihood(topK = 21, modelName = model_name, device = 0, batch_size = 512)
    return model

def read_fasta(args):
    

    file = args.input

    #print(f"Reading sequences from file: {file}")
    records = sorted(((rec.id, str(rec.seq)) for rec in SeqIO.parse(file, "fasta")), key=lambda x: len(x[1]))
    dict_fasta = {"ids": [rid for rid, _ in records],
                "seq": [seq for _, seq in records]}

    return dict_fasta

def human_time(c):
    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = c % 60
    
    return f"Days: {days}, Hours: {hours}, Minutes: {minutes}, Seconds: {seconds}" 

def estimated_finish(start, stop, idx, n_sequences):
    c = stop-start
    estimated_time2finish = int(n_sequences/idx)*c -stop
    return estimated_time2finish

def _rand_index_shuffle( seq):
    arr = np.arange(len(seq))
    np.random.shuffle(arr)
    return arr

def _str_prepare_index( seq, arr):

    list_seq = list(seq)
    for idx in arr:
        list_seq[idx] = '<mask>'
    seq_m = " ".join(list_seq)
    return seq_m

def data_preprocessing( args):
        seq, _id = args
        
        arr = _rand_index_shuffle(seq)
        #print(arr)
        partition_width = len(seq)//PARTITION_SIZE 
        #print(partition_width)
        if len(seq)%PARTITION_SIZE == 0:
            remainder = 0
        else:
            remainder = 1
        batch = []
        idxs  = []
        for i in range(PARTITION_SIZE+1):
            if  arr[i*partition_width:(i+1)*partition_width].size<1:
                continue
            batch.append(_str_prepare_index(seq, arr[i*partition_width:(i+1)*partition_width]))
           # print(f"arr: {arr[i*partition_width:(i+1)*partition_width]}\n")
            idxs.append(arr[i*partition_width:(i+1)*partition_width])
        return batch, _id, idxs
        

def write_feature(dict_fasta, model, args):
    
    max_entries = args.max_entries
    n_sequences = len(dict_fasta["ids"])
    start_all = time()
    n_batches = n_sequences // max_entries
    for idx in range(n_batches+1):
        
        start = time()
        #generate probability features
        list_seq = dict_fasta["seq"][idx*max_entries:(1+idx)*max_entries]
        list_id = dict_fasta["ids"][idx*max_entries:(1+idx)*max_entries]
        #print(list_seq)
        #exit()
        print(f"Starting computing batch: {idx}")
        dict_likelihoods = model.calc_likelihood_15(list_seq, list_id)
        print(f"Done computing batch: {idx}")
         #print(dict_likelihoods)

        with open(f"{args.output}/{args.name}_{idx}.pkl", "bw") as f:
            pickle.dump(dict_likelihoods, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote batch: {idx} to file: {args.output}/{args.name}_{idx}.pkl")
        stop = time()
        if args.verbose:
            print(f"Generated features for se with index {idx*max_entries}:{(1+idx)*max_entries} out of {n_sequences} sequences")
            print(f"Elapsed time for batch: {human_time(stop-start)} Total time elapesed: {human_time(stop-start_all)}")
            #print(f"Estimated time to finish {human_time(estimated_finish(start_all, stop, ((1+idx)*max_entries), n_sequences))}")
            


def main(args):
    
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    
    
    model = load_model(args)
    print("Loaded model")
    dict_fasta = read_fasta(args)
    print(f"Read fasta file with {len(dict_fasta['ids'])} number of sequences")
    write_feature(dict_fasta, model, args)
    print("Done computing  MLM features") 
    
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
