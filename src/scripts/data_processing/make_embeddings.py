import transformers

import pandas as pd
import numpy as np
from tqdm import tqdm

import pickle

from transformers import pipeline
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, AutoModelForMaskedLM

import torch
import os
#from esm import Alphabet, FastaBatchedDataset, ProteinBertModel

from Bio import SeqIO

import argparse

parser = argparse.ArgumentParser( 
    """ This script will read a fasta file """)

parser.add_argument("-i", "--input", type = str, required = True, help = 
                    "Path to fasta file to be converted")
parser.add_argument("--store_all", action= "store_true",
                   help="Store all info if set")
parser.add_argument("--max_entries", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--ID", type=str, default="")
parser.add_argument("-o", "--output", type = str, default = "./embedings.pkl", help = 
                   "Path to output ")

NON_STANDARD_AMINO = ["B", "U", "Z", "X"]


def read_fasta(file):
    
    df_fasta = { "id":[], 'seq':[], "TM":[], "len":[]}
    count = 0
    for rec in SeqIO.parse(file, "fasta"):
        sequence = str(rec.seq)
        
        if any(amino in sequence for amino in NON_STANDARD_AMINO):
            #print(f"Sequences can not contain non standard amino acids Removing: {rec.id}")
            count += 1
            continue
            #raise ValueError (f"Sequences can not contain non standard amino acids")
        try: 
            int(float(rec.description.split()[-1].split('=')[-1]))
        except:
            print(f"{float(rec.description.split()[-1].split('=')[-1])} Not a value")
            continue
        df_fasta["id"].append(rec.id)
        df_fasta["seq"].append(str(rec.seq))
        df_fasta["len"].append(len(rec.seq))
        df_fasta["TM"].append(float(rec.description.split()[-1].split('=')[-1]))
    df_fasta = pd.DataFrame(df_fasta)
    df_fasta.sort_values(by="len", inplace=True)
    print(f"Sequences can not contain non standard amino acids Removing: {count} sequences")
    return df_fasta



def creat_embedings(df_fasta):

    print("Using new emb")
    model, alphabet = torch.hub.load("facebookresearch/esm", "esm2_t33_650M_UR50D") # esm1_t34_670M_UR50S "esm1b_t33_650M_UR50S" 
    model = model.to(device = 0)
    batch_converter = alphabet.get_batch_converter()

    total_entries = df_fasta.shape[0]
    print(f"Total number of entries in data frame is: {total_entries}")
    file_nr = 0 
    for idx in range((total_entries // args.max_entries) + 1):
        start = idx*args.max_entries
        end   = (idx+1)*args.max_entries
        
        if start >= end:
            break  # Safety check
            
        print(f"Starting computing entries {start} to {end}")  
        results = []
        tmp_df = df_fasta.iloc[start:end].copy()

        sequences = list(zip(tmp_df["id"], tmp_df["seq"]))
        
        for batch_start in tqdm(range(0, len(sequences), args.batch_size), desc=f"Batching [{start}:{end}]"):
            output = args.output
            if os.path.isfile(output):
                print(f"file: {output} alredy exist")
                continue
            
            batch_end = batch_start + args.batch_size
            batch_data = sequences[batch_start:batch_end]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            
            with torch.no_grad():
                out = model(batch_tokens.to(device=0), repr_layers=[33], return_contacts=False)
            
            emb = out["representations"][33].cpu().numpy()
            emb_mean = np.mean(emb, axis=1)  # shape: (sequence_length, hidden_dim) -> (hidden_dim,)            
            results.extend(emb_mean.tolist())


        print(f"Done computing embeddings embeddings for entries {start} to {end}")  
        tmp_df = tmp_df.iloc[:len(results)] 
        tmp_df["Embedding"] = results
        #output = os.path.join(args.output, f"embeddings_TEMP_{int(tmp_df['TM'].min())}_{int(tmp_df['TM'].max())}_ID_{args.ID}_file_{file_nr}.pt")
        print(f"Writing embeddings to file {output}")
        write_pickel(tmp_df, output)
       # write_record(tmp_df, output)
        print(f"written file {file_nr}")
        file_nr += 1
        


        
def write_pickel(df_fasta, output):
    with open(output, 'wb') as f:
        pickle.dump(df_fasta, f)
        
def write_record(tmp_df, output):
    embeddings = np.stack(tmp_df['Embedding'].values)  # shape: (N, D)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    torch.save({
        'embeddings': torch.tensor(embeddings_tensor),
        'temperatures': torch.tensor(tmp_df['TM'].values, dtype=torch.float32),
        'ids': tmp_df['id'].tolist()
    }, output)
    

def main(args):
    
    df_fasta = read_fasta(args.input)
    df_fasta = creat_embedings(df_fasta)

    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
