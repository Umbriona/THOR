#! /usr/bin/env python

import os
from multiprocessing import Pool
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf 
from Bio import SeqIO

from scipy.stats import boxcox

import matplotlib.pyplot as plt
import seaborn as sns

# DEFINES
STR = "******************\nClass description\n******************\n\n\nClass    temperature-range\n"
AAS = 'XACDEFGHIKLMNPQRSTVWY'
# Create lookup table
TABLE = {aa: i-1 for i, aa in enumerate(AAS)}
TABLE['U'] = -1

LOWEST_TEMP=-20
HIGHEST_TEMP=200


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-i', '--input', type=str,  required = True,
                   help = "input files. Mesophiles need to be first")
parser.add_argument('-o', '--output', required=True,
                   help = "")
parser.add_argument( '--max_entries', type=int, default=20000,
                  help = "maximum entries in one records file")



parser.add_argument('-v', '--verbose', action="store_true")


################################################### Healper functions  ###########################################################

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def to_int(seq): 
    tmp = -np.ones((512,), dtype = np.int64)
    seq = [ TABLE[aa] for aa in str(seq).upper()]
    tmp[:len(seq)] = seq
    return tmp

def parse_ofset(name):
    temp_low = int(name.split('_')[0])
    temp_high= int(name.split('_')[1])
    return temp_low + (temp_high - temp_low)/2

def get_class_sizes(args, data_df):
    temp_ranges = [LOWEST_TEMP]+args.ranges
    temp_ranges.append(HIGHEST_TEMP)
    print(temp_ranges)
    class_sizes = []
    for idx in range(len(temp_ranges)-1):
        class_sizes.append(np.sum(np.logical_and(data_df['Temperature']>temp_ranges[idx], data_df['Temperature']<=temp_ranges[idx+1])))
    
    return np.array(class_sizes), np.array(temp_ranges)


################################################### Load Data  ###########################################################

def load_sequences(file):
    """
    Loads fasta file in to directory
    requires args.input to be a fasta file
    """

    # Load all classes from fasta without upsacle (To not leak data)
    
    data_df = {"Id":[], "Temperature":[], "Sequence":[]}
    ofset = 103 - 2
    n=0
    for rec in SeqIO.parse(file, "fasta"):
        data_df["Id"].append(rec.id)
        data_df["Temperature"].append(float(rec.description.split()[1]) - ofset)
        data_df["Sequence"].append(str(rec.seq))
        n+=1
    data_df = pd.DataFrame(data_df)
    data_df["Sequence_Int"] = data_df["Sequence"].apply(to_int)
    
        
    return data_df, n




################################################### Writing record ###########################################################

def write_record(args, file, data_df): 
    
    entries = data_df.shape[0]
    counter_entries = 0
    counter_files = 0
    while counter_entries < entries-1:
        name =  file.format(counter_files)
        print(f"Name is {name}")           
        with tf.io.TFRecordWriter(name) as tfrecord:
            for counter_inner, row in enumerate(data_df[counter_entries:].iterrows()):
                features = {
                  'seq': _int64_feature(row[1]["Sequence_Int"]),
                  'value': _float_feature(row[1]["Temperature"])
                  }
                element = tf.train.Example(features = tf.train.Features(feature=features))
                tfrecord.write(element.SerializeToString())
                if counter_inner > args.max_entries:
                    break
        counter_entries += counter_inner
        counter_files += 1

    return 0

################################################### make README ####################################################

def readme(args):
    ## List all records created first
    list_of_records = os.listdir(os.path.dirname(args.output))
    with open(os.path.join(args.output, "README.md"), "w") as file_writer:
        file_writer.write("""# Directory populated with records files from directory ../TrainingSet.\n
        ## Files\n""")
        for file in list_of_records:
            file_writer.write(f"**{file}")
    
################################################### Main ###########################################################
def main(args):
    
        # require output dir
    output_dir = os.path.dirname(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print(f"Output dir is {output_dir}")
    feature="Temperature"
    meso_df, n_meso = load_sequences(args.input[0])
    target_df, n_target = load_sequences(args.input[1])
    
    ## scale target data
    scale= n_meso/n_target
    target_df = target_df.sample(frac = scale, replace=True)
    print(f"Meso data frame has {meso_df.shape[0]} rows")
    print(f"Target data frame has {target_df.shape[0]} rows")


    
    file_meso = os.path.join(output_dir, "Meso_{}.tfrecord")
    file_target = os.path.join(output_dir, "Target_{}.tfrecord")
    
    write_record(args, file_meso, meso_df)
    write_record(args, file_target, target_df)
    
    return 0

if __name__ == '__main__':

    args = parser.parse_args()
    main(args)