#! /usr/bin/env python

import os
from multiprocessing import Pool
import argparse

import numpy as np
import tensorflow as tf 
from Bio import SeqIO

# DEFINES
STR = "******************\nClass description\n******************\n\n\nClass    temperature-range\n"
AAS = 'XACDEFGHIKLMNPQRSTVWY'
# Create lookup table
TABLE = {aa: i-1 for i, aa in enumerate(AAS)}
TABLE['U'] = -1


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-i', '--input', nargs='+', type=str, required = True,
                   help = "input files")
parser.add_argument('-o', '--output', required=True,
                   help = "")
parser.add_argument('-s', '--shards', type = int, default = 2,
                   help = 'How many files to save the records in. -s 20 will split the data in 20 files')



parser.add_argument('-v', '--verbose', action="store_true")



def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))




def to_int(seq): 
    tmp = -np.ones((512,), dtype = np.int64)
    seq = [ TABLE[aa] for aa in str(seq).upper()]
    tmp[:len(seq)] = seq
    return tmp

def parse_ofset(name):
    temp_low = int(name.split('_')[0])
    temp_high= int(name.split('_')[1])
    return temp_low + (temp_high - temp_low)/2


def load_sequences(args):
    data_list = {}
    count_all = 0
    # Load all classes from fasta without upsacle (To not leak data)
    for file in args.input:
        data = {rec.id: (to_int(str(rec.seq).upper()), rec.description.split()[-1]) for rec in SeqIO.parse(file, "fasta")}
        #Count upscaling 
        if(args.verbose):
            count_all += len(data) 
            print("{} Sequences in file {}".format(count_all , file))
        data_list[file.split('.')[0]] = (data, file.split('.')[0])
    if(args.verbose):    
        print(f"\n\n{count_all} Train Sequences in total\n")
    return data_list

def scale_sequences(args, data_list):
    list_data_set = []

    # Upsacale classes and store id in list that will be saved 
    for idx, data in enumerate(list(data_list.keys())): 
        up_scale = int(data.split('_')[-1])
        for ids in list(data_list[data][0].keys()):
            list_data_set += [(data, ids) for _ in range(up_scale)]
        
        if (args.verbose):
            len_data_scale = len(data_list[data][0].keys())*up_scale
            tot_len_set = len(list_data_set)
            print(f"Data {data} Upscale {up_scale}  Len data scale {len_data_scale} Tot len set {tot_len_set}")

    return list_data_set


def write_record(args, data_list, list_data_set): 
    # Save the data to shards number of files.
    shards = args.shards
                    
    index_item = np.random.choice(len(list_data_set), size = (len(list_data_set),), replace = False)
    shards_range = np.linspace(0,len(list_data_set), num = shards)

    # require output dir
    if not os.path.isdir(args.output):
        os.mkdirs(args.output)
    
    print(args.output)
    for shard in range(shards-1):
        name = os.path.join(args.output, str(shard) +".tfrecord")
        if(args.verbose):
            print(f"Saving shard: {shard} with sequences between {shards_range[shard]} and {shards_range[shard+1]}")
        print(name)
        with tf.io.TFRecordWriter(name) as tfrecord:
            for index in index_item[int(shards_range[shard]):int(shards_range[shard+1])] if shard != shard-2 else index_item[int(shards_range[shard]):]:
                idx, ids = list_data_set[index]
                features = {
                  'class': _float_feature(float(data_list[idx][0][ids][1])-50.5),  # 50.5 is ofset (103-2)/2
                  'seq': _int64_feature(data_list[idx][0][ids][0])
                  }
                element = tf.train.Example(features = tf.train.Features(feature=features))
                tfrecord.write(element.SerializeToString())




def main(args):
    
    data_list = load_sequences(args)
    scaled_data_list = scale_sequences(args, data_list)
    write_record(args, data_list, scaled_data_list)
    
    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)