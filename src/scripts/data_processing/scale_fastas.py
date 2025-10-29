#! /usr/bin/env python

import os
from multiprocessing import Pool
import argparse

import numpy as np
import tensorflow as tf 
from Bio import SeqIO


# Create lookup table
TABLE = {aa: i-1 for i, aa in enumerate(AAS)}
TABLE['U'] = -1


parser = argparse.ArgumentParser(""" """)

parser.add_argument('-i', '--input', nargs='+', type=str, required = True,
                   help = "input files")
parser.add_argument('-v', '--verbose', action="store_true")

def get_upscale():

    return 

def main(args):
    
    
    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)