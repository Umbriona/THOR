import os, shutil
import tarfile
import math
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import argparse

# Defines
FASTA_STRING = ">{} {}\n{}\n"
TMP_DIR = "/tmp/tmp_img"

parser = argparse.ArgumentParser("""

This program Takes in a directory 


""")

parser.add_argument("-i", "--input", type = str, required=True)
parser.add_argument("-m", "--meta_data", type = str, default = None)
parser.add_argument("-o", "--output", type = str, required=True)

parser.add_argument('--use_only_published', action="store_true")
parser.add_argument('--skip_property', action="store_true", 
                    help="Include all sequences regardless if properties are not avalable")
parser.add_argument('--property', type=int, choices=[0,1,2], default = 1, 
                    help = """What property will be stored \n
                    0: None all sequences will be written \n 
                    1: Temperature, only those sequences with recorded sample temperature will be written
                    2: PH, only those sequences with recorded sample PH will be written""")

parser.add_argument('-v', '--verbose', type=int, choices=[0,1,2], default = 1)

def chec_files(args):
    if os.path.isfile(args.output):
        print("The file {} already exist. Cowardly exist ... ".format(args.output))
        exit()
    
def clean_temp_files(args):
    shutil.rmtree(os.path.join(TMP_DIR,os.path.basename(args.output)))

def set_prop(args):
    if args.property == 0:
        prop = "None"
    elif args.property == 1:
        prop = "Sample Collection Temperature"
    elif args.property == 2:
        prop = "pH"
    else:
        prop = None
    return prop

def extract_all(args):
    
    #make sure that the directory where the files are extracted to is empty 
    if not os.path.isdir(os.path.join(TMP_DIR,os.path.basename(args.output))):
        os.mkdir(os.path.join(TMP_DIR,os.path.basename(args.output)))
          
    with tarfile.open(args.input, 'r') as file_reader:
        file_reader.extractall(os.path.join(TMP_DIR,os.path.basename(args.output)))
#        base_name = os.path.basename(args.input).split('.')[0]
#        base_dir = os.path.basedir(args.input)
#        directory = os.path.join(base_dir, base_name)
#    return directory

def get_meta(df_meta, genome, prop, args):
    genome_id = int(genome.split('.')[0])

    # get meta
    row = df_meta.loc[df_meta["IMG Genome ID "] == genome_id]
    
    sample_prop = row[prop]
    try:
        nan = math.isnan(sample_prop.values[0])
        if nan:
            val = '-'
    except TypeError:
        val = sample_prop.values[0].split(" ")[0]
        unit = sample_prop.values[0].split(" ")[-1]
        if unit != "C":
            raise TypeError("Conversion from {} not implemented needs to be C".format(unit))
    except:
        raise ValueError("Value of type {} for Sample Collection Temp is not permitted".format(type(sample_prop.values[0]))) 
    sample_prop = val
    range_temp  = row["Temperature Range"].values[0]
    
    if row['Is Published'].values == "Yes":
        is_published = True
    else:
        is_published = False
    
    
    return genome_id, range_temp, sample_prop, is_published 
        
def get_seqs(f, genome_id, range_temp, sample_prop):
    # get seqs
    prot_fasta_file = [info.name for info in f.getmembers() if info.name[-3:] == "faa"]
    prot_fasta_file_obj = f.extractfile(prot_fasta_file[0])

    prot_fasta_file = StringIO("".join([line.decode('utf-8') for line in prot_fasta_file_obj.readlines()]))

    #collect data

    fasta_dict = {"Id": [], "records": [], "seq": []}
    for rec in SeqIO.parse(prot_fasta_file,"fasta"):
        fasta_dict["Id"].append(rec.id)
        fasta_dict["records"].append(rec.description + "| {} | {} | {}".format(genome_id, range_temp , sample_prop))
        fasta_dict["seq"].append(str(rec.seq))
    return fasta_dict
    

def main(args):
    # preliminary checks
    chec_files(args)
    prop = set_prop(args)
    # Extracting files
    extract_all(args)
    df_meta = pd.read_csv(args.meta_data, sep='\t')
    
    for genome in os.listdir(os.path.join(TMP_DIR,os.path.basename(args.output))):
        if genome[-7:] == ".tar.gz" :
            # Get meta data
            genome_id, range_temp, sample_prop, is_published = get_meta(df_meta, genome, prop ,args)
            
            # Only include valid genomes in data set
            if args.use_only_published and not is_published:
                print("Genome {} is not published and will not be included in the final data set".format(genome[:-7]))
                continue
            if sample_prop == '-' and not args.skip_property:
                print("Genome {} has no recorded {} and will not be included in the final data set".format(genome[:-7], prop))
                continue
            # Get Sequence data
            with tarfile.open(os.path.join(TMP_DIR,os.path.basename(args.output),genome), 'r') as file_reader:
                fasta_dict = get_seqs(file_reader, genome_id, range_temp, sample_prop)
            # Write fasta 
            n_seqs = len(fasta_dict["Id"])
            temperature = fasta_dict["records"][0].split(" | ")[-1]
            with open(args.output, "a") as d:
                for seq_id, seq_rec, seq in zip(fasta_dict["Id"], fasta_dict["records"], fasta_dict["seq"]):
                    d.write(FASTA_STRING.format(seq_id,seq_rec,seq))
            print("Genome {} with {} sequences and {} sample temp was written to data set ".format(genome[:-7], n_seqs, temperature))
        else:
            print("skipping file {}".format(genome))
    
    clean_temp_files(args)
    print("Done")
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
