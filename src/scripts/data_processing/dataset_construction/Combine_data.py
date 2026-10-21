import os

from Bio import SeqIO


import argparse

parser = argparse.ArgumentParser(""" Combines fasta files""")

parser.add_argument('-i', '--input', nargs='+', type=str, required=True, 
                    help = "Fasta files to ")
parser.add_argument('-o', '--output', type=str, required = True)

parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--min_length', type=int, default=100)

# Define

FASTA_STRING = ">{} {}\n{}\n"

def write_fasta(data, args):
    with open(args.output, "w") as file_writer:
        for i, rec in enumerate(data.items()):
            file_writer.write(FASTA_STRING.format(rec[0], rec[1][0], rec[1][1]))
    print("Written file {} with {:9.0f} sequences".format(args.output, i))
    return 0

def read_fasta(args):
    data = {}

    for file in args.input:
        too_long = 0
        too_short= 0
        count=0
        for i, rec in enumerate(SeqIO.parse(file, "fasta")):
            if len(str(rec.seq)) > args.max_length:
                too_long += 1
            elif len(str(rec.seq)) < args.min_length:
                too_short += 1
            else:
                count += 1
                data[rec.id] = (rec.description.split()[-1], rec.seq)
                
        print("Loaded {:9.0f} sequences from {} {:6.0f} was too long {:6.0f} was too short".format(count, file, too_long, too_short))
    return data

def main(args):
    data = read_fasta(args)
    write_fasta(data, args)
    print("Done!")
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)