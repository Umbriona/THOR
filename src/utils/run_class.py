import matplotlib.pyplot as plt
import glob, subprocess
import tensorflow as tf
import pandas as pd
import numpy as np
import threading
import shutil

from Bio.Blast.Applications import NcbiblastxCommandline
from threading import Thread
from check_blast import *


def mkdir(thisDir):
	if os.path.exists(thisDir):
		shutil.rmtree(thisDir)
	os.makedirs(thisDir)


# Build a DB from some fasta file from a list of fastas
db_contents = ['/Users/niktat/Desktop/callback_function/fasta_file/train_meso_A0A090YHU4.fasta', '/Users/niktat/Desktop/callback_function/query_fasta/train_meso_F6F3S2.fasta']

# query_fasta = ['/Users/niktat/Desktop/callback_function/query_fasta/train_meso_F6F3S2.fasta']
matrix_type = ['BLOSUM45', 'BLOSUM62']

# Emulate generated sequences
generated = (glob.glob('/Users/niktat/Desktop/callback_function/query_fasta/*'))

db_location = db_contents[0]

# Directories
temp_dir = 'tempDir/'
result_dir = 'results/'
tensorBoard = 'board/logdir/'

mkdir(temp_dir)
mkdir(result_dir)
mkdir(tensorBoard)

output_file = 'blast_out'
out_format = '"6 qseqid sseqid score pident"'
query_meso = queryDB(output_file, out_format, [db_location], temp_dir, result_dir,  matrix_type[0])

# Scores - odd numbers
logger1 = tf.summary.create_file_writer(f'{tensorBoard}max_scores')
logger3 = tf.summary.create_file_writer(f'{tensorBoard}med_scores')
logger5 = tf.summary.create_file_writer(f'{tensorBoard}min_scores')

# Identities - even numbers
logger0 = tf.summary.create_file_writer(f'{tensorBoard}max_id')
logger2 = tf.summary.create_file_writer(f'{tensorBoard}med_id')
logger4 = tf.summary.create_file_writer(f'{tensorBoard}min_id')

# Histograms
loggerH = tf.summary.create_file_writer(f'{tensorBoard}hist')

test_summary_writer = [logger0, logger1, logger2, logger3, logger4, logger5, loggerH]

mutex = threading.RLock()#threading.Semaphore()

for epoch in range (len(generated)):
	print (epoch)
	in_queue = Thread(target=query_meso, args=(generated[epoch], epoch, test_summary_writer, mutex, ))#.start()
	in_queue.start()
	print ('Do other ThermalGan things here.')

# delete temp file queue
with mutex:
	queue_del = Thread(target=shutil.rmtree(temp_dir))
	queue_del.start()
