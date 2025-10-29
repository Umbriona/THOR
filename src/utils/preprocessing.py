import numpy as np
from Bio import SeqIO
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

def convert_table(seq, w):    
    aas = 'ACDEFGHIKLMNPQRSTVWYX'
    dict_ = {i:aa for i, aa in enumerate(aas)}
    seq_str = "".join([dict_[res] for res in seq[w==1]])
    return seq_str 

def synt2binary(seq):
    bin_seq = np.zeros((100,5), dtype=np.float32)

    for i in range(100):
        bin_seq[i, seq[i]] += 1 
    return bin_seq

def to_binary(seq, max_length, start_stop = False):
    # eoncode non-standard amino acids like X as all zeros
    # output a array with size of L*20
    seq = seq.upper()
    if not start_stop:
        aas = 'ACDEFGHIKLMNPQRSTVWYX'
        vocab=21
    else:
        aas = 'ACDEFGHIKLMNPQRSTVWYX<>'
        vocab=23
    pos = dict()
    for i in range(len(aas)): pos[aas[i]] = i
    
    binary_code = dict()
    for aa in aas: 
        code = np.zeros(vocab, dtype = np.float32)
        code[pos[aa]] = 1
        binary_code[aa] = code
    
    seq_coding = np.zeros((max_length,vocab), dtype = np.float32)
    for i,aa in enumerate(seq): 
        code = binary_code.get(aa,np.zeros(vocab, dtype = np.float32))
        seq_coding[i,:] = code
    return seq_coding

def to_int(seq, max_length, start_stop = False):
    seq.upper()
    if not start_stop:
        aas = 'ACDEFGHIKLMNPQRSTVWYX'
    else:
        aas = 'ACDEFGHIKLMNPQRSTVWYX<>'
  
    d = dict()
    for i in range(len(aas)): d[aas[i]] = i

    tmp =np.array([d[i] if i in aas else 20 for i in seq])
    out = np.ones((max_length,))*20
    index = tmp.size if tmp.size<max_length else max_length
    out[:index] = tmp[:index]
    return out

def loss_weight(mask, max_length):
    len_seq = len(mask)
    seq_w = [1 for i in mask] 
    tmp = np.ones((max_length,))
    tmp[:len_seq] = seq_w
    tmp[len_seq:] = 0.0
    return tmp




    
def zero_padding(inp,length=500,start=False):
    # zero pad input one hot matrix to desired length
    # start .. boolean if pad start of sequence (True) or end (False)
    #assert len(inp) <= length
    out = np.zeros((length,inp.shape[1]))
    if start:
        out[-inp.shape[0]:] = inp[:length,:]
    else:
        out[0:inp.shape[0]] = inp[:length,:]
    return out

def prepare_dataset(file_name, file_format = 'fasta', seq_length = 1024, t_v_split = 0.1,start_stop = False, max_samples = 5000):
    
    if start_stop:
        seq_length -= 2

    count=0
    dict_ = {'id':[] ,'mask':[],'seq':[], 'mask_bin':[], 'seq_bin':[], 'loss_weight':[], 'seq_int':[]}
    # loading data to dict
    for i, rec in enumerate(SeqIO.parse(file_name, file_format)):
        count +=1
        if count >max_samples:
            break
        if len(rec.seq)>seq_length:
            continue
        dict_['id'].append(rec.id)
        if not start_stop:
            dict_['seq'].append(str(rec.seq))
            dict_['loss_weight'].append(loss_weight(rec.seq ,seq_length))
            dict_['seq_int'].append(to_int(rec.seq, max_length=seq_length))
            dict_['seq_bin'].append(to_binary(rec.seq, max_length=seq_length))
        else:
            str_seq = '<'+str(rec.seq)+'>'
            dict_['seq'].append(str(rec.seq))
            dict_['loss_weight'].append(loss_weight(str_seq ,seq_length+2))
            dict_['seq_int'].append(to_int(str_seq, max_length=seq_length+2, start_stop=start_stop))
            dict_['seq_bin'].append(to_binary(str_seq, max_length=seq_length+2, start_stop=start_stop))
   # Splitting data to training and validation sets

    int_train, int_test, W_train, W_test, bin_train, bin_test, id_train, id_test = train_test_split(np.array(dict_['seq_int'],dtype=np.int8),
                                                    np.array(dict_['loss_weight'],dtype=np.float32),
                                                    np.array(dict_['seq_bin'], dtype = np.float32),
                                                    dict_['id'],
                                                    test_size=t_v_split, random_state=42)
    n_train = int_train.shape[0]
    n_test  = int_test.shape[0]
    dataset_train = tf.data.Dataset.from_tensor_slices((id_train,bin_train,W_train))
    dataset_validate = tf.data.Dataset.from_tensor_slices((id_test,bin_test,W_test))
    return dataset_train, dataset_validate, n_train, n_test
    
def prepare_dataset_U_N(file_path, file_name, file_format = 'fasta', seq_length = 1024, t_v_split = 0.1):
    

    count=0
    dict_ = {'id':[] ,'mask':[],'seq':[], 'mask_bin':[], 'seq_bin':[], 'loss_weight':[], 'seq_int':[]}
    # loading data to dict
    for i, rec in enumerate(SeqIO.parse(os.path.join(file_path,file_name),'fasta')):
        count +=1
        if count >10000:
            break
        if len(rec.seq)>seq_length:
            continue
        dict_['id'].append(rec.id)
        dict_['seq'].append(rec.seq)
        dict_['loss_weight'].append(loss_weight(rec.seq ,seq_length))
        dict_['seq_int'].append(to_int(rec.seq, max_length=seq_length))
        dict_['seq_bin'].append(to_binary(rec.seq, max_length=seq_length))
   # Splitting data to training and validation sets

    int_train, int_test, W_train, W_test, bin_train, bin_test, id_train, id_test = train_test_split(np.array(dict_['seq_int'],dtype=np.int8),
                                                    np.array(dict_['loss_weight'],dtype=np.float32),
                                                    np.array(dict_['seq_bin'], dtype = np.float32),
                                                    dict_['id'],
                                                    test_size=t_v_split, random_state=42)
   
    dataset_train = tf.data.Dataset.from_tensor_slices((int_train,bin_train,W_train))
    dataset_validate = tf.data.Dataset.from_tensor_slices((int_test,bin_test,W_test))
    return dataset_train, dataset_validate, 

def calc_class(val):
    if val <=15:
        return 0
  #  elif val > 15 and val<=26:
  #      return 1
    elif val > 26 and val<=37:
        return 1
   # elif val > 37 and val<=48:
  #      return 3
    elif val > 48 and val<=59:
        return 2
   # elif val > 59 and val<=70:
   #     return 5
    elif val > 70:
        return 3


def prepare_dataset_class(file_path, file_names,
                               file_format = 'fasta', 
                               seq_length = 1024,
                               t_v_split = 0.1,
                               max_samples = 5000):
    
    

    dict_ = {'id':[] ,
             'ogt':[],
             'mask':[],
             'seq':[], 
             'mask_bin':[],
             'seq_bin':[],
             'loss_weight':[],
             'seq_int':[]}
    # loading data to dict
    for name in file_names:
        count = 0
        for i, rec in enumerate(SeqIO.parse(os.path.join(file_path, name),'fasta')):
            if len(rec.seq)>seq_length:
                continue
            if count >max_samples:
                break
            count += 1
            dict_['id'].append(rec.id)
            c = calc_class(float(rec.description.split()[-1]))
            arr = np.zeros((4,))
            arr[c] = 1
            dict_['ogt'].append(arr)
            dict_['seq_bin'].append(to_binary(str(rec.seq), max_length=seq_length))
        print(name, count)
   # Splitting data to training and validation sets
    bin_train, bin_test, ogt_train, ogt_test = train_test_split(np.array(dict_['seq_bin'], dtype = np.float32),
                                                                 np.array(dict_['ogt'], dtype = np.float32),
                                                                 test_size=t_v_split, random_state=42)

    dataset_train = tf.data.Dataset.from_tensor_slices((bin_train, ogt_train))
    dataset_validate = tf.data.Dataset.from_tensor_slices((bin_test,ogt_test))
    return dataset_train, dataset_validate

def prepare_dataset_reg(file_path, file_names,
                               file_format = 'fasta', 
                               seq_length = 1024,
                               t_v_split = 0.1,
                               max_samples = 5000):
    
    

    dict_ = {'id':[] ,
             'ogt':[],
             'mask':[],
             'seq':[], 
             'mask_bin':[],
             'seq_bin':[],
             'loss_weight':[],
             'seq_int':[]}
    # loading data to dict
    for name in file_names:
        count = 0
        for i, rec in enumerate(SeqIO.parse(os.path.join(file_path, name),'fasta')):
            if len(rec.seq)>seq_length:
                continue
            if count >max_samples:
                break
            count += 1
            dict_['id'].append(rec.id)
            dict_['ogt'].append(float(rec.description.split()[-1])-41.9)
            dict_['seq_bin'].append(to_binary(str(rec.seq), max_length=seq_length))
        print(name, count)
   # Splitting data to training and validation sets
    if t_v_split > 0:
        bin_train, bin_test, ogt_train, ogt_test = train_test_split(np.array(dict_['seq_bin'], dtype = np.float32),
                                                                     np.array(dict_['ogt'], dtype = np.float32),
                                                                     test_size=t_v_split, random_state=42)

        dataset_train = tf.data.Dataset.from_tensor_slices((bin_train, ogt_train))
        dataset_validate = tf.data.Dataset.from_tensor_slices((bin_test,ogt_test))
        return dataset_train, dataset_validate
    else:
        dataset_validate = tf.data.Dataset.from_tensor_slices((np.array(dict_['seq_bin'], dtype = np.float32),
                                                               np.array(dict_['ogt'], dtype = np.float32)))
        return  dataset_validate
    


def _parse_function_reg(item):
    feature_description = {
    'temp': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'seq': tf.io.FixedLenFeature([512], tf.int64, default_value=np.zeros((512,)))
}
  # Parse the input `tf.train.Example` proto using the dictionary above.
    item = tf.io.parse_single_example(item, feature_description)
    item = (tf.one_hot(item['seq'],21, off_value=0.0), item["temp"])
    return item

def _parse_function_in(item):
    feature_description = {
    'temp': tf.io.FixedLenFeature([], tf.float32, default_value = 0.0),
    'seq': tf.io.FixedLenFeature([512], tf.int64, default_value = -np.ones((512,)))
}
  # Parse the input `tf.train.Example` proto using the dictionary above.
    item = tf.io.parse_single_example(item, feature_description)
    item = (item['seq'], 1.0)
    return item

def _parse_function_out(item):
    feature_description = {
    'temp': tf.io.FixedLenFeature([], tf.float32, default_value = 0.0),
    'seq': tf.io.FixedLenFeature([512], tf.int64, default_value = -np.ones((512,)))
}
  # Parse the input `tf.train.Example` proto using the dictionary above.
    item = tf.io.parse_single_example(item, feature_description)
    item = (item['seq'], 0.0)
    return item

def _parse_function_onehot(item, key = "class"):
    feature_description = {
    key : tf.io.FixedLenFeature([], tf.float32, default_value = 0.0),
    'seq': tf.io.FixedLenFeature([512], tf.int64, default_value = -np.ones((512,)))
}
  # Parse the input `tf.train.Example` proto using the dictionary above.
    item = tf.io.parse_single_example(item, feature_description)
    item = (tf.one_hot(item['seq'],21, off_value=0.0), item[key])
    return item

def _parse_function_onehot_cycle(item, key = "class", model = "classifier"):
    feature_description = {
    key : tf.io.FixedLenFeature([], tf.float32, default_value = 0.0),
    'seq': tf.io.FixedLenFeature([512], tf.int64, default_value = -np.ones((512,)))
}
  # Parse the input `tf.train.Example` proto using the dictionary above.
    item = tf.io.parse_single_example(item, feature_description)
    y = tf.constant([0], dtype=tf.int64)
    item = (tf.one_hot(item['seq'],21, off_value=0.0), item[key], tf.reshape(tf.cast(tf.math.greater_equal(item['seq'], y), tf.float32), shape=(512,1)))
    return item

def parse_upsample(name):
    up_sample = int(name.split('.')[0].split('_')[-1])
    return up_sample

def parse_ofset(name):
    temp_low = int(name.split('_')[0])
    temp_high= int(name.split('_')[1])
    return temp_high - temp_low
    
def load_data(config, model = "classifier"):
    # get file names and paths
    base_dir = config["base_dir"]
    files_train = os.path.join(base_dir, config["train_dir"])
    files_val = os.path.join(base_dir, config["val_dir"])
    
    num_shards = 100
    #num_shards = len(os.listdir(files_train))

    # get file names      Dataset
    print(f"Loading records from {files_train}\nTaking {config['n_shards']} records")

    file_names_train = tf.data.Dataset.list_files(files_train).take(config["n_shards"])# .shuffle(config["n_shards"])
    file_names_val = tf.data.Dataset.list_files(files_val)# .shuffle(config["n_shards"])

    # load and parse data from in group

    tfdata_train = file_names_train.interleave(lambda filename: tf.data.TFRecordDataset(filename), num_parallel_calls = tf.data.AUTOTUNE )
    tfdata_val = file_names_val.interleave(lambda filename: tf.data.TFRecordDataset(filename), num_parallel_calls = tf.data.AUTOTUNE )
    
    #tfdata_train = tfdata_train.cache()
    #tfdata_val = tfdata_val.cache()
    
    #if model == "classifier":
    #    tfdata_train = tfdata_train.map(_parse_function_onehot, num_parallel_calls = tf.data.AUTOTUNE)
    #    tfdata_val = tfdata_val.map(_parse_function_onehot, num_parallel_calls = tf.data.AUTOTUNE)
    #else:
    tfdata_train = tfdata_train.map(_parse_function_onehot_cycle, num_parallel_calls = tf.data.AUTOTUNE)
    tfdata_val = tfdata_val.map(_parse_function_onehot_cycle, num_parallel_calls = tf.data.AUTOTUNE)
    
    tfdata_train = tfdata_train.shuffle(int(2e4), reshuffle_each_iteration=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    tfdata_val = tfdata_val.shuffle(int(2e4), reshuffle_each_iteration=True).prefetch(buffer_size=tf.data.AUTOTUNE)

    

    return tfdata_train, tfdata_val

def load_data_quick(config, model = "classifier"):
    # get file names and paths
    base_dir = config["base_dir"]
    files_train = os.path.join(base_dir, config["train_dir"])
    files_val = os.path.join(base_dir, config["val_dir"])

    tfdata_train =  tf.data.TFRecordDataset(files_train)
    tfdata_val = tf.data.TFRecordDataset(files_val)

    tfdata_train = tfdata_train.map(_parse_function_onehot_cycle, num_parallel_calls = tf.data.AUTOTUNE)
    tfdata_val = tfdata_val.map(_parse_function_onehot_cycle, num_parallel_calls = tf.data.AUTOTUNE)
    
    tfdata_train = tfdata_train.cache().shuffle(int(1e4), reshuffle_each_iteration=True)
    tfdata_val = tfdata_val.cache().shuffle(int(1e4), reshuffle_each_iteration=True)

    

    return tfdata_train, tfdata_val

def _parse_function_bert_cycle(item, max_length = 512, key = "class"):
    feature_description = {
    'seq': tf.io.FixedLenFeature([max_length], tf.int64, default_value = -np.ones((max_length,))),
     key : tf.io.FixedLenFeature([], tf.float32, default_value = 0.0),
    'probability': tf.io.FixedLenFeature([21*max_length], tf.float32, default_value = np.zeros((21*max_length,))),
    'length': tf.io.FixedLenFeature([], tf.int64, default_value = 0)
     }
    
  # Parse the input `tf.train.Example` proto using the dictionary above.
    item = tf.io.parse_single_example(item, feature_description)
    y = tf.constant([0], dtype=tf.int64)
    item = (tf.one_hot(item['seq'],21, off_value=0.0),
            item[key],
            tf.reshape(item['probability'], shape = (512, 21)),
            item["length"],
            tf.reshape(tf.cast(tf.math.greater_equal(item['seq'], y), tf.float32), shape=(max_length,1)))
    return item



def _parse_function_bert_cycle_inference(item, max_length = 512, key = "class"):
    feature_description = {
    'seq': tf.io.FixedLenFeature([max_length], tf.int64, default_value = -np.ones((max_length,))),
     key : tf.io.FixedLenFeature([], tf.float32, default_value = 0.0),
    'probability': tf.io.FixedLenFeature([21*max_length], tf.float32, default_value = np.zeros((21*max_length,))),
    'length': tf.io.FixedLenFeature([], tf.int64, default_value = 0),
    'id': tf.io.FixedLenFeature([], tf.string)
     }
    
  # Parse the input `tf.train.Example` proto using the dictionary above.
    item = tf.io.parse_single_example(item, feature_description)
    y = tf.constant([0], dtype=tf.int64)
    item = (tf.one_hot(item['seq'],21, off_value=0.0),
            item[key],
            tf.reshape(item['probability'], shape = (512, 21)),
            item["length"],
            tf.reshape(tf.cast(tf.math.greater_equal(item['seq'], y), tf.float32), shape=(max_length,1)),
            item['id'])
    return item


def load_data_bert(config, data_set):
    base_dir = config[data_set]["base_dir"]
    files_train = os.path.join(base_dir, config[data_set]["train_dir"])
    files_val = os.path.join(base_dir, config[data_set]["val_dir"])
    max_length = config[data_set]["max_length"]
    print(max_length)
    seed = int(config.get("seed", 1337))
    num_shards = 100
    #num_shards = len(os.listdir(files_train))

    # get file names      Dataset
    print(f"Loading records from {files_train}\nTaking {config[data_set]['n_shards']} records")

    file_names_train = tf.data.Dataset.list_files(files_train,shuffle=True).take(config[data_set]["n_shards"]) #.shuffle(config["n_shards"])
    file_names_val = tf.data.Dataset.list_files(files_val,shuffle=False) #.shuffle(config["n_shards"])

    tfdata_train = tf.data.TFRecordDataset(file_names_train, num_parallel_reads = tf.data.AUTOTUNE) #file_names_train.interleave(lambda filename: tf.data.TFRecordDataset(filename), num_parallel_calls = tf.data.AUTOTUNE )
    tfdata_val = tf.data.TFRecordDataset(file_names_train, num_parallel_reads = tf.data.AUTOTUNE) #file_names_val.interleave(lambda filename: tf.data.TFRecordDataset(filename), num_parallel_calls = tf.data.AUTOTUNE )

    tfdata_train = tfdata_train.map(lambda x: _parse_function_bert_cycle(x, max_length=max_length), num_parallel_calls = tf.data.AUTOTUNE)
    tfdata_val = tfdata_val.map(lambda x: _parse_function_bert_cycle(x, max_length=max_length), num_parallel_calls = tf.data.AUTOTUNE)
    
    tfdata_train = tfdata_train.shuffle(int(2e4), reshuffle_each_iteration=True).batch(config['CycleGan']['batch_size'], drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    tfdata_val = tfdata_val.cache().batch(256, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return tfdata_train, tfdata_val

def load_data_bert_inference(config):
    base_dir = config["base_dir"]
    files_train = os.path.join(base_dir, config["train_dir"])
    files_val = os.path.join(base_dir, config["val_dir"])
    max_length = config["max_length"]
    print(max_length)
    # get file names      Dataset
    print(f"Loading records from {files_train}\nTaking {config['n_shards']} records")

    file_names_train = tf.data.Dataset.list_files(files_train, shuffle=False)
    file_names_val = tf.data.Dataset.list_files(files_val, shuffle=False)

    tfdata_train = file_names_train.interleave(lambda filename: tf.data.TFRecordDataset(filename), num_parallel_calls = tf.data.AUTOTUNE )
    tfdata_val = file_names_val.interleave(lambda filename: tf.data.TFRecordDataset(filename), num_parallel_calls = tf.data.AUTOTUNE )

    tfdata_train = tfdata_train.map(lambda x: _parse_function_bert_cycle_inference(x, max_length=max_length), num_parallel_calls = tf.data.AUTOTUNE)
    tfdata_val = tfdata_val.map(lambda x: _parse_function_bert_cycle_inference(x, max_length=max_length), num_parallel_calls = tf.data.AUTOTUNE)
    
    tfdata_train = tfdata_train.prefetch(buffer_size=tf.data.AUTOTUNE)
    tfdata_val = tfdata_val.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return tfdata_train, tfdata_val

MAX_LEN = 512
VOCAB = 21  # AA classes incl. gap

# ---- defaults for missing features ----
DEFAULT_PROBS_BYTES = tf.constant(b"\x00" * (20 * MAX_LEN))  # zeros → 0.0 probs
DEFAULT_SEQ_BYTES   = tf.constant(b"\xff" * MAX_LEN)        # 0xFF → -1 when decoded as int8

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard AAs
VOCAB = 21                         # 20 + extra blank column to match your probs [*,21]

# --- Build a lookup table once (outside the map/parse function) ---
aa_chars = tf.constant(list(AA_ORDER))  # ["A","C",...,"Y"]
aa_ids   = tf.range(len(AA_ORDER), dtype=tf.int32)  # [0..19]
AA_TABLE = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(aa_chars, aa_ids),
    default_value=-1  # unknowns/gaps map to -1
)

def parse_example(item):
    feature_description = {
        "probs_q": tf.io.FixedLenFeature([], tf.string, default_value=DEFAULT_PROBS_BYTES),
        "seq":     tf.io.FixedLenFeature([], tf.string,  default_value=DEFAULT_SEQ_BYTES),
        "length":  tf.io.FixedLenFeature([], tf.int64,   default_value=0),
        "temp":    tf.io.FixedLenFeature([], tf.float32),
        "id":      tf.io.FixedLenFeature([], tf.string,  default_value=b""),
    }
    f = tf.io.parse_single_example(item, feature_description)
    return f

def probs_q_to_512x21(f):
    # uint8 bytes -> [L*20] -> [L,20] -> pad/clip -> add zero column

    raw = tf.io.parse_tensor(f["probs_q"], out_type=tf.uint8)

    raw = tf.pad(raw, [[0, MAX_LEN - tf.shape(raw)[0]], [0,0]])    # [20*MAX_LEN]
    raw = tf.concat([raw, tf.zeros([MAX_LEN, 1], raw.dtype)], axis=1)
    return raw   # [20*MAX_LEN]

def seq_bytes_to_ids(seq_bytes, max_length=MAX_LEN, records_type="ESM"):
    """
    seq_bytes: scalar tf.string like b'M A K ...'
    returns: int32 [max_length] with ids in 0..19, -1 for pad/unknown
    """
    # Normalize & split on spaces (handles multiple spaces too)
    s = tf.strings.upper(seq_bytes)
    if records_type!="ESM":
        tokens = tf.strings.split(s)          # ["M","A","K",...]
    else:
        tokens = tf.strings.bytes_split(s) 
    # Lookup letter -> id (unknowns -> -1)
    ids = AA_TABLE.lookup(tokens)         # [L]
    # Clip/pad to max_length with -1
    ids = ids[:max_length]
    ids = tf.pad(ids, [[0, max_length - tf.shape(ids)[0]]], constant_values=-1)
    ids = tf.ensure_shape(ids, [max_length])
    return ids

# -------- example parser combining all ----------
def _compact_parse_function_bert_cycle(item, max_length=512, key="class"):
    f = parse_example(item)
    probs_512x21 = probs_q_to_512x21(f)          # [512,21]
    seq_ids = seq_bytes_to_ids(f["seq"], max_length=max_length, records_type="prot_bert")
    example = (
        seq_ids,                          # [L]
        f["temp"],                            # scalar float32
        probs_512x21,)                       # [L,21] float32
    return example  

def load_compact_data_bert(config, data_set):
    base_dir = config[data_set]["base_dir"]
    files_train = os.path.join(base_dir, config[data_set]["train_dir"])
    files_val = os.path.join(base_dir, config[data_set]["val_dir"])
    max_length = config[data_set]["max_length"]
    print(max_length)
    seed = int(config.get("seed", 1337))
    num_shards = 100
    print(f"Loading records from {files_train}\nTaking {config[data_set]['n_shards']} records")

    file_names_train = tf.data.Dataset.list_files(files_train,shuffle=False) #.take(config[data_set]["n_shards"]) #.shuffle(config["n_shards"])
    file_names_val = tf.data.Dataset.list_files(files_val,shuffle=False) #.shuffle(config["n_shards"])

    tfdata_train = tf.data.TFRecordDataset(file_names_train, num_parallel_reads = tf.data.AUTOTUNE,compression_type="GZIP") # tf.data.AUTOTUNE file_names_train.interleave(lambda filename: tf.data.TFRecordDataset(filename), num_parallel_calls = tf.data.AUTOTUNE )
    tfdata_val = tf.data.TFRecordDataset(file_names_val, num_parallel_reads = tf.data.AUTOTUNE, compression_type="GZIP") #file_names_val.interleave(lambda filename: tf.data.TFRecordDataset(filename), num_parallel_calls = tf.data.AUTOTUNE )

    tfdata_train = tfdata_train.map(lambda x: _compact_parse_function_bert_cycle(x, max_length=max_length), num_parallel_calls = tf.data.AUTOTUNE).cache()
    tfdata_val = tfdata_val.map(lambda x: _compact_parse_function_bert_cycle(x, max_length=max_length), num_parallel_calls = tf.data.AUTOTUNE)
    
    tfdata_train = tfdata_train.shuffle(int(5e4), reshuffle_each_iteration=True)
    tfdata_val = tfdata_val
    
    return tfdata_train, tfdata_val

