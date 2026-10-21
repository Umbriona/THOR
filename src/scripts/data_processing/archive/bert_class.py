from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, AutoModelForMaskedLM
import transformers
import numpy as np
#from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
from  time import time
import torch
#from multiprocessing import Pool
from torch.multiprocessing import Pool, set_start_method
transformers.logging.set_verbosity_error()

set_start_method('spawn', force=True)

PARTITION_SIZE = 20

class MaskedSequenceDataset(Dataset):
    def __init__(self, seqs, idxs, _ids, tokenizer, max_length):
        self.texts = seqs
        self._id = _ids
        self.idxs=idxs
        self.tokenizer = tokenizer
        self.max_length = 512

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        _id = self._id[idx]
        _idx = self.idxs[idx]
        #encoded = self.tokenizer(text,  device="cuda:0", padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return text # , _idx, _id #, encoded['attention_mask'].squeeze()
        #encoded['input_ids'].squeeze(0)
class bertModels:
  def __init__(self, topK, modelName):
    self.topK = topK
    self.modelName = modelName

    if modelName not in ('BertRost','ESM'):
      print ('Invalid Model Name. Valid Names: "ESM" or "BertRost". Exiting...')
      exit()

    if modelName == 'BertRost':
      self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
      self.model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")

    if modelName == 'ESM':
      # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")  #ESM2 this is crap
      # self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D") #ESM2 this is crap
      # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
      # self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S")
      self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")
      self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")

    self.unmasker = pipeline('fill-mask', device="cuda:0", model=self.model, tokenizer=self.tokenizer, top_k = self.topK)

class bertModels_batch:
  def __init__(self, topK, modelName, device, batch_size):
    self.topK = topK
    self.modelName = modelName
    self.device = device
    self.batch_size =  batch_size

    if modelName not in ('BertRost','ESM'):
      print ('Invalid Model Name. Valid Names: "ESM" or "BertRost". Exiting...')
      exit()

    if modelName == 'BertRost':
      self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
      self.model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")

    if modelName == 'ESM':
      # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")  #ESM2 this is crap
      # self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D") #ESM2 this is crap
      #self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
      #self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S")
       self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")
       self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")

    try:
      device = "cuda:1" #int((device))
      print (device)
    except ValueError:
      print("No CUDA integer given, defaulting to CPU")
      device = 'cpu'

    self.unmasker = pipeline('fill-mask', model=self.model, device = self.device, tokenizer=self.tokenizer, top_k = self.topK)


class BertPatcher(bertModels):
  def __init__(self, topK, modelName, seqA, seqB):
    super(BertPatcher, self).__init__(topK, modelName)

    self.seqA = seqA
    self.seqB = seqB

    if (self.seqA == self.seqB):
      print ('Sequences are the same. Exiting...')
      exit()

  def pre_masker(self, reference, target):
    muts = []
    try:
      assert (len(reference) == len(target))
    except:
      print ('Error - Sequences must be of equal length!')
      exit()

    target = list(target)
    for i, ref in enumerate(reference):
      if ref != target[i]:
        # print (f'Got {target[i]}, expected {ref}')
        muts.append(target[i])
        target[i] = 'X'

    target = ''.join(target)
    return target, muts

  def str_prepare(self, orignal, mutant):
    idx_diff = [i for i in range(len(mutant)) if orignal[i] != mutant[i]]
    replacements = len(idx_diff)*['X']
    mutant = list(mutant)

    for (index, replacement) in zip(idx_diff, replacements):
      mutant[index] = replacement

    mutant = ' '.join(mutant)
    if self.modelName == 'BertRost':
      mutant = mutant.replace('X', '[MASK]')
      return mutant, idx_diff
    mutant = mutant.replace('X', '<mask>') # Masking style for ESM 
    return mutant, idx_diff

  def itter_dict(self, dct, idx, mut):
    mut_score = 0
    new_dct = {}

    # print (f'\nTop Substitutes for AA @ position {idx} (Mutation = {mut})')
    new_dct[idx] = {}
    for i in range (len(dct)):
      # print (f'Substitute AA: {dct[i].get("token_str")} with Score: {dct[i].get("score")}')
      new_dct[idx][dct[i].get("token_str")] = dct[i].get("score")
      if mut == dct[i].get("token_str"):
        mut_score = dct[i].get("score")

    max_like_score = dct[0].get("score")
    return  max_like_score, mut_score, new_dct


  def __call__(self):

    target, muts = self.pre_masker(self.seqA, self.seqB)
    mutant, idx_diff = self.str_prepare(self.seqA, target)
    print(f"Mutant: {mutant}\n Idx_diff: {idx_diff}")
    ans = self.unmasker(mutant)
    top_score = [] # this is top amino acid
    divider = len(ans)
    dict_all = {}
    sum1 = 0
    sum2 = 0

    for i in range (len(ans)):
      try:
        top_score.append(ans[i][0].get('token_str'))
        score, mut_score, new_dct = self.itter_dict(ans[i], idx_diff[i], muts[i])
        dict_all.update(new_dct)
        sum1 += score
        sum2 += mut_score
      except: # Single Mutation
        top_score.append(ans[i].get('token_str'))
        sum1, sum2, new_dct = self.itter_dict(ans, idx_diff[i], muts[i])
        dict_all.update(new_dct)
        divider = 1
        break
      # print('---------------')

    print (f'Top Substitutions suggested by BERT: {top_score}')
    print (f'Top Substitutions overall likelihood: {sum1/divider}')
    print (f'Overall likelihood of the original substitutions: {sum2/divider}')

    return dict_all


class BertLikelihood(bertModels_batch):
    def __init__(self, topK, modelName, device, batch_size):
        super(BertLikelihood, self).__init__(topK, modelName, device, batch_size)
        self.modelName = modelName
        if self.modelName == 'BertRost':
            self.mask = '[MASK]'
        else:
            self.mask = '<mask>' # Masking style for ESM


    def _check_lists(self, listA, listB):
        assert len(listA) == len(listB), 'Error - Lists must be of equal length!'
        
    def _check_string_equality(self, seqA, seqB):
        assert seqA != seqB, 'Some sequences are the same. Exiting...'

    def _make_substitution_index_list(self, seqA, seqB, _id):
        dict_tmp = {i:(resA, resB) for i, (resA, resB) in enumerate(zip(seqA, seqB)) if resA != resB}
        dict_substitutions = {"wt_seq":seqA,
                              "mut_seq":seqB,
                              "substitutions":dict_tmp, 
                              "id": [_id for _ in dict_tmp.keys()]}
        return dict_substitutions

        
    def _make_marginal_sequence_likelihood_masks(self, seqA, seqB):
        pass
    def make_marginal_likelihod_batch(self, listA, listB):
        pass

    def pre_masker(self, reference, target):
    
        all_targets = []
        all_muts = []
        
        for i in range (len(reference)):
            muts = []
            # print (len(reference))
            # print (len(reference[i]))

            assert len(reference[i]) == len(target[i]), 'Error - Sequences must be of equal length!'
            target[i] = list(target[i])
        
            new_ref = reference[i]
            new_target = target[i]
            
            for j, ref in enumerate(new_ref):
                if ref != new_target[j]:
                    muts.append(new_target[j])
                    new_target[j] = 'X'
            new_target = ''.join(new_target)
            all_targets.append(new_target)
            all_muts.append(muts)
           
        return all_targets, all_muts

    def str_prepare(self, orignal_all, mutant_all):
    
        idx_diff = []
        mut_lst = []
        idx_diff_lst = []
      
        for j in range (len(orignal_all)):
            orignal = orignal_all[j]
            mutant = mutant_all[j]
        
            idx_diff = [i for i in range(len(mutant)) if orignal[i] != mutant[i]]
            idx_diff_lst.append(idx_diff)
            replacements = len(idx_diff)*['X']
            mutant = list(mutant)
        
            for (index, replacement) in zip(idx_diff, replacements):
                mutant[index] = replacement
            mutant = ' '.join(mutant)
            if self.modelName == 'BertRost':
                mutant = mutant.replace('X', '[MASK]')
                mut_lst.append(mutant)
            else:
                mutant = mutant.replace('X', '<mask>') # Masking style for ESM
                mut_lst.append(mutant)
        
        return mut_lst, idx_diff_lst

    def _str_prepare_single_index(self, dict_substitutions):
    
        idx_diff = list(dict_substitutions["substitutions"])
        wildtype = list(dict_substitutions["wt_seq"])
        mutant = list(dict_substitutions["mut_seq"])
        wt_single_diff_list = []
        mut_single_diff_list = []
        for idx in idx_diff:
            tmp_wildtype = wildtype.copy()
            tmp_wildtype[idx] = self.mask
            wt_single_diff_list.append(" ".join(tmp_wildtype))
            tmp_mutant = mutant.copy()
            tmp_mutant[idx] = self.mask
            mut_single_diff_list.append(" ".join(tmp_mutant))
        dict_substitutions["single_sub_masked_wt"] = wt_single_diff_list
        dict_substitutions["single_sub_masked_mut"] = wt_single_diff_list
        return dict_substitutions

    def top_score_reconstruct(self, top_score, idx_diff, mutant):
        mutant = list(mutant)
        for (index, replacement) in zip(idx_diff, top_score):
            mutant[index] = replacement

        mutant = ''.join(mutant)
        return mutant

    def itter_dict(self, dct, idx, mut):
        mut_score = 0
        new_dct = {}

        # print (f'\nTop Substitutes for AA @ position {idx} (Mutation = {mut})')
        new_dct[idx] = {}
        for i in range (len(dct)):
            # print (f'Substitute AA: {dct[i].get("token_str")} with Score: {dct[i].get("score")}')
            new_dct[idx][dct[i].get("token_str")] = dct[i].get("score")
            if mut == dct[i].get("token_str"):
                mut_score = dct[i].get("score")

        max_like_score = dct[0].get("score")
        return  max_like_score, mut_score, new_dct


    def calc_single_substitution_likelihoods(self, list_wt, list_mut, list_id):

        dict_substitutions_list = list(map(self._make_substitution_index_list, list_wt, list_mut, list_id))
        dict_substitutions_dict = {rec["id"][0]:rec for rec in dict_substitutions_list}
        list_ids = []
        list_ids += [dict_substitutions_dict[_id]["id"] for _id in list_id]
        
        list_dict_substitutions = map(self._str_prepare_single_index, dict_substitutions_list)

        list_single_masked_seq_wt = []
        list_single_masked_seq_mut = []
        for lis in list_dict_substitutions:
            list_single_masked_seq_wt += lis["single_sub_masked_wt"]
            list_single_masked_seq_mut += lis["single_sub_masked_mut"]
        
        ans_all_mut = self.unmasker(list_single_masked_seq_mut, batch_size = self.batch_size)
        ans_all_wt = self.unmasker(list_single_masked_seq_wt, batch_size = self.batch_size)
        dict_ans = {}
        print(f"len all: {len(ans_all_mut)}")
        for ans_mut, ans_wt, _id in zip(ans_all_mut, ans_all_wt, list_ids):
            print(f"len mut1: {len(ans_mut)}")
            tmp_mut ={}
            tmp_wt  ={}
            print(_id)
            for rec in _id:
                print(f"id: {rec}, dict[<id>]: {dict_substitutions_dict[rec]}")
            print(f"ans_mut: {ans_mut}")    
            for ans_var_mut, ans_wt, key in zip(ans_mut, ans_wt, [dict_substitutions_dict[rec]["substitutions"].keys() for rec in _id]):
                print(f"ans_var_mut: {ans_var_mut}")
                
                tmp_mut[key] = {rec['token_str']:rec["score"] for rec in ans_var_mut}
                tmp_wt[key] = {rec['token_str']:rec["score"] for rec in ans_var_wt}
            dict_ans[d["id"]]= {"WT":tmp_wt,
                               "MUT":tmp_mut}
        return dict_ans


    def _rand_index_shuffle(self, seq):
        arr = np.arange(len(seq))
        np.random.shuffle(arr)
        return arr

    def _str_prepare_index(self, seq, arr):
        
        list_seq = list(seq)
        for idx in arr:
            list_seq[idx] = self.mask
        seq_m = " ".join(list_seq)
        return seq_m
    
    def data_preprocessing(self, args):
        seq, _id = args
        
        arr = self._rand_index_shuffle(seq)
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
            batch.append(self._str_prepare_index(seq, arr[i*partition_width:(i+1)*partition_width]))
           # print(f"arr: {arr[i*partition_width:(i+1)*partition_width]}\n")
            idxs.append(arr[i*partition_width:(i+1)*partition_width])
        return batch, _id, idxs
    def unmasker_mp(self, batch_seq):
        return self.unmasker(batch_seq, batch_size=3)

    def calc_likelihood_15(self, list_seq, list_id):
        batch_seq = [] 
        batch_id  = []
        batch_idx  = []
        batch_map = []
        start = time()
        for seq, _id in zip(list_seq, list_id):
            arr = self._rand_index_shuffle(seq)
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
                batch.append(self._str_prepare_index(seq, arr[i*partition_width:(i+1)*partition_width]))
               # print(f"arr: {arr[i*partition_width:(i+1)*partition_width]}\n")
                idxs.append(arr[i*partition_width:(i+1)*partition_width])
                batch_map.append((self._str_prepare_index(seq, arr[i*partition_width:(i+1)*partition_width]),arr[i*partition_width:(i+1)*partition_width] , _id))
            #batch_seq += batch
            #batch_id += [_id for _ in range(PARTITION_SIZE+remainder)]
            #batch_idx += idxs
            batch_seq.append(batch)
            batch_id +=  [_id for _ in range(PARTITION_SIZE+remainder)] 
            batch_idx += idxs
        print(f"preprocess time: {time() - start}")
       
        #dataset = MaskedSequenceDataset(seqs = batch_seq, idxs =batch_idx, _ids = batch_id, tokenizer = self.tokenizer, max_length=512)
        start = time()
        with Pool(6) as p:
             ans_all = p.map(self.unmasker_mp, batch_seq)
        ans_all_tmp = []
        for ans_b in ans_all:
            ans_all_tmp += [ans for ans in ans_b]
        ans_all = ans_all_tmp
	#print(list_id[0])
        #print(list_seq[0])
        #print(batch_seq[0])
        #print(dataset)
        #print(dataset.__getitem__(0))
        #dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=16)
        #idx_all = []
        #id_all  = []
        #ans_all = []
        #ans_all = self.unmasker(dataset, batch_size = 512)
        #for batch in dataloader:
        #    batch_seq, batch_idx, batch_id = batch
        #    idx_all += batch_idx
        #    id_all += batch_id
        #    print(batch_seq)

            #tmp = self.unmasker(batch, batch_size=512)
        #print(f"len all: {len(ans_all)}")
        id_all = batch_id
        idx_all = batch_idx

        print(f"gpu time: {time() - start}")
        start = time()
        dict_likelihoods = {}
        for id_, idxs, rec in zip(id_all, idx_all, ans_all):
            tmp_mut = {}
            if id_ not in dict_likelihoods:
                dict_likelihoods[id_] = {}
            if idxs.size != 1:
                for rc, idx in zip(rec, np.sort(idxs)):
                    dict_likelihoods[id_][idx] = {r['token_str']:r["score"] for r in rc}
            else:
                dict_likelihoods[id_][idxs[0]] = {r['token_str']:r["score"] for r in rec}
        print(f"post process time: {time() - start}")

        return dict_likelihoods
    
    def calc_likelihood_15_mt(self, batch_seq, batch_id, batch_idx): #list_seq, list_id):

        #data = Dataset.from_dict({"text": batch_seq})
            
        ans_all = self.unmasker(batch_seq, batch_size = self.batch_size)
        #print(f"len all: {len(ans_all)}")
            
        dict_likelihoods = {}
        for id_, idxs, rec in zip(batch_id, batch_idx, ans_all):
            tmp_mut = {}
            if id_ not in dict_likelihoods:
                dict_likelihoods[id_] = {}
            if idxs.size != 1:
                for rc, idx in zip(rec, np.sort(idxs)):
                    dict_likelihoods[id_][idx] = {r['token_str']:r["score"] for r in rc}
            else:
                dict_likelihoods[id_][idxs[0]] = {r['token_str']:r["score"] for r in rec}


        return dict_likelihoods
        
    def __call__(self, seqA, seqB):
    
        all_targets, all_muts = self.pre_masker(seqA, seqB)

        mutant_lst, idx_diff_lst = self.str_prepare(seqA, all_targets)
        print(f"Mutant list: {mutant_lst}\n Idx list: {idx_diff_lst}")

        
        # for vera A40 512 batch size seems ideal
        ans_all = self.unmasker(mutant_lst, batch_size = self.batch_size)
        
        dct_lst = []
        all_recon = []
        for k in range (len(ans_all)):
            ans = ans_all[k]
            idx_diff = idx_diff_lst[k]
            muts = all_muts[k]
            target = all_targets[k]
            
            top_score = []
            divider = len(ans)
            dict_all = {}
            sum1 = 0
            sum2 = 0
            
            for i in range (len(ans)):
                try:
                    top_score.append(ans[i][0].get('token_str'))
                    score, mut_score, new_dct = self.itter_dict(ans[i], idx_diff[i], muts[i])
                    dict_all.update(new_dct)
                    sum1 += score
                    sum2 += mut_score
                except: # Single Mutation 
                    top_score.append(ans[i].get('token_str'))
                    sum1, sum2, new_dct = self.itter_dict(ans, idx_diff[i], muts[i])
                    dict_all.update(new_dct)
                    divider = 1
                    break
                # print('---------------')
            print (f'Top Substitutions suggested by BERT: {top_score}')
            print (f'Top Substitutions overall likelihood: {sum1/divider}')
            print (f'Overall likelihood of the original substitutions: {sum2/divider}')
            dct_lst.append(dict_all)
            all_recon.append(self.top_score_reconstruct(top_score, idx_diff, target))
        
        return dct_lst, all_recon













