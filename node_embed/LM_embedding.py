import torch
import re
import os
import numpy as np
import gc
import argparse
from transformers import BertModel, BertTokenizer, T5Tokenizer, T5EncoderModel
from tqdm import tqdm
from bio_embeddings.embed import SeqVecEmbedder
from featurization import load_seqs, load_mll_seqs
import time


def save_single_feature(mod, id_list, fea_list, tg_dir, sub_dir):
    # pass

    if mod == 'LM':
        tg_dir = os.path.join(tg_dir, sub_dir)
        if not os.path.exists(tg_dir):
            os.makedirs(tg_dir, exist_ok=True)
        
        id_list_mod = [element.replace('>', '').replace('|', '_') for element in id_list]
        
        for id in tqdm(id_list_mod):
            element = fea_list[id_list_mod.index(id)]
            np.save(tg_dir + id + '.npy', element)
            
    elif mod == 'S2V':
        tg_dir = os.path.join(tg_dir, 's2v/')
        if not os.path.exists(tg_dir):
            os.makedirs(tg_dir, exist_ok=True)
        id_list_mod = [element.replace('>', '').replace('|', '_') for element in id_list]
        
        for id in tqdm(id_list_mod):
            element = fea_list[id_list_mod.index(id)]
            np.save(tg_dir + id + '.npy', element)


class LM_Embed:

    def __init__(self, language_model, cache_dir, max_len=101, rare_aa=True, task='cls'):
        self.lang_model = language_model
        self.cache_dir = cache_dir
        self.max_len = max_len
        self.rare_aa = rare_aa
        self.task = task

        if self.lang_model in ['Rostlab/prot_t5_xl_uniref50', 'Rostlab/prot_t5_xl_bfd']:
            self.tokenizer = T5Tokenizer.from_pretrained(language_model, cache_dir=cache_dir, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(language_model, cache_dir=cache_dir)
            
        elif self.lang_model in ['Rostlab/prot_bert', 'Rostlab/prot_bert_bfd']:
            self.tokenizer = BertTokenizer.from_pretrained(language_model, cache_dir=cache_dir, do_lower_case=False)
            self.model = BertModel.from_pretrained(language_model, cache_dir=cache_dir)
            
        gc.collect()

    
    def extract_word_embs(self, fasta_file, label, batch_size, out_dir, sub_dir, get_padding):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(device)
        self.model = self.model.eval()
        
        if self.task == 'cls':
            # Making a list of sequences from the fasta:
            seqs_name, seqs_list, labels = load_seqs(fasta_file, label)
        elif self.task == 'mll':
            seqs_name, seqs_list, labels = load_mll_seqs(fasta_file)
        
        # Adding spaces in between sequence letters & replace rare aa:
        seqs_spaced = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seqs_list]
            
        # tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer(seqs_spaced, add_special_tokens=True, padding="max_length", max_length = self.max_len)
        
        # Retrieving the input IDs and mask for attention as tensors:
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        
        torch.cuda.empty_cache()
            
        # Loop to process the sequences into embeddings in batch:
        for i in range(0, len(input_ids), batch_size):
            batch_end = min(i + batch_size, len(input_ids))
            
            if i % batch_size == 0:
                print("Embedding Batch Ending with...", batch_end)
                
            with torch.no_grad():
                embeddings = self.model(input_ids=input_ids[i:batch_end], attention_mask=attention_mask[i:batch_end])[0]
                emb_array = embeddings.cpu().numpy()

            # Creating initial array or concatenating to existing array:
            if i == 0:
                embedding_res = emb_array
            else:
                embedding_res = np.concatenate((embedding_res, emb_array))
        
        # Extracting features using the function below:
        if not get_padding:
            features = self.extract_features(embedding_res, attention_mask) 
        else:
            features = embedding_res
        
        print("Saving LM Embeddings...")
        out_dir_lm = os.path.join(out_dir, 'lm')
        os.makedirs(out_dir_lm, exist_ok=True)
        save_single_feature(mod='LM', id_list=seqs_name, fea_list=features, tg_dir=out_dir_lm, sub_dir=sub_dir)
            
    
    def extract_features(self, emb_res, att_msk):
        features = [] 
        for seq_num in range(len(emb_res)):
            seq_len = (att_msk[seq_num] == 1).sum()

            if self.lang_model in ['Rostlab/prot_bert', 'Rostlab/prot_bert_bfd']:
                seq_emd = emb_res[seq_num][1:seq_len-1]

            elif self.lang_model in ['Rostlab/prot_t5_xl_uniref50', 'Rostlab/prot_t5_xl_bfd']:
                seq_emd = emb_res[seq_num][:seq_len-1]

            features.append(seq_emd)
        
        features_arr = np.array(features, dtype=object)

        return features_arr
            


class S2V_Embed:
    def __init__(self, fasta_file, label, s2v_model_path, task):
        # * function to encode each residue in a single sequence into 1024 features
        # @param seq_ids: seqs array generated by `load_seqs`
        # @param model_path: folder contains the pretraining model weight of `SeqVecEmbedder`: 1. 'weights.hdf5'; 2. 'options.json'
        # @output: seqs to vec encoding list of the fasta sequences
        self.fasta = fasta_file
        self.label = label
        self.model_path = s2v_model_path
        self.task = task
            
    def extract_seq_embs(self, out_dir):
        if self.task == 'cls':
            # Making a list of sequences from the fasta:
            ids, seqs_list, labels = load_seqs(self.fasta, self.label)
        elif self.task == 'mll':
            ids, seqs_list, labels = load_mll_seqs(self.fasta)
    
        weights = os.path.join(self.model_path, 'weights.hdf5')
        options = os.path.join(self.model_path, 'options.json')
        
        # Emptying cache to ensure enough memory:
        torch.cuda.empty_cache()
        
        seqs_vec_encoding = []
        seq_embedder = SeqVecEmbedder(weights_file=str(weights), options_file=str(options), cuda_device=0)
        
        print("Getting S2V Embeddings...")
        for seq in tqdm(seqs_list):
            seq2vec_embedding = seq_embedder.embed(seq)
            residue_embedding = np.sum(seq2vec_embedding, axis=0) 
            seqs_vec_encoding.append(residue_embedding)
            # print(residue_embedding.shape)
            
            
        
        print("Saving S2V Embeddings...")
        save_single_feature(mod='S2V', id_list=ids, fea_list=seqs_vec_encoding, tg_dir=out_dir, sub_dir='')

    
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-fasta_file', type=str,
                    default='data/XU_train/positive/XU_pretrain_val_positive_example.fasta') # * csv when doing mll
    
    parser.add_argument('-lm_model', type=str, 
                        default="Rostlab/prot_t5_xl_uniref50",
                        help='Rostlab/prot_t5_xl_bfd, Rostlab/prot_t5_xl_uniref50, Rostlab/prot_bert')
    parser.add_argument('-cache_dir', type=str, 
                        default="model_files")
    parser.add_argument('-s2v_model_path', type=str, 
                        default='node_embed/seq2vec_pretrain_model')
    parser.add_argument('-task', type=str, 
                        default='cls')
    
    # parser.add_argument('-out_dir', type=str, help='root dir of output embeddings',
    #                     default='data/time_test/XUAMP/')
    
    parser.add_argument('-sub_dir', type=str, help='sub dir of output embeddings',
                        default='uni/')
    parser.add_argument('-max_length', type=int, help='max sequence length + 1',
                        default=101) # !
    parser.add_argument('-b', type=int, help='batch size',
                        default=500) # !
    parser.add_argument('-get_padding', type=bool, default=False)
    
    args = parser.parse_args()
    
    lm_model = args.lm_model
    cache_dir = args.cache_dir
    
    fasta_file = args.fasta_file
    s2v_model_path = args.s2v_model_path
    task = args.task
    
    out_dir = os.path.dirname(fasta_file) + "/"
    sub_dir = args.sub_dir
    
    max_len = args.max_length
    b = args.b
    get_padding = args.get_padding
    
    start_time = time.time()
    
    S2V_EMBED = S2V_Embed(fasta_file, 1, s2v_model_path, task)
    S2V_EMBED.extract_seq_embs(out_dir)
    
    LM_EMBED = LM_Embed(lm_model, cache_dir, max_len, rare_aa=True, task=task)
    LM_EMBED.extract_word_embs(fasta_file, label=1, batch_size=b, out_dir=out_dir, sub_dir=sub_dir, get_padding=get_padding)
    
    lm_model = 'Rostlab/prot_t5_xl_bfd'
    sub_dir = 'bfd/'
    
    LM_EMBED = LM_Embed(lm_model, cache_dir, max_len, rare_aa=True, task=task)
    LM_EMBED.extract_word_embs(fasta_file, label=1, batch_size=b, out_dir=out_dir, sub_dir=sub_dir, get_padding=get_padding)

    lm_model = 'Rostlab/prot_bert'
    sub_dir = 'brt/'
    
    LM_EMBED = LM_Embed(lm_model, cache_dir, max_len+1, rare_aa=True, task=task)
    LM_EMBED.extract_word_embs(fasta_file, label=1, batch_size=b, out_dir=out_dir, sub_dir=sub_dir, get_padding=get_padding)
    







