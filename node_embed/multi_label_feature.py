import gc
import os
import argparse
import pandas as pd
import glob
from .featurization import get_contact_map,onehot_encoding,position_encoding,\
    hhm_encoding,pssm_encoding,seq2vec_encoding,lm_encoding,feature_concat,build_parse_matrix,build_parse_matrix_multi_lm


def load_mll_seqs(csv_file):
    # * load multi-lable learning material from csv file.
    df = pd.read_csv(csv_file)
    ids = []
    seqs = []
    labels = []
    for ind, row in df.iterrows():
        id = row['id']
        if id[0] != '>':
            id = '>' + id
        seq = row['sequence']
        label = df.iloc[ind, 2:].values.tolist()
        ids.append(id)
        seqs.append(seq)
        labels.append(label)
    
    return ids, seqs, labels


def save_mll_fasta(ids, seqs, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(['{}\n{}'.format(id, seq) for id, seq in zip(ids, seqs)]) + '\n')
        f.flush()
        
        
def load_data_mll(csv_file, label, task, npz_folder, threshold, seperate, use_all_nr, use_lm = 'uni', cross_valid=False):
    

    if use_all_nr:
        pssm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/pssm/'
    else:
        pssm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/blos/'
        
    hhm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/hhm/'
    s2v_folder = '/'.join(npz_folder.split('/')[:-1]) + '/s2v/'
    lm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/lm/*/'
    
    seq_ids, seqs, labels = load_mll_seqs(csv_file)
    
    A_list, E_list = get_contact_map(npz_folder, seq_ids, threshold, add_self_loop=True)
    
    onehot = onehot_encoding(seqs) # onehot[0].shape => (92, 20)
    
    pos = position_encoding(seqs)
    
    hhm = hhm_encoding(seq_ids, hhm_folder)
    
    pssm = pssm_encoding(seq_ids, pssm_folder)
    
    s2v = seq2vec_encoding(seq_ids, s2v_folder) # s2v[0].shape => (92, 1024)
        
    lm_dirs = glob.glob(lm_folder)
    
    if use_lm == 'uni':    
        lm_dir = [dir for dir in lm_dirs if 'uni' in dir][0]
        lm = lm_encoding(seq_ids, lm_dir)
        
    elif use_lm == 'bfd':
        lm_dir = [dir for dir in lm_dirs if 'bfd' in dir][0]
        lm = lm_encoding(seq_ids, lm_dir)
        
    elif use_lm == 'both':
        lm_list = []
        for lm_dir in lm_dirs: # todo
            lm = lm_encoding(seq_ids, lm_dir)
            lm_list.append(lm)
    
    # lm = lm_encoding(seq_ids, lm_folder)
    data_list = []
    n_samples = len(A_list)
    
    data_list = []
    n_samples = len(A_list)
    
    if not seperate:
        if use_lm != 'both':
            feature_concat_list = feature_concat(onehot, pos, hhm, pssm, s2v, lm)
        else:
            feature_concat_list = feature_concat(onehot, pos, hhm, pssm, s2v, lm_list[0], lm_list[1])
        for i in range(n_samples):
            data_list.append(build_parse_matrix(A_list[i], E_list[i], feature_concat_list[i], labels[i]))
            
            
    else:
        feature_concat_list = feature_concat(onehot, pos, hhm, pssm)
        s2v_list = s2v
        if use_lm != 'both':
            lm_list = lm
            for i in range(n_samples):
                data_list.append(build_parse_matrix(A_list[i], E_list[i], feature_concat_list[i], labels[i], S=s2v_list[i], L=lm_list[i]))
        else:
            for i in range(n_samples):
                lm_list_t = [list(row) for row in zip(*lm_list)]
                data_list.append(build_parse_matrix_multi_lm(A_list[i], E_list[i], feature_concat_list[i], labels[i], S=s2v_list[i], L=lm_list_t[i]))
                
        
    # print(len(data_list), data_list[0])
        
    return data_list, labels
    

import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss, confusion_matrix, multilabel_confusion_matrix, roc_auc_score, roc_curve, auc
def get_multi_label_metrics(trueY, scoreY, labels_list, threshold=0.5): #scoreY是2維的
    predY = (scoreY > threshold) * 1
    cm = multilabel_confusion_matrix(trueY, predY)
    df = pd.DataFrame(columns=['label', 'tn', 'fp', 'fn', 'tp'])
    for i in range(len(labels_list)):
        df.loc[len(df)] = [labels_list[i]] + list(cm[i].ravel())

    df['acc'] = ((df['tn']+df['tp'])/(df['tn']+df['fp']+df['fn']+df['tp']))
    df['pre'] = (df['tp']/(df['tp']+df['fp']).replace(0, np.nan))
    df['recall'] = (df['tp']/(df['tp']+df['fn']).replace(0, np.nan))
    df['auc'] = 0
    for i in range(len(labels_list)):
        df.loc[i, 'auc'] = roc_auc_score(trueY[:, i], scoreY[:, i])
    df['f1'] = ((2*df['pre']*df['recall'])/(df['pre']+df['recall']).replace(0, np.nan))

        
    mcc_upper = (df['tp']*df['tn']-df['fp']*df['fn'])
    mcc_lower = (((df['tp']+df['fp'])*(df['tp']+df['fn'])*(df['tn']+df['fp'])*(df['tn']+df['fn']))**(1/2)).replace(0, np.nan)
    df['mcc'] = (mcc_upper/mcc_lower).astype(float).round(4)
    
    df.loc[len(df)] = ['macro', np.nan, np.nan, np.nan, np.nan, df['acc'].mean(), df['pre'].mean(), df['recall'].mean(), df['auc'].mean(), df['f1'].mean(), df['mcc'].mean()]
    df[['acc', 'pre', 'recall', 'auc', 'f1', 'mcc']] = df[['acc', 'pre', 'recall', 'auc', 'f1', 'mcc']].astype(float).round(4)
    df[['tn', 'fp', 'fn', 'tp']] = df[['tn', 'fp', 'fn', 'tp']].round()
    
    return df

def get_macro_curves(trueY, scoreY, n_labels):
    for i in range(n_labels):
        fpr, tpr, _ = roc_curve(trueY[:, i], scoreY[:, i])



if __name__ == '__main__':
    
    # * get fasta from csv
    # file = 'data/mul_label_data/multilabel.csv'
    # ids, seqs, labels = load_mll_seqs(file)
    
    # fasta = 'data/mul_label_data/multilabel.fasta'
    # save_mll_fasta(ids, seqs, fasta)
    
    load_data_mll(csv_file = 'data/mul_label_data/multilabel_demo.csv',
                  label = 1,
                  task = 'mll',
                  npz_folder = 'data/mll_bac/npz',
                  threshold = 37,
                  seperate = True,
                  use_all_nr = False,
                  use_lm = 'both',
                  cross_valid=False)
    
    
    
    
print('..')