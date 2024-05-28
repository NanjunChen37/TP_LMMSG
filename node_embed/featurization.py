

# todo: 1. DeprecationWarning: an integer is required (got type numpy.float64). 
# Implicit conversion to integers using int is deprecated, and may be removed in 
# a future version of Python. y = torch.tensor([Y], dtype=torch.long)

# todo: 2.Creating a tensor from a list of numpy.ndarrays is extremely slow. 
# Please consider converting the list to a single numpy.ndarray with numpy.array() before 
# converting to a tensor. (Triggered internally at  ../torch/csrc/utils/tensor_new.cpp:201.)

import torch
import re
import os
import numpy as np
from torch_geometric.data import Data
# from msa_generation import df_to_fasta
from pathlib import Path
from tqdm import tqdm
import glob
import pandas as pd


def load_seqs(fasta_file, label=1, task='cls'):
    
    # * function to load fasta file for classification task
    # @param fasta_file: source file name in fasta format
    # @param label: 1 & 0 indicate wether AMP or not
    # @param task: cls & reg indicate if load seqs for classification or regression
    # @output:1. seq_ids: name list; 2. seqs: peptide sequence list; 3. labels: label list
    
    seq_ids = []
    seqs = []
    t = 0
    # Filter out some peptide sequences
    pattern = re.compile('[^ARNDCQEGHILKMFPSTWYV]')
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line[0] == '>':
                t = line.replace('|', '_')
            elif len(pattern.findall(line)) == 0:
                seqs.append(line)
                seq_ids.append(t)
                t = 0
                
    if task == 'cls':
        if label == 1:
            labels = np.ones(len(seq_ids))
        else:
            labels = np.zeros(len(seq_ids))
            
        return seq_ids, seqs, labels
    

def load_cv_seqs(fasta_file):
    seq_ids = []
    seqs = []
    labels = []
    pattern = re.compile('[^ARNDCQEGHILKMFPSTWYV]')
    
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            
            if line[0] == '>':
                id = line.split('|')[0]
                label = line.split('|')[1]
                seq_ids.append(id)
                labels.append(label)
                
            elif len(pattern.findall(line)) == 0:
                seqs.append(line)
                
    labels = np.array([float(x) for x in labels])
    
    return seq_ids, seqs, labels


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
    

def get_contact_map(npz_folder, seq_ids, threshold, add_self_loop=True): 
    
    # * function to load fasta file for classification task
    # @param npz_folder: the folder that store all .npz files
    # @param seq_ids: all seqs id return by function `load_seqs`
    # @param threshold: default as 37. The distance range (2 to 20 Å) is binned into 36 equally spaced segments, plus one bin indicating that residues are not in contact.
    # @param add_self_loop: default as True. Wether add self loop in the peptide graph
    # @output: 1. list_A: the adjacency matrix; 2. list_E: the edge info include dist, omega, theta, phi

    print('getting contact map ...')
    if npz_folder[-1] != '/':
        npz_folder += '/'
        # print(npz_folder)
    
    list_A = []
    list_E = []

    for seq_id in seq_ids:
        npz = seq_id[1:] + '.npz'
        f = np.load(npz_folder + npz)
        # print(f)
        mat_dist = f['dist']
        # print(mat_dist.shape)
        # print(mat_dist)
        mat_omega = f['omega']
        mat_theta = f['theta']
        mat_phi = f['phi']

        dist = np.argmax(mat_dist, axis=2)  # 37 equally spaced segments
        # print(dist)
        omega = np.argmax(mat_omega, axis=2)
        theta = np.argmax(mat_theta, axis=2)
        phi = np.argmax(mat_phi, axis=2)

        A = np.zeros(dist.shape, dtype=np.int64)

        A[dist < threshold] = 1
        A[dist == 0] = 0
        # A[omega < threshold] = 1
        if add_self_loop:
            A[np.eye(A.shape[0]) == 1] = 1
        else:
            A[np.eye(A.shape[0]) == 1] = 0

        # print(dist[A == 0])
        dist[A == 0] = 0
        omega[A == 0] = 0
        theta[A == 0] = 0
        phi[A == 0] = 0
        
        # print(dist)

        dist = np.expand_dims(dist, -1)
        omega = np.expand_dims(omega, -1)
        theta = np.expand_dims(theta, -1)
        phi = np.expand_dims(phi, -1)
        
        edges = np.concatenate((dist, omega, theta, phi), axis=-1)

        # print(edges.shape)
        
        list_A.append(A)
        list_E.append(edges)

    return list_A, list_E


def onehot_encoding(seqs):
    
    # * function to conduct onehot encoding on fasta sequences
    # @param seqs: seqs array generated by `load_seqs`
    # @output: onehot encoding list of the fasta sequences
    
    print('running onehot encoding ...')
    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    encoding_map = np.eye(len(residues))
    residues_map = {residue: encoding_map[ind] for ind, residue in enumerate(residues)}

    seqs_onehot = []
    for seq in seqs:
        seq_onehot = [residues_map[r] for r in seq]
        seqs_onehot.append(np.array(seq_onehot))

    return seqs_onehot


def position_encoding(seqs):

    # * Position encoding features introduced in "Attention is all your need", the b is changed to 1000 for the short length of peptides.
    # @param seqs: seqs array generated by `load_seqs`
    # @output: position encoding list of the fasta sequences

    print('running position encoding ...')
    d = 20
    b = 1000
    seqs_pos_encoding = []
    
    for seq in seqs:
        N = len(seq)
        value = []
        
        for pos in range(N):
            seq_pos_encoding = []
            
            for i in range(d // 2):
                seq_pos_encoding.append(pos / (b ** (2 * i / d)))
            value.append(seq_pos_encoding)
            
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        seqs_pos_encoding.append(pos_encoding)
        
    return seqs_pos_encoding


def load_hhm(seq_id, hhm_folder):

    # * function to extract the useful feature in .hhm file
    # @param seq_id: sequence id of each sequence that index .hhm file
    # @param hhm_folder: folder contains all .hhm files
    # @output: position encoding list of the fasta sequences
    
    if hhm_folder[-1] != '/': hhm_folder += '/'
    with open(hhm_folder + seq_id + '.hhm', 'r') as f:
        lines = f.readlines()
        seqs_hhm_array = []
        row_tag = 0
        
        for line in lines:
            line = line.strip()
            
            if line == '#':
                row_tag = 1
                continue
            if row_tag != 0 and row_tag < 5:
                row_tag += 1
                continue
            if row_tag >= 5:
                line = line.replace('*', '0')
                line_split = line.split('\t')
                
                if len(line_split) >= 20:
                    first_num = [int(line_split[0].split(' ')[-1])]  
                    nums = list(map(int, line_split[1:20]))
                    first_num.extend(nums)
                    normed = [i if i == 0 else 2 ** (-0.001 * i) for i in first_num]
                    seqs_hhm_array.append(normed)
                    
    return seqs_hhm_array


def hhm_encoding(seq_ids, hhm_folder):

    # * function to conduct hhm encoding based on .hhm files
    # @param seq_ids: seqs array generated by `load_seqs`
    # @param hhm_folder: folder contains all .hhm files
    # @output: hhm encoding list of the fasta sequences
    
    print('running hhm encoding ...')
    
    if hhm_folder[-1] != '/': hhm_folder += '/'
    hhm_file_paths = os.listdir(hhm_folder + 'output/') # !
    seqs_hhm_encoding = []
    
    for seq_id in seq_ids:
        
        if seq_id[0] == '>':
            assert seq_id[1:] + '.hhm' in hhm_file_paths
            seq_hhm_encoding = load_hhm(seq_id[1:], hhm_folder + 'output/')
            seqs_hhm_encoding.append(np.array(seq_hhm_encoding))

    return seqs_hhm_encoding


def load_pssm(query, pssm_path):
    """
    :param query: query id
    :param pssm_path: dir saving pssm files
    """
    if pssm_path[-1] != '/': pssm_path += '/'
    with open(pssm_path + query + '.pssm', 'r') as f:
        lines = f.readlines()
        res = []
        for line in lines[3:]:
            line = line.strip()
            lst = line.split(' ')
            while '' in lst:
                lst.remove('')
            if len(lst) == 0:
                break
            r = lst[2:22]
            r = [int(x) for x in r]
            res.append(r)
    return res


def pssm_encoding(ids, pssm_dir):
    """
    parser pssm features
    """
    if pssm_dir[-1] != '/': pssm_dir += '/'
    pssm_fs = os.listdir(pssm_dir + 'output/')

    res = []
    for id in ids:
        name = id
        if id[0] == '>': name = id[1:]
        if name + '.pssm' in pssm_fs:
            # psiblast
            tmp = load_pssm(name, pssm_dir + 'output/')
            res.append(np.array(tmp))
        else:
            # blosum
            tmp = load_pssm(name, pssm_dir + 'blosum/')
            res.append(np.array(tmp))
    return res


def seq2vec_encoding(seq_ids, s2v_folder):
    
    # * 
    # @param seq_ids: seqs array generated by `load_seqs`
    # @param s2v_folder: 
    # @output: 
    
    print('running seq2vec encoding ...')
    seq_ids = [element.replace('>', '').replace('|', '_') for element in seq_ids]
    # print(seq_ids)
    
    if s2v_folder[-1] != '/': s2v_folder += '/'
    
    s2v_list = []
    for id in seq_ids:
        fea_path = s2v_folder + id + '.npy'
        data = np.load(fea_path)
        s2v_list.append(data)
        
    return s2v_list

        
def lm_encoding(seq_ids, lm_folder):
        
    # * 
    # @param seq_ids: seqs array generated by `load_seqs`
    # @param s2v_folder: 
    # @output: 
    note = lm_folder.split('/')[-2]
    print(f'running language model encoding from {note} ...')
    seq_ids = [element.replace('>', '').replace('|', '_') for element in seq_ids]
    # print(seq_ids)
    
    if lm_folder[-1] != '/': lm_folder += '/'
    
    lm_list = []
    for id in seq_ids:
        fea_path = lm_folder + id + '.npy'
        data = np.load(fea_path)
        lm_list.append(data)
        
    return lm_list

    

def build_parse_matrix(A, E, X, Y, S=None, L=None, eps=1e-6):

    # * function to build parse matrix from adjacency and edge matrix
    # @param A: adjacency matrix (n_nodes, n_nodes)
    # @param E: edge matrix (n_nodes, n_nodes, n_edge_features)
    # @param X: node embeddings (n_nodes, n_node_features)
    # @param Y: labels (classification labels)
    # @output: Data object with node feature, edge index, edge feature, labels
    
    num_row, num_col = A.shape
    rows = []
    cols = []
    e_vec = []

    for i in range(num_row):
        for j in range(num_col):
            if A[i][j] >= eps:
                rows.append(i)
                cols.append(j)
                e_vec.append(E[i][j])
                
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x = torch.tensor(X, dtype=torch.float32) # todo 2
    
    edge_attr = torch.tensor(np.array(e_vec), dtype=torch.float32)
    y = torch.tensor(np.array([Y]), dtype=torch.long)
    # edge_attr = torch.tensor(e_vec, dtype=torch.float32) # todo 1
    # y = torch.tensor([Y], dtype=torch.long)

    if S is None:
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    elif S is not None and L is None:
        s = torch.tensor(S, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, s=s)
    
    else:
        s = torch.tensor(S, dtype=torch.float32)
        l = torch.tensor(L, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, s=s, l=l)
    
    
def build_parse_matrix_multi_lm(A, E, X, Y, S=None, L=None, eps=1e-6):

    # * function to build parse matrix from adjacency and edge matrix
    # @param A: adjacency matrix (n_nodes, n_nodes)
    # @param E: edge matrix (n_nodes, n_nodes, n_edge_features)
    # @param X: node embeddings (n_nodes, n_node_features)
    # @param Y: labels (classification labels)
    # @output: Data object with node feature, edge index, edge feature, labels
    
    num_row, num_col = A.shape
    rows = []
    cols = []
    e_vec = []

    for i in range(num_row):
        for j in range(num_col):
            if A[i][j] >= eps:
                rows.append(i)
                cols.append(j)
                e_vec.append(E[i][j])
                
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x = torch.tensor(X, dtype=torch.float32) # todo 2
    
    edge_attr = torch.tensor(np.array(e_vec), dtype=torch.float32)
    y = torch.tensor(np.array([Y]), dtype=torch.long)
    # edge_attr = torch.tensor(e_vec, dtype=torch.float32) # todo 1
    # y = torch.tensor([Y], dtype=torch.long)

    if S is None:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    else:
        s = torch.tensor(S, dtype=torch.float32)
        lm_dic = {}
        for i in range(len(L)):
            lm_name = f'l{i}'
            lm_value = torch.tensor(L[i], dtype=torch.float32)
            lm_dic[lm_name] = lm_value
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, s=s, **lm_dic)


def feature_concat(*args):
    
    print('concat residue features ...')
    feature_list = args[0]
    for list in args[1:]:
        for i in range(len(list)):
            feature_list[i] = np.hstack((feature_list[i], list[i]))

    return feature_list

def feature_concat_nest(*args):
    
    print('concat residue features ...')
    feature_list = args[0]
    for list in args[1:-2]:
        for i in range(len(list)):
            feature_list[i] = np.hstack((feature_list[i], list[i]))
            
    for i in range(len(list)): 
        feature_list[i] = np.hstack((feature_list[i], args[-2][i]))
        
    for i in range(len(list)): 
        for lm in range(len(args[-1])):
            feature_list[i] = np.hstack((feature_list[i], args[-1][lm][i]))

    return feature_list


def load_data_adv_hhm(fasta_file, label, task, npz_folder, threshold, seperate, use_all_nr, use_lm, cross_valid):
    
    if not cross_valid:
        if use_all_nr:
            pssm_folder = '/'.join(fasta_file.split('/')[:-1]) + '/pssm/'
        else:
            pssm_folder = '/'.join(fasta_file.split('/')[:-1]) + '/blos/'
            
        hhm_folder = '/'.join(fasta_file.split('/')[:-1]) + '/hhm/'
        s2v_folder = '/'.join(fasta_file.split('/')[:-1]) + '/s2v/'
        lm_folder = '/'.join(fasta_file.split('/')[:-1]) + '/lm/*/'
        seq_ids, seqs, labels = load_seqs(fasta_file, label, task)
    
    else:
        if use_all_nr:
            pssm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/pssm/'
        else:
            pssm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/blos/'
            
        hhm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/hhm/'
        s2v_folder = '/'.join(npz_folder.split('/')[:-1]) + '/s2v/'
        lm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/lm/*/'
        seq_ids, seqs, labels = load_cv_seqs(fasta_file)
    
    A_list, E_list = get_contact_map(npz_folder, seq_ids, threshold, add_self_loop=True)
    # np.set_printoptions(threshold=np.inf)
    # print(A_list[0])
    # print(E_list[0])
    
    # b = np.pad(A_list[0], ((0, 100 - len(A_list[0])), (0, 100 - len(A_list[0]))), mode='constant')
    # print(b)
    
    onehot = onehot_encoding(seqs) # onehot[0].shape => (92, 20)
    
    pos = position_encoding(seqs)
    
    hhm = hhm_encoding(seq_ids, hhm_folder)
    
    pssm = pssm_encoding(seq_ids, pssm_folder)
    
    s2v = seq2vec_encoding(seq_ids, s2v_folder) # s2v[0].shape => (92, 1024)
        
    lm_dirs = glob.glob(lm_folder)
    lm_num = len(use_lm)  
    
    if lm_num == 1:
        if 'uni' in use_lm:    
            lm_dir = [dir for dir in lm_dirs if 'uni' in dir][0]
            lm = lm_encoding(seq_ids, lm_dir)
        elif 'bfd' in use_lm:
            lm_dir = [dir for dir in lm_dirs if 'bfd' in dir][0]
            lm = lm_encoding(seq_ids, lm_dir)
        elif 'brt' in use_lm:
            lm_dir = [dir for dir in lm_dirs if 'brt' in dir][0]
            lm = lm_encoding(seq_ids, lm_dir)
    
    else:
        lm_list = []
        for lm_dir in lm_dirs: 
            if any(use_lm_item in lm_dir for use_lm_item in use_lm):
                lm = lm_encoding(seq_ids, lm_dir)
                lm_list.append(lm)
    
    # lm = lm_encoding(seq_ids, lm_folder)
    data_list = []
    n_samples = len(A_list)
    
    if not seperate:
        if lm_num == 0:
            feature_concat_list = feature_concat(onehot, pos, hhm, pssm)
            # feature_concat_list = feature_concat(onehot, pos, hhm, pssm, s2v)
        elif lm_num == 1:
            feature_concat_list = feature_concat(onehot, pos, hhm, pssm, s2v, lm)
        else:
            # feature_concat_list = feature_concat(onehot, pos, hhm, pssm, s2v, lm_list[0], lm_list[1])
            feature_concat_list = feature_concat_nest(onehot, pos, hhm, pssm, s2v, lm_list)
        for i in range(n_samples):
            data_list.append(build_parse_matrix(A_list[i], E_list[i], feature_concat_list[i], labels[i]))
        
    else:
        feature_concat_list = feature_concat(onehot, pos, hhm, pssm)
        s2v_list = s2v
        if lm_num == 0:
            for i in range(n_samples):
                data_list.append(build_parse_matrix(A_list[i], E_list[i], feature_concat_list[i], labels[i], S=s2v_list[i]))
        elif lm_num == 1:
            lm_list = lm
            for i in range(n_samples):
                data_list.append(build_parse_matrix(A_list[i], E_list[i], feature_concat_list[i], labels[i], S=s2v_list[i], L=lm_list[i]))
        else:
            for i in range(n_samples):
                lm_list_t = [list(row) for row in zip(*lm_list)]
                data_list.append(build_parse_matrix_multi_lm(A_list[i], E_list[i], feature_concat_list[i], labels[i], S=s2v_list[i], L=lm_list_t[i]))
                
        
    # print(len(data_list), data_list[0])
        
    return data_list, labels


def load_data_adv_nohhm(fasta_file, label, task, npz_folder, threshold, seperate, use_all_nr, use_lm, cross_valid):
    
    if not cross_valid:
        if use_all_nr:
            pssm_folder = '/'.join(fasta_file.split('/')[:-1]) + '/pssm/'
        else:
            pssm_folder = '/'.join(fasta_file.split('/')[:-1]) + '/blos/'
            
        # hhm_folder = '/'.join(fasta_file.split('/')[:-1]) + '/hhm/'
        s2v_folder = '/'.join(fasta_file.split('/')[:-1]) + '/s2v/'
        lm_folder = '/'.join(fasta_file.split('/')[:-1]) + '/lm/*/'
        seq_ids, seqs, labels = load_seqs(fasta_file, label, task)
    
    else:
        if use_all_nr:
            pssm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/pssm/'
        else:
            pssm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/blos/'
            
        # hhm_folder = '/'.join(npz_folder.split('/')[:-2]) + '/hhm/'
        s2v_folder = '/'.join(npz_folder.split('/')[:-1]) + '/s2v/'
        lm_folder = '/'.join(npz_folder.split('/')[:-1]) + '/lm/*/'
        seq_ids, seqs, labels = load_cv_seqs(fasta_file)
    
    A_list, E_list = get_contact_map(npz_folder, seq_ids, threshold, add_self_loop=True)
    
    onehot = onehot_encoding(seqs) # onehot[0].shape => (92, 20)
    
    pos = position_encoding(seqs)
    
    # hhm = hhm_encoding(seq_ids, hhm_folder)
    
    pssm = pssm_encoding(seq_ids, pssm_folder)
    
    s2v = seq2vec_encoding(seq_ids, s2v_folder) # s2v[0].shape => (92, 1024)
        
    lm_dirs = glob.glob(lm_folder)
    lm_num = len(use_lm)  
    
    if lm_num == 1:
        if 'uni' in use_lm:    
            lm_dir = [dir for dir in lm_dirs if 'uni' in dir][0]
            lm = lm_encoding(seq_ids, lm_dir)
        elif 'bfd' in use_lm:
            lm_dir = [dir for dir in lm_dirs if 'bfd' in dir][0]
            lm = lm_encoding(seq_ids, lm_dir)
        elif 'brt' in use_lm:
            lm_dir = [dir for dir in lm_dirs if 'brt' in dir][0]
            lm = lm_encoding(seq_ids, lm_dir)
    
    else:
        lm_list = []
        for lm_dir in lm_dirs: 
            if any(use_lm_item in lm_dir for use_lm_item in use_lm):
                lm = lm_encoding(seq_ids, lm_dir)
                lm_list.append(lm)
    
    # lm = lm_encoding(seq_ids, lm_folder)
    data_list = []
    n_samples = len(A_list)
    
    if not seperate:
        if lm_num == 0:
            feature_concat_list = feature_concat(onehot, pos, pssm)
            # feature_concat_list = feature_concat(onehot, pos, hhm, pssm, s2v)
        elif lm_num == 1:
            feature_concat_list = feature_concat(onehot, pos, pssm, s2v, lm)
        else:
            # feature_concat_list = feature_concat(onehot, pos, hhm, pssm, s2v, lm_list[0], lm_list[1])
            feature_concat_list = feature_concat_nest(onehot, pos, pssm, s2v, lm_list)
        for i in range(n_samples):
            data_list.append(build_parse_matrix(A_list[i], E_list[i], feature_concat_list[i], labels[i]))
        
    else:
        feature_concat_list = feature_concat(onehot, pos, pssm)
        s2v_list = s2v
        if lm_num == 0:
            for i in range(n_samples):
                data_list.append(build_parse_matrix(A_list[i], E_list[i], feature_concat_list[i], labels[i], S=s2v_list[i]))
        elif lm_num == 1:
            lm_list = lm
            for i in range(n_samples):
                data_list.append(build_parse_matrix(A_list[i], E_list[i], feature_concat_list[i], labels[i], S=s2v_list[i], L=lm_list[i]))
        else:
            for i in range(n_samples):
                lm_list_t = [list(row) for row in zip(*lm_list)]
                data_list.append(build_parse_matrix_multi_lm(A_list[i], E_list[i], feature_concat_list[i], labels[i], S=s2v_list[i], L=lm_list_t[i]))
                
        
    # print(len(data_list), data_list[0])
        
    return data_list, labels
    
def load_data_adv(fasta_file, label, task, npz_folder, threshold, seperate, use_hhm, use_all_nr, use_lm, cross_valid):
    
    if use_hhm == True:
        data_list, labels = load_data_adv_hhm(fasta_file, label, task, npz_folder, threshold, seperate, use_all_nr, use_lm, cross_valid)
        
    else:
        data_list, labels = load_data_adv_nohhm(fasta_file, label, task, npz_folder, threshold, seperate, use_all_nr, use_lm, cross_valid)

    return data_list, labels

if __name__ == "__main__":
    
    fasta_file = '/media/cnj/T9/data/test_data/positive/demo.fasta'
    
    label = 1
    threshold = 37
    task = 'cls'
    npz_folder = '/media/cnj/T9/data/test_data/positive/npz'
    use_lm = ['uni', 'bfd', 'brt']
    
    
    d, l = load_data_adv(fasta_file, label, task, npz_folder, threshold, seperate=True, use_all_nr=False, use_lm = use_lm, cross_valid=False)
    
    print(d, l)

    from torch_geometric.loader import DataLoader
    import torch.nn.functional as F
    
    # def tensor2d_norm(tensor_2d):
    
    #     mean = tensor_2d.mean(dim=0)
    #     std = tensor_2d.std(dim=0)
    #     std_tensor = (tensor_2d - mean) / std
        
    #     return std_tensor
    
    # def get_cnn_minibatch(data):
    #     s2v_list = []
    #     lm_list = []
    #     for _ in range(len(data)):
            
    #         s2v = data[_].s
    #         lm = data[_].l
    #         t_s2v = s2v.permute(1, 0)
    #         t_lm = lm.permute(1, 0)
            
    #         norm_s2v = tensor2d_norm(t_s2v) 
    #         norm_lm = tensor2d_norm(t_lm) 
    #         assert(norm_s2v.shape[0] == 1024 and norm_lm.shape[0] == 1024) 
    #         assert(norm_s2v.shape[1] == norm_lm.shape[1]) 
            
    #         pad_s2v = F.pad(norm_s2v, (0, 100 - norm_s2v.shape[1]))
    #         pad_lm = F.pad(norm_lm, (0, 100 - norm_lm.shape[1]))
            
    #         s2v_list.append(pad_s2v)
    #         lm_list.append(pad_lm)
        
    #     s2v_batch = torch.stack(s2v_list, dim=0)
    #     lm_batch = torch.stack(lm_list, dim=0)
        
    #     return s2v_batch, lm_batch
    
 

    
    print('++')
    

    
    
    
    

    