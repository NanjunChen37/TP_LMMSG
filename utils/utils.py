import random
import numpy as np
import os
import torch
import torch.nn.functional as F
import csv
import datetime
import pandas as pd


def current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


def cv_shuffle(pos, neg, shuffle):
    pass


def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    # print("[Info] cudnn.deterministic set to True. CUDNN-optimized code may be slow.")
    

def tensor2d_norm(tensor_2d):

    mean = tensor_2d.mean(dim=0)
    std = tensor_2d.std(dim=0)
    std_tensor = (tensor_2d - mean) / std
    
    return std_tensor


# def get_cnn_minibatch(data, use_lm, max_len = 101):
#     if use_lm != 'both':
#         s2v_list = []
#         lm_list = []
#         n_num_list = []
#         for _ in range(len(data)):
#             node_num = data[_].num_nodes
            
#             s2v = data[_].s
#             lm = data[_].l
#             t_s2v = s2v.permute(1, 0)
#             t_lm = lm.permute(1, 0)
            
#             norm_s2v = tensor2d_norm(t_s2v) 
#             norm_lm = tensor2d_norm(t_lm) 
#             assert(norm_s2v.shape[0] == 1024 and norm_lm.shape[0] == 1024) 
#             assert(norm_s2v.shape[1] == norm_lm.shape[1]) 
            
#             pad_s2v = F.pad(norm_s2v, (0, max_len - norm_s2v.shape[1]))
#             pad_lm = F.pad(norm_lm, (0, max_len - norm_lm.shape[1]))
            
#             s2v_list.append(pad_s2v)
#             lm_list.append(pad_lm)
#             n_num_list.append(node_num)
        
#         s2v_batch = torch.stack(s2v_list, dim=0)
#         lm_batch = torch.stack(lm_list, dim=0)
#         node_num = torch.tensor(n_num_list)
        
#     else:
#         s2v_list = []
#         lm1_list = []
#         lm2_list = []
#         n_num_list = []
#         for _ in range(len(data)):
#             node_num = data[_].num_nodes
            
#             s2v = data[_].s
#             lm1 = data[_].l0
#             lm2 = data[_].l1
            
#             t_s2v = s2v.permute(1, 0)
#             t_lm1 = lm1.permute(1, 0)
#             t_lm2 = lm2.permute(1, 0)
            
#             norm_s2v = tensor2d_norm(t_s2v) 
#             norm_lm1 = tensor2d_norm(t_lm1) 
#             norm_lm2 = tensor2d_norm(t_lm2) 
#             assert(norm_s2v.shape[0] == 1024 and norm_lm1.shape[0] == 1024) 
#             assert(norm_s2v.shape[1] == norm_lm1.shape[1]) 
            
#             pad_s2v = F.pad(norm_s2v, (0, max_len - norm_s2v.shape[1]))
#             pad_lm1 = F.pad(norm_lm1, (0, max_len - norm_lm1.shape[1]))
#             pad_lm2 = F.pad(norm_lm2, (0, max_len - norm_lm2.shape[1]))
            
#             s2v_list.append(pad_s2v)
#             lm1_list.append(pad_lm1)
#             lm2_list.append(pad_lm2)
#             n_num_list.append(node_num)
        
#         s2v_batch = torch.stack(s2v_list, dim=0)
#         node_num = torch.tensor(n_num_list)
        
#         lm1_batch = torch.stack(lm1_list, dim=0)
#         lm2_batch = torch.stack(lm2_list, dim=0)
        
#         lm_batch = [lm1_batch, lm2_batch]
    
#     return s2v_batch, lm_batch, node_num 


def get_cnn_minibatch(data, use_lm, max_len = 101):
    if len(use_lm) == 1:
        s2v_list = []
        lm_list = []
        n_num_list = []
        for _ in range(len(data)):
            node_num = data[_].num_nodes
            
            s2v = data[_].s
            lm = data[_].l
            t_s2v = s2v.permute(1, 0)
            t_lm = lm.permute(1, 0)
            
            norm_s2v = tensor2d_norm(t_s2v) 
            norm_lm = tensor2d_norm(t_lm) 
            assert(norm_s2v.shape[0] == 1024 and norm_lm.shape[0] == 1024) 
            assert(norm_s2v.shape[1] == norm_lm.shape[1]) 
            
            pad_s2v = F.pad(norm_s2v, (0, max_len - norm_s2v.shape[1]))
            pad_lm = F.pad(norm_lm, (0, max_len - norm_lm.shape[1]))
            
            s2v_list.append(pad_s2v)
            lm_list.append(pad_lm)
            n_num_list.append(node_num)
        
        s2v_batch = torch.stack(s2v_list, dim=0)
        lm_batch = torch.stack(lm_list, dim=0)
        node_num = torch.tensor(n_num_list)
        
    else:
        s2v_list = []
        n_num_list = []
        lm_num = len(use_lm) 
        lm_list = [list() for _ in range(lm_num)]
        lm_batch = []
        
        for i in range(len(data)):
            node_num = data[i].num_nodes
            
            s2v = data[i].s
            t_s2v = s2v.permute(1, 0)
            norm_s2v = tensor2d_norm(t_s2v)
            assert norm_s2v.shape[0] == 1024
            pad_s2v = F.pad(norm_s2v, (0, max_len - norm_s2v.shape[1]))
            s2v_list.append(pad_s2v)
            
            lm_data = []
            for j in range(lm_num):
                lm = getattr(data[i], f"l{j}")
                t_lm = lm.permute(1, 0)
                norm_lm = tensor2d_norm(t_lm)
                assert norm_lm.shape[0] == 1024
                pad_lm = F.pad(norm_lm, (0, max_len - norm_lm.shape[1]))
                lm_list[j].append(pad_lm)
                
            n_num_list.append(node_num)
        
        for i in range(len(lm_list)):
            lm_batch_single = torch.stack(lm_list[i], dim=0)
            lm_batch.append(lm_batch_single)
            
        s2v_batch = torch.stack(s2v_list, dim=0)
        node_num = torch.tensor(n_num_list)
    
    return s2v_batch, lm_batch, node_num 


def stat(model, acc, auc, file):

    data = {'model': model, 'acc': acc, 'auc': auc}

    filename = file
    file_exists = False
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            if headers == ['model', 'acc', 'auc']:
                file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['model', 'acc', 'auc'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
        
        
def train_stat(model, acc, auc, e, file):

    data = {'model': model, 'acc': acc, 'auc': auc, 'best_epoch': e,}

    filename = file
    file_exists = False
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            if headers == ['model', 'acc', 'auc', 'best_epoch',]:
                file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['model', 'acc', 'auc', 'best_epoch',])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
        
        
def test_stat(model, acc, auc, f1, mcc, sn, sp, cm, file):

    data = {
        'model': model,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'mcc': mcc,
        'sn': sn,
        'sp': sp,
        'cm': str(cm.tolist()),
    }

    filename = file
    file_exists = False
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            if headers == ['model', 'acc', 'auc', 'f1', 'mcc', 'sn', 'sp', 'cm']:
                file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['model', 'acc', 'auc', 'f1', 'mcc', 'sn', 'sp', 'cm'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
        
        

def test_stat_aupr(model, acc, auc, aupr, f1, mcc, sn, sp, cm, file):

    data = {
        'model': model,
        'acc': acc,
        'auc': auc,
        'aupr': aupr, 
        'f1': f1,
        'mcc': mcc,
        'sn': sn,
        'sp': sp,
        'cm': str(cm.tolist()),
    }

    filename = file
    file_exists = False
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            if headers == ['model', 'acc', 'auc', 'aupr', 'f1', 'mcc', 'sn', 'sp', 'cm']:
                file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['model', 'acc', 'auc', 'aupr', 'f1', 'mcc', 'sn', 'sp', 'cm'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
        
        
def save_results(id, true, pred, prob, file):
    
    df = pd.DataFrame({
        'id': id,
        'true': true,
        'pred': pred,
        'prob': prob
    })
    
    df.to_csv(file, index=False)
    
        
        
# ! ########
        
def mll_test_stat(model, acc, pre, auc, f1, mcc, file):

    data = {
        'model': model,
        'acc': acc,
        'pre': pre,
        'auc': auc,
        'mcc': mcc,
        'f1': f1,
    }

    filename = file
    file_exists = False
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            if headers == ['model', 'acc', 'pre', 'auc', 'mcc', 'f1']:
                file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['model', 'acc', 'pre', 'auc', 'mcc', 'f1'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
        
        
from sklearn.model_selection import train_test_split

def imbalence_split(graph_data_list, keep_ratio, seed):
    
    test_size = 1 - keep_ratio

    train_dataset, drop_dataset = train_test_split(graph_data_list, test_size=test_size, random_state=seed, shuffle=False)
    
    return train_dataset, drop_dataset