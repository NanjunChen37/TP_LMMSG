import gc
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils.utils import current_time
from utils.visualization import draw_roc
from node_embed.featurization import load_seqs

import numpy as np
from train import train
from test import independent_test
import logging
from logger.logger import get_logger


def data_shuffle(data_pos, data_neg, shuffle, seed):
    
    pos_data_array = np.array(list(zip(data_pos[0], data_pos[1])))
    neg_data_array = np.array(list(zip(data_neg[0], data_neg[1])))
    pos_label, neg_label = data_pos[2], data_neg[2]
    
    data_array = np.concatenate((pos_data_array, neg_data_array), axis=0)
    labels = np.concatenate((pos_label, neg_label), axis=0)
    
    np.random.seed(seed)
    random_ind = np.random.permutation(len(labels))
    s_data_array = data_array[random_ind]
    s_labels = labels[random_ind]

    return s_data_array, s_labels


def save_fasta_with_flag(fasta_array, labels, file_path, flag='training'):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(['{}|{}|{}\n{}'.format(fasta[0], label, flag, fasta[1]) for fasta, label in zip(fasta_array, labels)]) + '\n')
        f.flush()


def k_fold_split(pos_fasta_path, neg_fasta_path, save_path_dir, k_fold, seed):

    data_pos = load_seqs(pos_fasta_path, label=1, task='cls')
    data_neg = load_seqs(neg_fasta_path, label=0, task='cls')
    
    dataset, labels = data_shuffle(data_pos, data_neg, True, seed)
    kf = KFold(n_splits=k_fold, shuffle=True)  
    
    i = 1
    for train_index, test_index in kf.split(dataset):
        i_fold = '{}_fold'.format(i)
        data_save_dir = os.path.join(save_path_dir, 'data', i_fold)
        os.makedirs(data_save_dir, exist_ok=True)
        trainset_path = os.path.join(data_save_dir, 'train.fasta')
        testset_path = os.path.join(data_save_dir, 'test.fasta')

        x_train, y_train = dataset[train_index], labels[train_index]
        x_test, y_test = dataset[test_index], labels[test_index]
        
        save_fasta_with_flag(x_train, y_train, trainset_path, 'training')
        save_fasta_with_flag(x_test, y_test, testset_path, 'testing')
        
        print(f'{i} fold has been created.')
        i += 1


def result_combination(root_dir, sub_dir, keyword):
    df = pd.DataFrame()
    for dir_name in os.listdir(root_dir):
        if 'data'==dir_name:
            continue
        if os.path.isdir(os.path.join(root_dir, dir_name)):
            result_dir = os.path.join(root_dir, dir_name, sub_dir)
            for file_name in os.listdir(result_dir):
                if keyword in file_name:
                    df = df._append(pd.read_csv(os.path.join(result_dir, file_name)))
    
    df_mean = []
    for col in df:
        if col == 'model':
            df = df.drop('model', axis=1)
        elif col == 'cm':
            df_mean.append(np.array([eval(i) for i in df[col].tolist()]).mean(axis=0).astype(int).tolist()) 
        else:
            df_mean.append(str(round(df[col].mean(), 4)) + '±' + str(round(df[col].std(), 4)))
            
    df.insert(0, 'k_fold', ['{}_th'.format(i+1) for i in range(len(df))], allow_duplicates=False)
    df = df.round(4)
    df.loc[len(df)] = ['Mean'] + df_mean
    df.to_csv(os.path.join(root_dir, 'result.csv'), index=False)
    print('result has combinated in:', os.path.join(root_dir, 'result.csv'))
                    

def get_rocs(root_dir):
    
    fprs, tprs = [], []
    for sub_dir in os.listdir(root_dir):
        
        if os.path.isfile(os.path.join(root_dir, sub_dir)) or sub_dir=='data':
            continue
        
        roc_dir = os.path.join(root_dir, sub_dir, 'test')
        for roc_file in os.listdir(roc_dir):
            if 'roc' not in roc_file:
                continue
            roc_path = os.path.join(roc_dir, roc_file)
            # method_name = roc_file.split('_')[-1][:-4]
            # method_name = sub_dir if method_name == 'roc' else method_name
            roc_df = pd.read_csv(roc_path)
            fprs.append(roc_df['fpr'])
            tprs.append(roc_df['tpr'])
            
    avg_auc = pd.read_csv(os.path.join(root_dir,'result.csv'))['auc'].iloc[-1].split('±')
    avg_auc = list(map(float,avg_auc))
    
    return fprs, tprs, avg_auc
        
        
def k_fold_training(args, logger, pos_fasta_path, neg_fasta_path, model_name, k_fold, k_fold_dir, seed):
    
    if args.k_fold_rootpath is None:
        k_fold_rootpath = os.path.join(k_fold_dir, '{}_fold_{}_{}'.format(k_fold, model_name, current_time()))
        os.makedirs(k_fold_rootpath, exist_ok=True)
        k_fold_split(pos_fasta_path, neg_fasta_path, k_fold_rootpath, k_fold, seed)
    else:
        k_fold_rootpath = args.k_fold_rootpath
    
    for i in range(k_fold):
        i_fold = f'{i+1}_fold'
        print(f'The {i_fold} of {k_fold}_fold is running...')
        
        result_save_dir = os.path.join(k_fold_rootpath, i_fold)
        os.makedirs(result_save_dir, exist_ok=True)
        
        trainset_path = os.path.join(k_fold_rootpath, 'data', i_fold, 'train.fasta')
        testset_path = os.path.join(k_fold_rootpath, 'data', i_fold, 'test.fasta')
        
        args.cross_valid = True
        
        if hasattr(args, 'test_pos_t') and hasattr(args, 'test_neg_t'):
            # * train
            args.pos_t = trainset_path
            args.pos_v = testset_path
            args.neg_v = ""
            args.save = os.path.join(result_save_dir, 'model.model')
            train_results = train(args, logger)
            # * independent test
            args.saved_model = train_results[0]
            test_results_dir = os.path.join(result_save_dir, 'test')
            os.makedirs(test_results_dir, exist_ok=True)
            args.test_csv_file = os.path.join(test_results_dir, 'test.csv')
            args.result_csv_file = os.path.join(test_results_dir, f'result_{i+1}.csv')
            args.roc_fig_path = os.path.join(test_results_dir, 'curve.jpg')
            args.roc_curve_path = os.path.join(test_results_dir, 'roc.txt')
            args.cross_valid = None
            independent_test(args, logger)
            
            gc.collect()
            
        else:
            # * train
            args.pos_t = trainset_path
            args.pos_v = ""
            args.test_pos_t = testset_path
            args.test_pos_npz = args.pos_npz
            args.save = os.path.join(result_save_dir, 'model.model')
            train_results = train(args, logger)
            # * test
            args.saved_model = train_results[0]
            test_results_dir = os.path.join(result_save_dir, 'test')
            os.makedirs(test_results_dir, exist_ok=True)
            args.test_csv_file = os.path.join(test_results_dir, 'test.csv')
            args.result_csv_file = os.path.join(test_results_dir, f'result_{i+1}.csv')
            args.roc_fig_path = os.path.join(test_results_dir, 'curve.jpg')
            args.roc_curve_path = os.path.join(test_results_dir, 'roc.txt')
            independent_test(args, logger)
            
            gc.collect()
        
    result_combination(k_fold_rootpath, 'test', 'test')
    fig = draw_roc(*get_rocs(k_fold_rootpath))
    fig.savefig(os.path.join(k_fold_rootpath,'roc.pdf'), dpi=300)
    plt.close()
    
    print("###")
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # * model name
    parser.add_argument('--model', type=str, default='10fold_cv',) # !
    
    # * k_fold
    parser.add_argument('--k_fold_rootpath', type=str, default=None,
                        help='relative path')
    parser.add_argument('--fold', type=int, default=10,
                        help='Numbers of k fold (default: 10)')
    parser.add_argument('--save', type=str, default='cross_val/cls',
                        help='Result save path (default: result)')
    # * test
    parser.add_argument('-runtest', type=bool, default=True, help='')
    
    # * arguments
    parser.add_argument('-exp_id', type=str, default='model_train', help='')
    parser.add_argument('-exp_num', type=int, default='0', help='default setting')
    parser.add_argument('-task', type=str, default='cls', help='default setting')
    parser.add_argument('-log_conf', type=str, default='logger/logging.conf', help='')
    parser.add_argument('-seed', type=int, default=None, help='')
    
    # * feature
    parser.add_argument('-use_all_nr', type=bool, default=False, help='')
    parser.add_argument('-feature_sep', type=bool, default=True, help='')
    parser.add_argument('-use_hhm', type=bool, default=False, help='Use false if HHBlits is not used') # !
    parser.add_argument('-use_lm', type=str, default=['uni', 'bfd', 'brt'], nargs="+", help='LM folder name') # ['uni', 'bfd', 'brt']
    
    # * data
    parser.add_argument('--pos', type=str, default='data/cv_data/XU_train/XU_pretrain_all_positive.fasta',
                        help='Choose positive dataset path')
    parser.add_argument('--neg', type=str, default='data/cv_data/XU_train/XU_pretrain_all_negative.fasta',
                        help='Choose negative dataset path')
    parser.add_argument('-pos_npz', type=str, help='Path of the overall npz folder',
                        default='data/cv_data/XU_train/npz_no_hhm')
                        
    # * independent test
    parser.add_argument('-test_pos_t', type=str, default='data/cv_data/XU_test/XU_AMP.fasta',
                        help='Path of the positive test dataset')
    parser.add_argument('-test_pos_npz', type=str, help='Path of the positive npz folder',
                        default='data/cv_data/XU_test/npz_no_hhm')
    
    parser.add_argument('-test_neg_t', type=str, default='data/cv_data/XU_test/XU_nonAMP.fasta',
                        help='Path of the negative test dataset')
    parser.add_argument('-test_neg_npz', type=str, help='Path of the positive npz folder',
                        default='data/cv_data/XU_test/npz_no_hhm')
                        
    # * hyper-params
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate') # default 0.0001 for train
    parser.add_argument('-drop', type=float, default=0.4, help='Dropout rate') # default 0.4
    parser.add_argument('-top_k', type=float, default=10, help='top_k pooling') # !
    parser.add_argument('-e', type=int, default=200, help='Maximum number of epochs') # default 50
    parser.add_argument('-es', type=int, default=30, help='Early stop') # default 30
    parser.add_argument('-b', type=int, default=128, help='Batch size') # default 128
    parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim') # default 64
    parser.add_argument('-heads', type=int, default=8, help='Number of heads') # default 8
    parser.add_argument('-max_length', type=int, default=101, help='Max sequence length add 1 (e.g. 100+1)') # default 101 # !
    parser.add_argument('-cnn_od', type=int, default=64, help='cnn output layer dim') # !
    
    # * model paths
    parser.add_argument('-pretrained_model', type=str, default="", 
                        help='The path of pretraining model')
    parser.add_argument('-save', type=str, default='saved_models/train_example.model', 
                        help='The path saving the trained models')

    args = parser.parse_args()
    
    pos_fasta_path = args.pos
    neg_fasta_path = args.neg
    model_name = args.model
    k_fold = args.fold
    k_fold_dir = args.save
    seed = args.seed
    
    args.train_csv_file = current_time() + '_train_stat.csv'
    args.test_csv_file = None
        
    logger = get_logger(args.log_conf, 'kfold')
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in dict(vars(args)).items()))
    logger.info(f"============ Experiment ID: {args.exp_id} ============")
    
    k_fold_training(args, logger, pos_fasta_path, neg_fasta_path, model_name, k_fold, k_fold_dir, seed)
    