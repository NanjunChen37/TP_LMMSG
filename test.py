import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import argparse
# from utils.data_processing import load_data
from node_embed.featurization import load_data_adv  
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve, precision_score, auc
from models.models import GATModel, GATModel1, GATModel2
from utils.utils import get_cnn_minibatch, test_stat, test_stat_aupr, mll_test_stat, current_time, save_results
import logging
import logging.config
from logger.logger import get_logger
from utils.visualization import draw_roc
import os

from node_embed.multi_label_feature import load_data_mll, get_multi_label_metrics

def independent_test(args, logger):

    threshold = args.d
    task = 'cls'
    is_fea_sep = args.feature_sep
    use_hhm = args.use_hhm
    use_all_nr = args.use_all_nr
    use_lm = args.use_lm
    
    if not hasattr(args, 'cross_valid') or args.cross_valid is None:
        fasta_path_positive = args.test_pos_t
        npz_dir_positive = args.test_pos_npz
        data_list, labels = load_data_adv(fasta_path_positive, 1, task, npz_dir_positive, threshold, is_fea_sep, use_hhm, use_all_nr, use_lm, False)
        
        fasta_path_negative = args.test_neg_t
        npz_dir_negative = args.test_neg_npz
        neg_data = load_data_adv(fasta_path_negative, 0, task, npz_dir_negative, threshold, is_fea_sep, use_hhm, use_all_nr, use_lm, False)
        data_list.extend(neg_data[0])
        
    elif args.cross_valid:
        fasta_path = args.test_pos_t
        npz_path = args.test_pos_npz
        data_list, labels = load_data_adv(fasta_path, 1, task, npz_path, threshold, is_fea_sep, use_hhm, use_all_nr, use_lm, args.cross_valid)

        # labels = np.concatenate((labels, neg_data[1]), axis=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = torch.load(args.saved_model).to(device)

    test_dataloader = DataLoader(data_list, batch_size=args.b, shuffle=False)
    y_true = []
    y_pred = []
    y_prob = []

    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            
            if args.cnn_od is None:
                output = model(data.x, data.edge_index, data.batch)
            elif args.cnn_od is not None:
                s2v_batch, lm_batch, n_num = get_cnn_minibatch(data, use_lm, args.max_length)
                output = model(data.x, data.edge_index, s2v_batch, lm_batch, n_num, data.batch)
            
            out = output[0]

            pred = out.argmax(dim=1)
            # score = F.softmax(out)[:, 1]
            score = F.softmax(out, dim=1)[:, 1]

            y_prob.extend(score.cpu().detach().data.numpy().tolist())
            y_true.extend(data.y.cpu().detach().data.numpy().tolist())
            y_pred.extend(pred.cpu().detach().data.numpy().tolist())

        save_results(list(range(1, len(y_true) + 1)), y_true, y_pred, y_prob, args.result_csv_file)
        
        auc_v = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        
        prec, recall, _ = precision_recall_curve(y_true, y_prob)
        aupr = auc(recall, prec)

        logger.info(f"Test AUC: {auc_v:.5f}, ACC {acc:.5f}, f1 {f1:.5f}, MCC {mcc:.5f}, sn {sn:.5f}, sp {sp:.5f}")
        
        save_dir = args.save if hasattr(args, 'save') else args.saved_model
        # test_stat(save_dir, acc, auc, f1, mcc, sn, sp, cm, args.test_csv_file)
        test_stat_aupr(save_dir, acc, auc_v, aupr, f1, mcc, sn, sp, cm, args.test_csv_file)
        
        # if args.cross_valid:
        draw_roc([fpr], [tpr]).savefig(args.roc_fig_path, dpi=300) #!
        roc_df = pd.DataFrame((fpr, tpr), index=['fpr', 'tpr']).T
        roc_df.to_csv(args.roc_curve_path, index=False)
            
        
        return acc, auc_v, f1, mcc, sn, sp
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # * model paths
    parser.add_argument('-saved_model', type=str, help='The path of trained models',
                        default='saved_models/models/auc_XUAMP_blos_no_hhm_uni_bfd_brt_t27.pth')
    # * key args
    parser.add_argument('-task', type=str, default='cls', help='default setting')
    parser.add_argument('-log_conf', type=str, default='logger/logging.conf', help='')
    
    # * feature
    parser.add_argument('-use_all_nr', type=bool, default=False, help='')
    parser.add_argument('-feature_sep', type=bool, default=True, help='')
    parser.add_argument('-use_hhm', type=bool, default=False, help='Use false if HHBlits is not used') # !
    parser.add_argument('-use_lm', type=str, default=['uni', 'bfd', 'brt'], nargs="+", help='LM folder name') # ['uni', 'bfd', 'brt']
    
    # * input file
    parser.add_argument('-test_pos_t', type=str, help='positive test dataset',
                        default='data/XU_test/positive/test_AMP_example.fasta')
    
    parser.add_argument('-test_pos_npz', type=str, help='positive npz folder',
                        default='data/XU_test/positive/npz_no_hhm')

    parser.add_argument('-test_neg_t', type=str, help='negative test dataset',
                        default='data/XU_test/negative/test_nonAMP_example.fasta')
    
    parser.add_argument('-test_neg_npz', type=str, help='negative npz folder',
                        default='data/XU_test/negative/npz_no_hhm')
    
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
    
    # * distance threshold
    parser.add_argument('-d', type=int, default=27, help='Distance threshold 37-20A, 27-15A, 17-10A')
    
    args = parser.parse_args()
    
    train_stat_dir = './test_results'
    os.makedirs(train_stat_dir, exist_ok=True)
    
    args.test_csv_file = os.path.join(train_stat_dir, 'ave_test_stat.csv')
    args.result_csv_file = os.path.join(train_stat_dir, f'results.csv')
    args.roc_fig_path = os.path.join(train_stat_dir, current_time() + '_test_fig.png')
    args.roc_curve_path = os.path.join(train_stat_dir, current_time() + '_test_roc.txt')

    logger = get_logger(args.log_conf, 'test')
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in dict(vars(args)).items()))
    logger.info(f"============ Test with: {args.save if hasattr(args, 'save') else args.saved_model} ============")
    
    independent_test(args, logger)
