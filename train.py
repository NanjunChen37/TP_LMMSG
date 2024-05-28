import warnings
import torch.nn.functional as F
import torch
import time
import os
import random
import numpy as np
import logging.config
import logging
import datetime
import argparse
from utils.utils import get_cnn_minibatch, stat, train_stat, current_time, imbalence_split
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from node_embed.featurization import load_data_adv  
from node_embed.multi_label_feature import load_data_mll, get_multi_label_metrics
from models.models import GATModel, GATModel1, GATModel2, GATModel3, GATModelMll
from logger.logger import get_logger


warnings.filterwarnings("ignore")

def fix_seed(seed):
    if seed is None:
        seed = random.randint(1, 10000)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    # torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

def train(args, logger):
    threshold = args.d
    task = 'cls'
    is_fea_sep = args.feature_sep
    use_hhm = args.use_hhm
    use_all_nr = args.use_all_nr
    use_lm = args.use_lm
    fix_seed(args.seed)
    
    train_start_time = datetime.datetime.now()
    
    # loading and spliting data
    if args.pos_v == "" or args.neg_v == "":
        cross_valid = args.cross_valid
        train_fasta_path = args.pos_t
        train_npz_path = args.pos_npz
        # data_list, labels = load_data(fasta_path_positive, npz_dir_positive, threshold, 1)
        if args.pos_v == "":
            data_list, labels = load_data_adv(train_fasta_path, 1, task, train_npz_path, threshold, is_fea_sep, use_hhm, use_all_nr, use_lm, cross_valid)
            data_train, data_val, y_t, y_v = train_test_split(data_list, labels, test_size=0.2, shuffle=True, random_state=args.seed)
        else:
            val_fasta_path = args.pos_v
            val_npz_path = args.pos_npz
            data_train, _ = load_data_adv(train_fasta_path, 1, task, train_npz_path, threshold, is_fea_sep, use_hhm, use_all_nr, use_lm, cross_valid)
            data_val, _ = load_data_adv(val_fasta_path, 1, task, val_npz_path, threshold, is_fea_sep, use_hhm, use_all_nr, use_lm, cross_valid)
                 
    else:
        fasta_path_train_positive = args.pos_t
        fasta_path_val_positive = args.pos_v
        npz_dir_pos = args.pos_npz
        data_train, _ = load_data_adv(fasta_path_train_positive, 1, task, npz_dir_pos, threshold, is_fea_sep, use_hhm, use_all_nr, use_lm, False)
        data_val, _ = load_data_adv(fasta_path_val_positive, 1, task, npz_dir_pos, threshold, is_fea_sep, use_hhm, use_all_nr, use_lm, False)

        if hasattr(args, 'ratio'):
            data_train, _ = imbalence_split(data_train, args.ratio, args.seed)
            data_val, _ = imbalence_split(data_val, args.ratio, args.seed)

        fasta_path_train_negative = args.neg_t
        fasta_path_val_negative = args.neg_v
        npz_dir_neg = args.neg_npz
        neg_data_train, _ = load_data_adv(fasta_path_train_negative, 0, task, npz_dir_neg, threshold, is_fea_sep, use_hhm, use_all_nr, use_lm, False)
        neg_data_val, _ = load_data_adv(fasta_path_val_negative, 0, task, npz_dir_neg, threshold, is_fea_sep, use_hhm, use_all_nr, use_lm, False)

        data_train.extend(neg_data_train)
        data_val.extend(neg_data_val)

        data_train = shuffle(data_train)  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug(f'device: {str(device)}')

    node_feature_dim = data_train[0].x.shape[1]
    n_class = 2

    # tensorboard, record the change of auc, acc and loss
    writer = SummaryWriter()

    if args.pretrained_model == "":
        if args.cnn_od is None:
            model = GATModel(node_feature_dim, args.hd, n_class, args.drop, args.heads).to(device) # ! model here
        elif args.cnn_od is not None:
            model = GATModel3(node_feature_dim + args.cnn_od, args.hd, n_class, args.drop, args.use_lm, args.heads, args.top_k).to(device)
    else:
        model = torch.load(args.pretrained_model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    
    train_dataloader = DataLoader(data_train, batch_size=args.b)
    val_dataloader = DataLoader(data_val, batch_size=args.b)

    early_stop = args.es
    best_epoch = 0
    best_acc = 0
    best_auc = 0
    min_loss = 1000
    
    os.makedirs(args.save.split('/')[:-1][0], exist_ok=True)
    save_acc = '/'.join(args.save.split('/')[:-1]) + '/acc_' + args.save.split('/')[-1] + '_' + str(args.exp_num)
    save_auc = '/'.join(args.save.split('/')[:-1]) + '/auc_' + args.save.split('/')[-1] + '_' + str(args.exp_num)
    save_loss = '/'.join(args.save.split('/')[:-1]) + '/loss_' + args.save.split('/')[-1] + '_' + str(args.exp_num)
    save_auc_dict = '/'.join(args.save.split('/')[:-1]) + '/auc_' + args.save.split('/')[-1] + '_' + str(args.exp_num) + '.pth'
    
    for epoch in range(args.e):
        logger.debug(f'Epoch: {epoch}')
        
        # * train
        model.train()
        arr_loss = []
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            data = data.to(device)
            
            if args.cnn_od is None:
                output = model(data.x, data.edge_index, data.batch) # todo: train_output
            elif args.cnn_od is not None:
                s2v_batch, lm_batch, n_num = get_cnn_minibatch(data, use_lm, args.max_length)
                output = model(data.x, data.edge_index, s2v_batch, lm_batch, n_num, data.batch)
            
            out = output[0]
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            arr_loss.append(loss.item())
        
        avgl = np.mean(arr_loss)
        logger.debug(f"Average Training loss: {avgl:.5f}")

        # * val
        model.eval()
        with torch.no_grad():
            total_num = 0
            total_correct = 0
            preds = []
            y_true = []
            arr_loss = []
            for data in val_dataloader:
                data = data.to(device)

                if args.cnn_od is None:
                    output = model(data.x, data.edge_index, data.batch) # todo: train_output
                elif args.cnn_od is not None:
                    s2v_batch, lm_batch, n_num = get_cnn_minibatch(data, use_lm, args.max_length)
                    output = model(data.x, data.edge_index, s2v_batch, lm_batch, n_num, data.batch)
                    
                out = output[0]
                loss = criterion(out, data.y)
                arr_loss.append(loss.item())

                pred = out.argmax(dim=1)
                score = F.softmax(out, dim=1)[:, 1]
                correct = (pred == data.y).sum().float()
                total_correct += correct
                total_num += data.num_graphs
                preds.extend(score.cpu().detach().data.numpy())
                y_true.extend(data.y.cpu().detach().data.numpy())

            acc = (total_correct / total_num).cpu().detach().data.numpy()
            auc = roc_auc_score(y_true, preds)
            val_loss = np.mean(arr_loss)
            
            
            logger.debug(f'val loss: {val_loss:.5f}, val acc: {acc:.5f}, val auc: {auc:.5f}')

            writer.add_scalar('Loss', avgl, global_step=epoch)
            writer.add_scalar('Val Loss', val_loss, global_step=epoch)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr(), global_step=epoch)
            writer.add_scalar('acc', acc, global_step=epoch)
            writer.add_scalar('auc', auc, global_step=epoch)

            if acc > best_acc:
                best_acc = acc
                torch.save(model, save_acc) 

            if auc > best_auc:
                best_auc = auc
                torch.save(model, save_auc)
                torch.save(model.state_dict(), save_auc_dict)
                best_epoch = epoch

            if np.mean(val_loss) < min_loss:
                min_loss = val_loss
                torch.save(model, save_loss)
                
            if epoch - best_epoch > early_stop:
                print("Early stop at %d, %s " % (epoch, args.exp_id))
                break
            
            logger.debug('-' * 50)
            
        # * step
        scheduler.step()

    logger.info('=' * 50)
    logger.info(f'best epoch: {best_epoch}, best acc: {best_acc:.5f}, best auc: {best_auc:.5f}')
    logger.info(f'localtime = {time.asctime(time.localtime(time.time()))}')
    
    
    if args.train_csv_file is not None:
        train_stat(args.exp_id, best_acc, best_auc, best_epoch, args.train_csv_file)


    return save_auc, best_acc, best_auc


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
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
    
    # * train input file
    parser.add_argument('-pos_t', type=str, help='positive training dataset',
                        default='data/XU_train/positive/XU_pretrain_train_positive_example.fasta')
    
    parser.add_argument('-pos_v', type=str, help='positive validation dataset',
                        default='data/XU_train/positive/XU_pretrain_val_positive_example.fasta')
    
    parser.add_argument('-pos_npz', type=str, help='path of the positive npz folder',
                        default='data/XU_train/positive/npz_no_hhm')

    parser.add_argument('-neg_t', type=str, help='negative training dataset',
                        default='data/XU_train/negative/XU_pretrain_train_negative_example.fasta')
    
    parser.add_argument('-neg_v', type=str, help='negative validation dataset',
                        default='data/XU_train/negative/XU_pretrain_val_negative_example.fasta')
    
    parser.add_argument('-neg_npz', type=str, help='path of the negative npz folder',
                        default='data/XU_train/negative/npz_no_hhm')
    
    # # * train input file
    # parser.add_argument('-pos_t', type=str, help='positive training dataset',
    #                     default='data/train_data/positive/XU_pretrain_train_positive.fasta')
    
    # parser.add_argument('-pos_v', type=str, help='positive validation dataset',
    #                     default='data/train_data/positive/XU_val_positive.fasta')
    
    # parser.add_argument('-pos_npz', type=str, help='path of the positive npz folder',
    #                     default='data/train_data/positive/npz_no_hhm')

    # parser.add_argument('-neg_t', type=str, help='negative training dataset',
    #                     default='data/train_data/negative/XU_pretrain_train_negative.fasta')
    
    # parser.add_argument('-neg_v', type=str, help='negative validation dataset',
    #                     default='data/train_data/negative/XU_val_negative.fasta')
    
    # parser.add_argument('-neg_npz', type=str, help='path of the negative npz folder',
    #                     default='data/train_data/negative/npz_no_hhm')
    
    # * hyper-params
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate') 
    parser.add_argument('-drop', type=float, default=0.4, help='Dropout rate') 
    parser.add_argument('-top_k', type=float, default=10, help='top_k pooling') # !
    parser.add_argument('-e', type=int, default=2, help='Maximum number of epochs') # default 200
    parser.add_argument('-es', type=int, default=30, help='Early stop') 
    parser.add_argument('-b', type=int, default=128, help='Batch size') 
    parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim')
    parser.add_argument('-heads', type=int, default=8, help='Number of heads') 
    parser.add_argument('-max_length', type=int, default=101, help='Max sequence length add 1 (e.g. 100+1)')  # !
    parser.add_argument('-cnn_od', type=int, default=64, help='cnn output layer dim, choice int or None') # !
    
    # * model paths
    parser.add_argument('-pretrained_model', type=str, default="", 
                        help='The path of pretraining model')
    parser.add_argument('-save', type=str, default='saved_models/train_example.model', 
                        help='The path saving the trained models')
    
    # * distance threshold
    parser.add_argument('-d', type=int, default=27, help='Distance threshold 37-20A, 27-15A, 17-10A')
    
    args = parser.parse_args()
    
    args.train_csv_file = None
    # args.train_csv_file = current_time() + '_train_stat.csv'
    
    logger = get_logger(args.log_conf)
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in dict(vars(args)).items()))
    logger.info(f"============ Experiment ID: {args.exp_id} ============")

    train(args, logger)

    logger.info('=' * 20 + 'END' + '=' * 50)