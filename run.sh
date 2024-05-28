#! /bin/bash
## This bash script is for quick pre-processing

## Path settings
train_pos_dir=data/XU_train/positive
train_neg_dir=data/XU_train/negative
test_pos_dir=data/XU_test/positive
test_neg_dir=data/XU_test/negative

train_pos=$train_pos_dir/XU_pretrain_train_positive_example.fasta
train_neg=$train_neg_dir/XU_pretrain_train_negative_example.fasta
val_pos=$train_pos_dir/XU_pretrain_val_positive_example.fasta
val_neg=$train_neg_dir/XU_pretrain_val_negative_example.fasta
test_pos=$test_pos_dir/test_AMP_example.fasta
test_neg=$test_neg_dir/test_nonAMP_example.fasta

## check path to conda with: conda info | grep -i 'base environment'
source /home/cnj/anaconda3/etc/profile.d/conda.sh

conda activate tp_pre 

python generate_features_no_hmm.py -i $train_pos -oa3m $train_pos_dir/a3m_no_hhm/ -tr_ia3m $train_pos_dir/a3m_no_hhm/ -tr_onpz $train_pos_dir/npz_no_hhm/ -pssm_ifasta $train_pos -pssm_opssm $train_pos_dir/blos/

python generate_features_no_hmm.py -i $train_neg -oa3m $train_neg_dir/a3m_no_hhm/ -tr_ia3m $train_neg_dir/a3m_no_hhm/ -tr_onpz $train_neg_dir/npz_no_hhm/ -pssm_ifasta $train_neg -pssm_opssm $train_neg_dir/blos/

python generate_features_no_hmm.py -i $val_pos -oa3m $train_pos_dir/a3m_no_hhm/ -tr_ia3m $train_pos_dir/a3m_no_hhm/ -tr_onpz $train_pos_dir/npz_no_hhm/ -pssm_ifasta $val_pos -pssm_opssm $train_pos_dir/blos/

python generate_features_no_hmm.py -i $val_neg -oa3m $train_neg_dir/a3m_no_hhm/ -tr_ia3m $train_neg_dir/a3m_no_hhm/ -tr_onpz $train_neg_dir/npz_no_hhm/ -pssm_ifasta $val_neg -pssm_opssm $train_neg_dir/blos/

python generate_features_no_hmm.py -i $test_pos -oa3m $test_pos_dir/a3m_no_hhm/ -tr_ia3m $test_pos_dir/a3m_no_hhm/ -tr_onpz $test_pos_dir/npz_no_hhm/ -pssm_ifasta $test_pos -pssm_opssm $test_pos_dir/blos/

python generate_features_no_hmm.py -i $test_neg -oa3m $test_neg_dir/a3m_no_hhm/ -tr_ia3m $test_neg_dir/a3m_no_hhm/ -tr_onpz $test_neg_dir/npz_no_hhm/ -pssm_ifasta $test_neg -pssm_opssm $test_neg_dir/blos/

conda activate tp_lmmsg

python node_embed/LM_embedding.py -fasta_file $train_pos 

python node_embed/LM_embedding.py -fasta_file $train_neg

python node_embed/LM_embedding.py -fasta_file $val_pos

python node_embed/LM_embedding.py -fasta_file $val_neg

python node_embed/LM_embedding.py -fasta_file $test_pos

python node_embed/LM_embedding.py -fasta_file $test_neg

## The following is the train command

python train.py -b 32 -pos_t $train_pos -pos_v $val_pos -pos_npz $train_pos_dir/npz_no_hhm/ -neg_t $train_neg -neg_v $val_neg -neg_npz $train_neg_dir/npz_no_hhm/ -save saved_models/train_example.model

## The following is the test command

python test.py -test_pos_t $test_pos -test_pos_npz $test_pos_dir/npz_no_hhm/ -test_neg_t $test_neg -test_neg_npz $test_neg_dir/npz_no_hhm/ 
# python test.py -test_pos_t $test_pos -test_pos_npz $test_pos_dir/npz_no_hhm/ -test_neg_t $test_neg -test_neg_npz $test_neg_dir/npz_no_hhm/ -saved_model saved_models/models/auc_XUAMP_blos_no_hhm_uni_bfd_brt_t27.pth

## conda info --envs
## bash run_pre.sh