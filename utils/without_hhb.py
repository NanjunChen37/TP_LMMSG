import os
import argparse
from tqdm.std import trange

def run(input_fasta, sa3m_folder):
    ids = []
    seqs = []
    with open(input_fasta, 'r') as f:
        lines = f.readlines()
        print(f'Num of fasta seqs is {len(lines)/2}')
        for line in lines:
                line = line.strip()
                if line[0] == '>':
                    ids.append(line)
                else:
                    seqs.append(line)
                    
        if not os.path.exists(sa3m_folder):
            os.makedirs(sa3m_folder)
            
        for f in os.listdir(sa3m_folder):
            os.remove(sa3m_folder + f)
            
        for i in range(len(ids)):
            name = ids[i]
            fname = name.replace('|', '_')[1:]
            seq = seqs[i]
            with open(sa3m_folder + fname + '.a3m', 'w') as f:
                f.write(name + '\n')
                f.write(seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='/home/cnj/cs/peptide/sAMPpred-GAT-test/data/train_data/positive/XU_pretrain_val_positive.fasta', 
                        help='Input files in fasta format')
    parser.add_argument('-oa3m', type=str, default='/home/cnj/cs/peptide/sAMPpred-GAT-test/data/train_data/positive/single_a3m_val/', 
                        help='Output folder saving o3m files')
    args = parser.parse_args()
    
    input_fasta = args.i
    sa3m_folder = args.oa3m
    
    run(input_fasta, sa3m_folder)