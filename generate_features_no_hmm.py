import os
import utils.hhblits_search as hh
import utils.psiblast_search as psi
from tqdm.std import trange
import yaml
import argparse
import time

# Load the paths of tools and databases 
with open("config.yaml", 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

psiblast = cfg['psiblast']
rosetta = cfg['rosetta']

# Databases and model
nrdb90 = cfg['nrdb90']
nr = cfg['nr']
uniclust = cfg['uniclust']
rosetta_model = cfg['rosetta_model']

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
                

def generate_features(args):
    
    feas = args.feas

    if 'NPZ' in feas:
        rosetta_cmd = 'python ' + rosetta + ' ' + args.tr_ia3m + ' ' + args.tr_onpz + ' -m ' + rosetta_model
        os.system(rosetta_cmd)
        
    if 'PSSM' in feas:
        psi.run(psiblast, args.pssm_ifasta, args.pssm_opssm, nrdb90)
        

if __name__ == '__main__':
    # generate contact map, pssm and hhm features before train and test model.
    parser = argparse.ArgumentParser()
    parser.add_argument('-feas', type=str, nargs='+', default=['NPZ','PSSM'], help='Feature names')
    
    # convert .a3m without MSA
    parser.add_argument('-i', type=str, default='data/XU_train/positive/XU_pretrain_val_positive_example.fasta', 
                        help='Input files in fasta format')
    parser.add_argument('-oa3m', type=str, default='data/XU_train/positive/a3m_no_hhm/', 
                        help='Output folder saving o3m files')

    # trRosetta parameters
    parser.add_argument('-tr_ia3m', type=str, default='data/XU_train/positive/a3m_no_hhm/',
                        help='Input folder saving .a3m files')
    parser.add_argument('-tr_onpz', type=str, default='data/XU_train/positive/npz_no_hhm/',
                        help='Output folder saving .npz files')
    
    # # PSSM parameters
    parser.add_argument('-pssm_ifasta', type=str, default='data/XU_train/positive/XU_pretrain_val_positive_example.fasta', 
                        help='Input .fasta file for psiblast search')
    parser.add_argument('-pssm_opssm', type=str, default='data/XU_train/positive/blos/', 
                        help='Output folder saving .pssm files')

    args = parser.parse_args()
    
    start_time = time.time()
    
    run(args.i, args.oa3m)

    generate_features(args)
    
    end_time = time.time()

    # total time
    total_time = end_time - start_time

    minutes = total_time // 60
    seconds = total_time % 60
    hours = minutes // 60
    minutes = minutes % 60

    print("Pre-processing start at:", time.ctime(start_time))
    print("Pre-processing end at:", time.ctime(end_time))
    print("Total time", hours, "hours", minutes, "mins", seconds, "secs")
