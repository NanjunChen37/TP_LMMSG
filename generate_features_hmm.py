import os
import utils.hhblits_search as hh
import utils.psiblast_search as psi
import yaml
import argparse
import time

# Load the paths of tools and databases 
with open("config.yaml", 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

psiblast = cfg['psiblast']
hhblits = cfg['hhblits']
rosetta = cfg['rosetta']

# Databases and model
nrdb90 = cfg['nrdb90']
nr = cfg['nr']
# nr = "/media/cnj/OneTouch/cnjdata/nr/nr_decomp/nr_decomp"
uniclust = cfg['uniclust']
rosetta_model = cfg['rosetta_model']


def generate_features(args):
    """
    """
    feas = args.feas
    
    if 'HHM' in feas:
        hh.run(hhblits, args.hhm_ifasta, args.hhm_oa3m, args.hhm_ohhm, args.hhm_tmp, uniclust)

    if 'NPZ' in feas:
        rosetta_cmd = 'python ' + rosetta + ' ' + \
                      args.tr_ia3m + ' ' + args.tr_onpz + ' -m ' + rosetta_model
        os.system(rosetta_cmd)

    if 'PSSM' in feas:
        psi.run(psiblast, args.pssm_ifasta, args.pssm_opssm, nrdb90, nr)

if __name__ == '__main__':
    # generate contact map, pssm and hhm features before train and test model.
    parser = argparse.ArgumentParser()
    parser.add_argument('-feas', type=str, nargs='+', default=['PSSM', 'HHM', 'NPZ'], help='Feature names')

    # HHblits parameters
    parser.add_argument('-hhm_ifasta', type=str, default='data/XU_train/positive/XU_pretrain_val_positive_example.fasta',
                        help='Input a file with fasta format for hhblits search')
    parser.add_argument('-hhm_oa3m', type=str, default='data/XU_train/positive/a3m/',
                        help='Output folder saving .a3m files')
    parser.add_argument('-hhm_ohhm', type=str, default='data/XU_train/positive/hhm/',
                        help='Output folder saving .hhm files')
    parser.add_argument('-hhm_tmp', type=str, default='data/XU_train/positive/tmp/', 
                        help='Temp folder')

    # trRosetta parameters
    parser.add_argument('-tr_ia3m', type=str, default='data/XU_train/positive/a3m/',
                        help='Input folder saving .a3m files')
    parser.add_argument('-tr_onpz', type=str, default='data/XU_train/positive/npz/',
                        help='Output folder saving .npz files')

    # PSSM parameters
    parser.add_argument('-pssm_ifasta', type=str, default='data/XU_train/positive/XU_pretrain_val_positive_example.fasta', 
                        help='Input .fasta file for psiblast search')
    parser.add_argument('-pssm_opssm', type=str, default='data/XU_train/positive/blos/', 
                        help='Output folder saving .pssm files')

    args = parser.parse_args()
    
    start_time = time.time()

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
