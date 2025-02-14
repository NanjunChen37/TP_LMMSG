## Literature Info
For detailed model benchmarking and architecture introduction, please see the publication: [TP-LMMSG: a peptide prediction graph neural network incorporating flexible amino acid property representation](https://academic.oup.com/bib/article/25/4/bbae308/7699353)


## Environments

#### To avoid potential dependency conflicts among Python packages, and facilitate future upgrades, we employed a dual-environment approach for preprocessing and model training.

tp_pre environment is used for structure prediction and graph construction. The key package includes:
```
conda create -n tp_pre python=3.7
conda activate tp_pre
pip install tensorflow==1.14.0
pip install tqdm==4.66.1
pip install pyyaml
pip install --upgrade protobuf==3.19.0
conda install -c conda-forge -c bioconda hhsuite==3.3.0
```

tp_lmmsg is used for feature extraction, training, and testing. The key package includes: 
```
conda create -n tp_lmmsg python=3.7
conda activate tp_lmmsg
pyparsing=2.4.7, 
rdflib=4.2.2
bio-embeddings=0.2.2, 
torch=1.10.0 + cuda11.1
conda install -c conda-forge -c bioconda hhsuite==3.3.0
gensim=4.2.0
```

The pyg package is required in tp_lmmsg:
```
wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${TORCH}%2B${CUDA}/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
torch_geometric==2.4.0
```
The detailed info for pyg whl installation can be found at [pyg whl](https://data.pyg.org/whl/).

We provided the .yml env configuration for a fast deployment:

```
conda env create -f tp_pre.yml
conda env create -f tp_lmmsg.yml
```

## Tools and databases

To run the model completely, additional tools and databases need to be installed. 

### 1.First select a temp dir for download and organization. 
```
cd
mkdir temp
```

### 2. In order to use psiblast for blast PSSM feature, psiblast and nrdb90 is required:

1) Install psiblast:

    Download:
    ```
    wget ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.12.0/ncbi-blast-2.12.0+-x64-linux.tar.gz
    ```

    Decompress:
    ```
    tar zxvf ncbi-blast-2.12.0+-x64-linux.tar.gz
    ```

    Add the path to the system environment in ~/.bashrc.
    ```
    vim ~/.bashrc
    export BLAST_HOME={your_path}/ncbi-blast-2.12.0+
    export PATH=$PATH:$BLAST_HOME/bin
    ```

    Reload and check:
    ```
    source ~/.bashrc
    psiblast -h
    ```

2) Download database:
    Users can download the nrdb90.tar.gz file from our [huggingface repo](https://huggingface.co/HongHongStand/TP_LMMSG/tree/main) and decompress it.


### 3. hhblits in our framework is not necessary, while user can still include the tool:

1) Install and check
    After conda install:
    ```
    conda install -c conda-forge -c bioconda hhsuite==3.3.0
    ```

    Check:
    ```
    hhblits -h
    ```

2) Download database:
    The database uniclust30_2018_08 would be used. It can be downloaded from [this link](https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz).

    Once finished downloading, rename it according to the `config.yaml` (i.e. uniclust30_2018_08).

### 4. Implementation of trRosetta model:

1) 
    According to the introduction of [trRosetta](https://github.com/gjoni/trRosetta), you need to download the trRosetta [pretrain model](https://files.ipd.uw.edu/pub/trRosetta/model2019_07.tar.bz2).

2) 
    We have made a backup of the trRosetta pre-trained model in [huggingface repo](https://huggingface.co/HongHongStand/TP_LMMSG/tree/main) for a quick implementation.


### 5. Configuration of the dependencies of the language models:

1) 
    SeqVec pretrain model address:[SeqVec download](https://rostlab.org/~deepppi/seqvec.zip). For detailed introduction: [SeqVec model](https://github.com/mheinzinger/SeqVec).

    We have also made a backup of the SeqVec pre-trained model in [huggingface repo](https://huggingface.co/HongHongStand/TP_LMMSG/tree/main).


2) 
    ProtTrans prerequisites: [Transformers installations](https://huggingface.co/docs/transformers/installation). For detailed introduction: [ProtTrans model](https://github.com/agemagician/ProtTrans).


### 6. Model and database path configuration.

Now we can organize each module to the repository as follows:

```
  .
    ├── data
    │   │── XU_train
    │   │   │── positive
    │   │   └── negative
    │   │── XU_test
    │   │   │── positive
    │   │   └── negative
    │   └── other datasets ...
    │
    ├── model_files (for ProtTrans models)
    │
    ├── node_embed
    │   │── seqvec_pretrain_model
    │   │   │── options.json
    │   │   └──weights.hdf5
    │   │     
    │   │── node_embed/LM_embedding.py
    │   └── other python files 
    │
    ├── utils
    │   │── psiblast
    │   │   │── nrdb90
    │   │   │   ├── nrdb90.phr
    │   │   │   ├── nrdb90.pin
    │   │   │   ├── nrdb90.pog
    │   │   │   └── ...
    │   │   │── blosum62
    │   │   └── blosum62.json
    │   │
    │   │── trRosetta
    │   │   │── model2019_07
    │   │   │   ├── model.xaa.data-00000-of-00001
    │   │   │   ├── model.xaa.index
    │   │   │   ├── model.xaa.meta
    │   │   │   └── ...
    │   │   │── predict_many.py
    │   │   └── utils.py
    │   │     
    │   │── hhblits (optional)
    │   │   └── uniclust30_2018_08
    │   │       ├── uniclust30_2018_08_a3m_db
    │   │       ├── uniclust30_2018_08_a3m_db.index
    │   │       ├── uniclust30_2018_08_a3m_db.ffdata
    │   │       └── ...
    │   │     
    │   └── other python files 
    │   
    ├── saved_models (TP_LMMSG model check points)
    │ 
    └── Other files ...
```

At this point, we have completed all environment configuration and software installation. If user want to modify the tools, they could check `config.yaml` and python scripts in `node_embed\` and `utils\`.

## Train and Test

### Overview

To run the entire preprocessing, training, and testing pipeline, users can utilize the Bash script: `run.sh`. 

Due to the utilization of two environments in the entire workflow, please note the following steps before running the script:

Check the path to Conda by using the command: 
```
conda info | grep -i 'base environment'. 
```
Modify the following command in the `run.sh` file via `vim run.sh`:
```
source /home/username/anaconda3/etc/profile.d/conda.sh 
```
to match the correct path of the conda.sh file in your system.

Then use the following commands:
```
cd TP_LMMSG
bash run.sh
```

### Train on different datasets

To conduct training on different datasets, users can achieve this by modifying the paths in the `run.sh` script.

If you wish to make modifications to the training and testing code, you can directly edit `train.py` and `test.py`. You can either modify the `args` or use the command line for parameter tuning:

```
python train.py -use_hhm False -use_lm uni bfd brt -e 5 -d 37 -top_k 10
```
Note: --topk should be modified according to the shortest sequence in the dataset.
```
python test.py -use_hhm False -use_lm uni bfd brt -d 27
```
If you wish to include the hhblits, please modify the `run.sh` script: `generate_features_no_hmm.py` -> `generate_features_hmm.py`. Then conduct training and testing as normal.

