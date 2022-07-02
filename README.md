# Type-aware Embeddings for Multi-Hop Reasoning over Knowledge Graphs
#### This repo provides the source code & data of our paper: [Type-aware Embeddings for Multi-Hop Reasoning over Knowledge Graphs (IJCAI 2022)](https://arxiv.org/pdf/2205.00782.pdf).
## Dependencies
* conda create -n temp python=3.7 -y
* PyTorch 1.8.1
* tensorboardX 2.5.1
* numpy 1.21.6
## Running the code
### Dataset
* Download the datasets from [here](https://drive.google.com/drive/folders/15ZJo6zuoj0S3Sx_8nz7TKr3Tq7Ku8JMR?usp=sharing).
* Create the root directory ./data and put the datasets in.
* It should be noted that we only provide the data provided by the BetaE paper (the corresponding dataset in Table 7 of the paper). For the dataset corresponding to Q2B (the corresponding dataset in Table 1 of the paper), you can download it from [here](http://snap.stanford.edu/betae/KG_data.zip).
* You need to move *id2type.pkl*, *type2id.pkl*, *entity_type.npy* and *relation_type.npy* in the corresponding BetaE's dataset to the corresponding Q2B's dataset.
### Models
- [x] [GQE](https://arxiv.org/abs/1806.01445)
- [x] [Query2Box](https://arxiv.org/abs/1806.01445)
- [x] [BetaE](https://arxiv.org/abs/2010.11465)
- [x] [LogicE](https://arxiv.org/pdf/2103.00418.pdf)
* We added our TEMP module to the above four models.
### Training Model
* Take the GQE model in the FB15k-237 dataset as an example:
#### Generalization
```
export DATA_PATH=../data/FB15k-237-betae
export SAVE_PATH=../logs/FB15k-237/gqe_temp
export LOG_PATH=../logs/FB15k-237/gqe_temp.out
export MODEL=temp
export FAITHFUL=no_faithful

export MAX_STEPS=450000
export VALID_STEPS=10000
export SAVE_STEPS=10000
export ENT_TYPE_NEIGHBOR=32
export REL_TYPE_NEIGHBOR=64

CUDA_VISIBLE_DEVICES=0 nohup python -u ../main.py --cuda --do_train --do_valid --do_test \
  --data_path $DATA_PATH --save_path $SAVE_PATH -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps $MAX_STEPS --valid_steps $VALID_STEPS --save_checkpoint_steps $SAVE_STEPS \
  --cpu_num 1 --geo vec --test_batch_size 16 --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --print_on_screen \
  --faithful $FAITHFUL --model_mode $MODEL --neighbor_ent_type_samples $ENT_TYPE_NEIGHBOR --neighbor_rel_type_samples $REL_TYPE_NEIGHBOR \
  > $LOG_PATH 2>&1 &
```
#### Deductive
```
export DATA_PATH=../data/FB15k-237-betae
export SAVE_PATH=../logs/FB15k-237/gqe_faithful_temp
export LOG_PATH=../logs/FB15k-237/gqe_faithful_temp.out
export MODEL=temp
export FAITHFUL=faithful

export MAX_STEPS=450000
export VALID_STEPS=10000
export SAVE_STEPS=10000
export ENT_TYPE_NEIGHBOR=32
export REL_TYPE_NEIGHBOR=64

CUDA_VISIBLE_DEVICES=0 nohup python -u ../main.py --cuda --do_train --do_valid --do_test \
  --data_path $DATA_PATH --save_path $SAVE_PATH -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps $MAX_STEPS --valid_steps $VALID_STEPS --save_checkpoint_steps $SAVE_STEPS \
  --cpu_num 1 --geo vec --test_batch_size 16 --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --print_on_screen \
  --faithful $FAITHFUL --model_mode $MODEL --neighbor_ent_type_samples $ENT_TYPE_NEIGHBOR --neighbor_rel_type_samples $REL_TYPE_NEIGHBOR \
  > $LOG_PATH 2>&1 &
```
* Other running scripts can be seen in ./scripts.
## Citation
If you find this code useful, please consider citing the following paper.
```
@article{DBLP:journals/corr/abs-2205-00782,
  author    = {Zhiwei Hu and
               V{\'{\i}}ctor Guti{\'{e}}rrez{-}Basulto and
               Zhiliang Xiang and
               Xiaoli Li and
               Ru Li and
               Jeff Z. Pan},
  title     = {Type-aware Embeddings for Multi-Hop Reasoning over Knowledge Graphs},
  journal   = {CoRR},
  volume    = {abs/2205.00782},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.00782},
  doi       = {10.48550/arXiv.2205.00782},
  eprinttype = {arXiv},
  eprint    = {2205.00782},
}

## Acknowledgement
We refer to the code of [KGReasoning](https://hub.fastgit.xyz/snap-stanford/KGReasoning). Thanks for their contributions.
