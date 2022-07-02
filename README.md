# TEMP: Type-aware Embeddings for Multi-Hop Reasoning over Knowledge Graphs
#### This repo provides the source code & data of our paper: [TEMP: Type-aware Embeddings for Multi-Hop Reasoning over Knowledge Graphs (IJCAI 2022)](https://arxiv.org/pdf/2205.00782.pdf).
## Dependencies
* conda create -n temp python=3.7 -y
* PyTorch 1.8.1
* tensorboardX 2.5.1
* numpy 1.21.6
## Running the code
### Dataset
* Download the datasets from [Here](https://drive.google.com/drive/folders/15ZJo6zuoj0S3Sx_8nz7TKr3Tq7Ku8JMR?usp=sharing).
* Create the root directory ./data and put the dataset in.
* It should be noted that we only provide the data provided by the BetaE paper (the corresponding dataset in Table 7 of the paper). For the dataset corresponding to Q2B, you can download it from [here](http://snap.stanford.edu/betae/KG_data.zip).
* You need to move id2type.pkl, type2id.pkl entity_type.npy and relation_type.npy in the corresponding BetaE's dataset to the corresponding Q2B's dataset.
