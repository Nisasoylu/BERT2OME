# BERT2OME
Prepared by Necla Nisa Soylu

## BERT2OME: Prediction of 2′-O-Methylation Modifications From RNA Sequence by Transformer Architecture Based on BERT

Our proposed model BERT2OME was published in IEEE/ACM Transactions on Computational Biology and Bioinformatics, focuses on RNA 2’-O-methylation modification site prediction across multiple species by utilizing the Bidirectional Encoder Representations from Transformers (BERT) model. Our model converts RNA sequences into vector embeddings, capturing both syntactic and semantic information from massive language corpora. By considering a combination of BERT embeddings and a two-dimensional Convolutional Neural Network (CNN), BERT2OME outperforms traditional machine learning approaches and shallow neural networks in accurately predicting RNA modification sites. Through extensive experimentation and evaluation of datasets from various species including Homo sapiens, Saccharomyces cerevisiae, and Mus musculus, BERT2OME demonstrates robust performance and cross-species predictability. Furthermore, this study utilizes STREME to identify consensus sequence motifs associated with 2’-O-methylation modifications, on potential similarities between different types of RNA modifications. 
If you want 

4 different datasets are used for detecting 2'-O-methylation sites in the given RNA sequences.

- RMBase_800.xlsx (named as "Human 2 Dataset" in our paper) 
- H. sapiens Dataset (named as "Human 1 Dataset" in our paper)
- S. cerevisiae Dataset
- M. musculus Dataset

Last 3 datasets can be downloaded from the following website: http://lab.malab.cn/~acy/PTM_data/RNADataset.html

We used one of the well-known transformer base model BERT, for converting given RNA sequences into vector embeddings format.

- **VectorEmbeddingCreation_BERT.ipynb** file is used for converting RMBase (named as Human 1 Dataset in the paper), H. sapiens (named as Human 2 dataset in the paper), M. musculus and S. cerevisiae datasets into vector embedding formats by using BERT.

After the previous conversion, following files are generated:

- BERTHUMAN1EMBEDDINGSX.npy
- BERTHUMAN1EMBEDDINGSY.npy

- BERTHUMAN2EMBEDDINGSX.npy
- BERTHUMAN2EMBEDDINGSY.npy

- BERTMOUSEEMBEDDINGSX.npy
- BERTMOUSEEMBEDDINGSY.npy

- BERTYEASTEMBEDDINGSX.npy
- BERTYEASTEMBEDDINGSY.npy

Random Forest and XGBoost models are fed with these vector embeddings and compared with baseline models (training models with one-hot formatted RNA sequences). Then we implemented BERT+1D CNN model for different species. The similarity of the most similar sequences were about 41%. In order to minimize the risk of model overfitting, we have removed the sequences with more than 30% similarity. Last version of the baseline models and the proposed deep learning method (BERT2OME) can be found in the following part:

- **BERT_Models.ipynb**

Whole representation for the base methods (left part of the diagram) and the novel part (right part of the diagram). The workflow of the proposed method BERT2OME. 4 different datasets with 3 different species were used.
Vector embeddings were created using BERT model, then classification method 2D CNN was implemented to identify 2’-O-methylation sites in the given RNA sequences: 

![image](https://github.com/Nisasoylu/BERT2OME/assets/29046464/16d36a9b-c0f1-4dc9-8893-d626fa6bba2a)


For citation:
@ARTICLE{10018863,
  author={Soylu, Necla Nisa and Sefer, Emre},
  journal={IEEE/ACM Transactions on Computational Biology and Bioinformatics}, 
  title={BERT2OME: Prediction of 2′-O-Methylation Modifications From RNA Sequence by Transformer Architecture Based on BERT}, 
  year={2023},
  volume={20},
  number={3},
  pages={2177-2189},
  keywords={RNA;Bit error rate;Task analysis;Predictive models;Biological system modeling;Convolutional neural networks;Transformers;2′-O-methylation;RNA;BERT;convolutional neural network;transformers},
  doi={10.1109/TCBB.2023.3237769}}

