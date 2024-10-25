# 🚀 DUET: A Tuning-Free Device-Cloud Collaborative Parameters Generation Framework for Efficient Device Model Generalization (WWW 2023)

[![Static Badge](https://img.shields.io/badge/DOI-10.1145%2F3543507.3583451-logo?style=social&logo=acm&labelColor=blue&color=skyblue)](https://dl.acm.org/doi/abs/10.1145/3543507.3583451) [![Static Badge](https://img.shields.io/badge/arXiv-2209.05227-logo?logo=arxiv&labelColor=red&color=peachpuff)](https://arxiv.org/abs/2209.05227) [![Static Badge](https://img.shields.io/badge/Scholar-DUET-logo?logo=Googlescholar&color=blue)](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=DUET%3A+A+Tuning-Free+Device-Cloud+Collaborative+Parameters+Generation+Framework+for+Efficient+Device+Model+Generalization&btnG=) [![Static Badge](https://img.shields.io/badge/Semantic-DUET-logo?logo=semanticscholar&labelcolor=purple&color=purple)](https://www.semanticscholar.org/paper/DUET%3A-A-Tuning-Free-Device-Cloud-Collaborative-for-Lv-Zhang/af11f2f2fc5b9a8d3b8d03aedd2007af7731882c) [![Static Badge](https://img.shields.io/badge/GitHub-DUET-logo?logo=github&labelColor=black&color=lightgray)](https://github.com/HelloZicky/DUET) ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https://api.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Faf11f2f2fc5b9a8d3b8d03aedd2007af7731882c%3Ffields%3DcitationCount&style=social&logo=semanticscholar&labelColor=blue&color=skyblue&cacheSeconds=360)


------
**PyTorch** implementation of [DUET: A Tuning-Free Device-Cloud Collaborative Parameters Generation Framework for Efficient Device Model Generalization](https://arxiv.org/abs/2209.05227) (Zheqi Lv et al., WWW 2023) on **Sequential Recommendation** task based on **DIN, GRU4Rec, SASRec**. 

------
## 📂 MENU

  * Citation
  * Description
    + Abstract
    + Introduction
    + Method
  * Implementation
    + Environment
    + Folder Structure
    + Data Preprocessing
    + Train & Inference

------
## 🌟 Citation

Please cite this repository and paper if you use any of the code/diagrams here, Thanks! 📢📢📢

To cite, please use the following BibTeX entry:

```latex
@inproceedings{lv2023duet,
  title={DUET: A Tuning-Free Device-Cloud Collaborative Parameters Generation Framework for Efficient Device Model Generalization},
  author={Lv, Zheqi and Zhang, Wenqiao and Zhang, Shengyu and Kuang, Kun and Wang, Feng and Wang, Yongwei and Chen, Zhengyu and Shen, Tao and Yang, Hongxia and Ooi, Beng Chin and others},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={3077--3085},
  year={2023}
}
```

```latex
@article{lv2023duetgithub,
  title={DUET: A Tuning-Free Device-Cloud Collaborative Parameters Generation Framework for Efficient Device Model Generalization(Github)},
  author={{Lv}, Z.}
  howpublished = {https://github.com/Zicky/DUET},
  year={2023}
}
```
------
## 🔬 Description

#### **Abstract:** 

> Device Model Generalization (DMG) is a practical yet underinvestigated research topic for on-device machine learning applications. It aims to improve the generalization ability of pre-trained models when deployed on resource-constrained devices, such as improving the performance of pre-trained cloud models on smart mobiles. While quite a lot of works have investigated the data distribution shift across clouds and devices, most of them focus on model fine-tuning on personalized data for individual devices to facilitate DMG. Despite their promising, these approaches require on-device re-training, which is practically infeasible due to the overfitting problem and high time delay when performing gradient calculation on real-time data. In this paper, we argue that the computational cost brought by fine-tuning can be rather unnecessary. We consequently present a novel perspective to improving DMG without increasing computational cost, i.e., device-specific parameter generation which directly maps data distribution to parameters. Specifically, we propose an efficient Device-cloUd collaborative parametErs generaTion framework (DUET). DUET is deployed on a powerful cloud server that only requires the low cost of forwarding propagation and low time delay of data transmission between the device and the cloud. By doing so, DUET can rehearse the devicespecific model weight realizations conditioned on the personalized real-time data for an individual device. Importantly, our DUET elegantly connects the cloud and device as a “duet” collaboration, frees the DMG from fine-tuning, and enables a faster and more accurate DMG paradigm. We conduct an extensive experimental study of DUET on three public datasets, and the experimental results confirm our framework’s effectiveness and generalisability for different DMG tasks.

#### **Introduction**

> <img src="paper_image/introduction.png" alt="image-20230730213509898" style="zoom: 25%;" />

#### **Method**

> <img src="paper_image/method.png" alt="image-20230730213601917" style="zoom:25%;" />

------
## 📚 Implementation
#### Folder Structure

```shell
DUET_Repository
├── data
│   └── {dataset_name}
│       ├── ood_generate_dataset_tiny_10_30u30i
│       │   └── generate_dataset.py
│       └── raw_data_file
└── code
    └── DUET
        ├── loader
        ├── main
        ├── model
        ├── module
        ├── scripts
        ├── util
        └── README.md			
```

#### Environment
```python
pip install -r requirement.txt
```

#### Data Preprocessing
You can choose to download [Preprocessed Data](https://drive.google.com/drive/folders/17lGWmp7IBfgcb_w9d0nbnfgeRpxuL_OC?usp=sharing) or process it yourself.

```shell
cd data/{dataset_name}/ood_generate_dataset_tiny_10_30u30i
```

```python
python generate_dataset.py
```

#### Train & Inference

Code for **parallel training and inference** are aggregated in the repository.

```shell
cd code/DUET/scripts
bash _0_0_train.sh
bash _0_0_movielens_train.sh
```

Training and inference with a specific **generalization method (${type})**, a specific **dataset (${dataset})** and a specific **base model (${model})**

```shell
cd code/DUET/scripts
bash ${type}.sh ${dataset} ${model} ${cuda_num}
```
for example, 
```shell
cd code/DUET/scripts
bash _0_func_base_train.sh amazon_beauty sasrec 0
bash _0_func_finetune_train.sh amazon_beauty sasrec 0
bash _0_func_duet_train.sh amazon_beauty sasrec 0
```