# Running the experiments
## Dataset
* [DPPIN](https://github.com/DongqiFu/DPPIN)
  * temporal graph: `Dynamic.txt`: ```u, i, ts , edge feature```
  * node feature: `Node_Features.txt`: ```u, node features```

## Requirements
* Python >= 3.7
* Package requirements
```{bash}
pandas
torch == 1.8.1
tqdm == 4.59.0
numpy == 1.20.1
scikit_learn == 1.0.2
psutil == 5.8.0
jsonlines == 2.0.0
pytorch_geometric == 1.7.0
```

## How to run code
```{bash}
python3 main.py --dir '../social_data/' --batch_size 10 --k_shot 3 --k_query 3 --n_way 3 --num_task 20 --update_step 5 --nhid 32 --update_lr 0.001 --device 1
```


This repository is for the KDD' 2022 paper "Meta-Learned Metrics over Multi-Evolution Temporal Graphs" ([Link](https://dongqifu.github.io/publications/Temp-GFSM.pdf)) .

## Functionality
Temp-GFSM first models temporal graphs for multiple dynamic evolution pattern, then it learns the accurate and adaptive metrics over them via the representation learning techniques.

## Reference
If you use the materials from this repositiory, please refer to our paper.
```
@inproceedings{DBLP:conf/kdd/FuFMTH22,
  author    = {Dongqi Fu and
               Liri Fang and
               Ross Maciejewski and
               Vetle I. Torvik and
               Jingrui He},
  title     = {Meta-Learned Metrics over Multi-Evolution Temporal Graphs},
  booktitle = {{KDD} '22: The 28th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Washington, DC, USA, August 14 - 18, 2022},
  pages     = {367--377},
  publisher = {{ACM}},
  year      = {2022}
}
```
