# Running the experiments
## Dataset used in the paper
### Data Format:
 * temporal graph: `Dynamic.txt`: ```u, i, ts , edge feature```
 * node feature: `Node_Features.txt`: ```u, node features```
### Download 
* Social Data:
  * [Download Link](https://drive.google.com/file/d/1kKOXXTs4nvplnd0wZCWzq9YUL3D7ig1A/view?usp=drive_link)
  * Download to the folder 'social_data/'
* DPPIN：
  * [Download Link](https://drive.google.com/file/d/1YPSmutKRy5tX9dUNSNkVNlqOM1tS8b_G/view?usp=drive_link)
  * Download to the folder 'dppin_data/'
   

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
cd Temp-GFSM
python3 main.py --dir 'social_data/' --batch_size 10 --k_shot 3 --k_query 3 --n_way 3 --num_task 20 --update_step 5 --nhid 32 --update_lr 0.001 --device 1 --labels 
```

```
Arguments in `main.py` need to be specified for different datasets:
* Social Data:
   * --labels 'dblp_ct1_1,dblp_ct1_0,facebook_ct1_1,facebook_ct1_0,tumblr_ct1_1,tumblr_ct1_0,highschool_ct1_1,highschool_ct1_0,infectious_ct1_1,infectious_ct1_0,mit_ct1_1,mit_ct1_0'
   * --logdir 'logs-sd-b1-'
   * --output_file''output-sd-'
   * --total_sample_g {'dblp_ct1_1':755,'dblp_ct1_0':755,'facebook_ct1_1':995,'facebook_ct1_0':995,'highschool_ct1_1':179,'highschool_ct1_0':179, 'infectious_ct1_1':199, 'infectious_ct1_0':199,'mit_ct1_1':79,'mit_ct1_0':79,'tumblr_ct1_1':373, 'tumblr_ct1_0':373}
* DPPIN Data:
   * --labels 'Uetz,Yu,Babu,Breitkreutz,Gavin,Hazbun,Ho,Ito,Krogan_LCMS,Krogan_MALDI,Lambert,Tarassov'
   * --logdir 'logs-ppin-b1-'
   * --output_file''output-ppin-'
   * --total_sample_g 11
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
