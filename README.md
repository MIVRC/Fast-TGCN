# A Fine-grained Orthodontics Segmentation Model for 3D Intraoral Scan Data

## Prequisites
* python 3.7.4
* pytorch 1.4.0
* numpy 1.19.0
* plyfile 0.7.1

## Introduction
This work is the pytorch implementation of **Fast-TGCN**, which has been published in Computers in Biology and Medicine (https://www.sciencedirect.com/science/article/abs/pii/S0010482523012866)
#
## Dataset
The 3D-IOSSeg dataset we proposed can be obtained at the following link:
https://reurl.cc/0vjLXY

## Usage
To train the Fast-TGCN, please put the trainning data and testing data into data/train and data/test, respectively. Then, you can start to train a Fast-TGCN model by following command.

```shell
python train.py
```
#
## Citation
If you find our work useful in your research, please cite:
* Li, Juncheng, et al. "A fine-grained orthodontics segmentation model for 3D intraoral scan data." Computers in Biology and Medicine 168 (2024): 107821.
