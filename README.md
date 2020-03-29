# Introduction
This repository is for X-Linear Attention Networks for Image Captioning (CVPR 2020).

Please cite with the following BibTeX:

```
@inproceedings{cornia2020m2,
  title={{X-Linear Attention Networks for Image Captioning}},
  author={Yingwei Pan, Ting Yao, Yehao Li, and Tao Mei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Requirements
* Python 3
* CUDA 10
* numpy
* tqdm
* easydict
* [PyTorch](http://pytorch.org/) (>1.0)
* [torchvision](http://pytorch.org/)
* [pycocotools](https://github.com/cocodataset/cocoapi)

## Data preparation
1. Download the [bottom up features](https://github.com/peteanderson80/bottom-up-attention) and convert them to npz files
```
python3 tools/create_feats.py --infeats bottom_up_tsv --outfolder ./mscoco/feature/up_down_10_100
```

2. Download the annotations files into the mscoco folder. More details about data preparation can be referred to [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)

3. The X-LAN transformer models and results can be downloaded here.

## Training
### Train X-LAN model
```
bash experiments/xlan/train.sh
```

### Train X-LAN model using self critical
```
bash experiments/xlan_rl/train.sh
```

### Train X-LAN transformer model
```
bash experiments/xlan_transformer/train.sh
```

### Train X-LAN transformer model using self critical
```
bash experiments/xlan_transformer_rl/train.sh
```

## Evaluation
```
python3 main_test.py --folder experiments/model_folder --resume model_epoch
```

## Acknowledgements
Thanks the contribution of [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) and awesome PyTorch team.



