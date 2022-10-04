# SiMT: Self-improving Momentum Target

Official PyTorch implementation of "Meta-learning with Self-Improving Momentum Target" (NeurIPS 2022) by 
[Jihoon Tack](https://jihoontack.github.io/),
[Jongjin Park](https://scholar.google.com/citations?user=F9DGEgEAAAAJ&hl=ko),
[Hankook Lee](https://hankook.github.io/),
[Jaeho Lee](https://jaeho-lee.github.io/),
[Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html).

## 1. Dependencies
```
conda create -n simt python=3.8 -y
conda activate simt

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchmeta tensorboardX
```

## 2. Dataset
Download the following datasets and place at `/data` folder
- Regression
  - [Pascal](https://github.com/mingzhang-yin/Meta-learning-without-memorization), [ShapeNet](https://github.com/boschresearch/what-matters-for-meta-learning)
- Classification
  - [mini-ImageNet](https://github.com/renmengye/few-shot-ssl-public/), [tiered-ImageNet](https://github.com/renmengye/few-shot-ssl-public/), [CUB](https://paperswithcode.com/dataset/cub-200-2011), [Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
  - All classification datasets require preprocessing with [torchmeta](https://github.com/tristandeleu/pytorch-meta) library

## 3. Training
### 3.1. Training option
The options for the training method are as follows:
- `<MODE>`: {`maml`, `anil`, `metasgd`, `protonet`}
- `<MODEL>`: {`conv4`, `resnet12`}
- `<DATASET>`: {`shapenet`, `pose`, `miniimagenet`,`tieredimagenet`}, note that `pose` indicates Pascal dataset.
- One can use `--simt` option to train the backbone meta-learning scheme `<MODE>` with SiMT.

### 3.2. Training backbone algorithms
```
python main.py --mode <MODE> --model <MODEL> --dataset <DATASET>
```

### 3.3. Training SiMT
To train SiMT, one should choose the appropriate hyperparameters including momentum coefficient `ETA`, weight hyperparameter `LAM`, and dropout probability `P`.
```
python main.py --simt --mode <MODE> --model <MODEL> --dataset <DATASET> --eta ETA --lam LAM --drop_p P
```

## 4. Evaluation
### 4.1. Evaluation option
The options for the evaluation are as follows:
- `<PATH>`: the path of the pre-trained checkpoints with the best validation accuracy (e.g., `./logs/experiment_name/best.model`).
- `<MODE>`: {`maml`, `anil`, `metasgd`, `protonet`}
- `<MODEL>`: {`conv4`, `resnet12`}
- `<DATASET>`: {`shapenet`, `pose`, `miniimagenet`,`tieredimagenet`, `cub`, `cars`}, note that `pose` indicates Pascal dataset.
- One can use `--simt` option to evaluate with the momentum network.

### 4.2. Evaluating backbone algorithms
```
python eval.py --mode <MODE> --model <MODEL> --dataset <DATASET> --load_path <PATH>
```

### 4.3. Evaluating SiMT
```
python main.py --simt --mode <MODE> --model <MODEL> --dataset <DATASET> --load_path <PATH>
```

## Citation
```
@inproceedings{tack2020csi,
  title={CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances},
  author={Jihoon Tack and Sangwoo Mo and Jongheon Jeong and Jinwoo Shin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## Reference
- [torchmeta](https://github.com/tristandeleu/pytorch-meta)
- [BOIL](https://github.com/HJ-Yoo/BOIL)
- [MetaMix](https://github.com/huaxiuyao/MetaMix)
- [Sparse-MAML](https://github.com/Johswald/learning_where_to_learn)
