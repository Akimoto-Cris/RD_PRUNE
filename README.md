# Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks

Official Pytorch implementation of our ICCV'23 paper [_Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks_](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Efficient_Joint_Optimization_of_Layer-Adaptive_Weight_Pruning_in_Deep_Neural_ICCV_2023_paper.html) ([Kaixin Xu](https://xuk114.github.io/)\*, Zhe Wang\*, Xue Geng, Jie Lin, Min Wu, Xiaoli Li, Weisi Lin).
- \*Equal contribution


# Installation
1. Clone this repository
```sh
git clone https://github.com/Akimoto-Cris/RD_PRUNE.git
cd RD_PRUNE
```

2. Install NVIDIA-DALI for imagenet experiments:

- Please refer to [official installation guide](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)


# Experiments 

## Dataset Preparation

Set the desired dataset location in the [tools/dataloaders.py](/tools/dataloaders.py) (L9-10):
```python
data_route = {'cifar': '/path/to/cifar',
              'imagenet': '/path/to/imagenet'}
```

## Iterative Pruning using calibration set

- ResNet-32 on CIFAR: 
```sh
python iterate.py --dataset cifar --model resnet32_cifar --pruner rd --worst_case_curve --calib_size 1024
```
- ResNet-50 on ImageNet:
```sh
python iterate.py --dataset imagenet --model resnet50 --pruner rd --worst_case_curve --calib_size 256
```

## Zero-shot Iterative Pruning using random synthetic data
- ResNet-50 on ImageNet:
```sh
python iterate.py --dataset imagenet --model resnet50 --pruner rd --worst_case_curve --calib_size 256 --synth_data
```

## One-shot Iterative Pruning

This can be performed simply by modifying the [tools/modelloaders.py](/tools/modelloaders.py) and amount-per-iteration configuration. E.g. For One-shot pruning of ResNet-50 on Imagenet at 50% sparsity, 
change the L101 of [tools/modelloaders.py](/tools/modelloaders.py) as:
```python
        amount = 0.5
```
then run the following script:
```sh
python iterate.py --dataset imagenet --model resnet50 --pruner rd --worst_case_curve --calib_size 256 --iter_end 1
```

## Others

We've also left the baseline methods in the [tools/pruners.py](/tools/pruners.py), which means you can run baseline methods by setting the flag `--pruner` of the above scripts to the corresponding methods (E.g. `lamp/glob/unif/unifplus/erk`).

# Results 

- Iterative Pruning

| Model  | Dataset | Sparsity (%) | FLOPs Remained (%) | Top-1 | Dense | Top-1 diff | 
|:------:|:-----:|:------------:|:-----:|:-----:|:-----:|:-----:|
| ResNet-32     | CIFAR-10 | 95.5  | 3.59   | 90.83 ± 0.24 | 93.99  | -3.16 |
| VGG-16        | CIFAR-10 | 98.85 | 3.43   | 92.14 ± 0.18 | 91.71  | +0.43 |
| DenseNet-121  | CIFAR-10 | 98.85 | 2.02   | 87.7 ± 0.24  | 91.14  | -3.44 |
| EfficientNet-B0|CIFAR-10 | 98.85 | 4.58   | 85.63 ± 0.31 | 87.95  | -2.32 |
| VGG-16-BN     | ImageNet | 89.3  | 17.71  | 68.88        | 73.37  | -4.49 |
| ResNet-50     | ImageNet | 41    | 53.5   | 75.90        | 76.14  | -0.24 |

- One-shot Pruning

| Model  | Dataset | Sparsity (%) | FLOPs Remained (%) | Top-1 | Dense | Top-1 diff | 
|:------:|:-----:|:------------:|:-----:|:-----:|:-----:|:-----:|
| ResNet-50     | ImageNet | 58    | 34.5   | 75.59        | 76.14  | -0.55 |

- Zero-data Pruning
  
| Model  | Dataset | Sparsity (%) | FLOPs Remained (%) | Top-1 | Dense | Top-1 diff | 
|:------:|:-----:|:------------:|:-----:|:-----:|:-----:|:-----:|
| ResNet-50     | ImageNet | 50    | 42.48   | 75.13        | 76.14  | -1.01 |


# Acknowledgement

This implementation is built on top of [ICLR'21 LAMP](https://github.com/jaeho-lee/layer-adaptive-sparsity).
We thank authors for the awesome repo. 


# Citation

We appreciate it if you would please cite the following paper if you found the implementation useful for your work:
```
@InProceedings{Xu_2023_ICCV,
    author    = {Xu, Kaixin and Wang, Zhe and Geng, Xue and Wu, Min and Li, Xiaoli and Lin, Weisi},
    title     = {Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {17447-17457}
}
```
