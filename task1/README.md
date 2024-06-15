

# Dependencies

- `PyTorch`
- `numpy`
- `matplotlib`
- `pyyaml`
- `tensorboard`


# Key Features

Based on ResNet-18 model, we compare the following three different methods:

- **Unsupervised Pretrain**. We use [SimCLR](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf) built upon ResNet-18 for unsupervised pretraining on [ImageNet](https://www.image-net.org), and the source codes come from a [pytorch implementation](https://github.com/sthalles/SimCLR). Based on the features from the last linear layer of the pretrained model, we train a linear classifier on CIFAR-100.
- **Supervised Pretrain**. The supervised pretrained ResNet-18 model on [ImageNet](https://www.image-net.org) comes from [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch). Based on the features from the last linear layer of the pretrained model, we train a linear classifier on CIFAR-100.
- **Supervised Model**. We directly train the ResNet-18 model on CIFAR-100.

To run the codes, please follow the following instructions.

# Unsuprevised Pretrained ResNet-18

Download necessary files of [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) dataset into `./data`:
```bash
mkdir -p ./data/imagenet/train
cd ./data/imagenet
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```

Unzip the files
```bash
tar -xvf ILSVRC2012_img_train.tar -C ./train
cd ./train
for d in n*; do tar -xvf $d/*.tar; done
```

# Supervised Pretrained ResNet-18

Specify the path to load the pretrained model
```bash
export TORCH_HOME="checkpoint/SupervisedPretrained"
```

Setup the pretrained models:
```bash
cd supervised_pretrained
python setup.py install
```

