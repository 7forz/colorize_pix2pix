# Colorize-pix2pix

![Example](images/example.jpg)

This is a mini project for a machine learning course. To understand, refactor and simplify the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) project for colorization.

# Requirements
- pytorch
- torchvision
- Pillow
- scikit-image

# Usage

### datasets
Place your datasets in datasets/ directory, and pass it to *dataroot* parameter when train/test

### training
`python3 train.py --dataroot ./datasets/train --gpu`

### continue training
`python3 train.py --dataroot ./datasets/train --epoch_start 30 --gpu`

### testing
`python3 test.py --dataroot ./datasets/test --gpu [--load_epoch 10]  # can specify epoch # to load`

### help
`python3 train.py -h`

`python3 train.py -h`