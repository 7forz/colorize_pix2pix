
# Colorize-pix2pix

This is a mini project for a machine learning course. To understand, reconstruct and simplify the [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) project for colorization. Still work in progress..

# Requirements
- pytorch
- torchvision
- pillow
- skimage

# Usage

### training
`python3 train.py --dataroot ./datasets/train --gpu`

### continue training
`python3 train.py --dataroot ./datasets/train --epoch_start 30 --gpu`

### testing
`python3 test.py --dataroot ./datasets/test`

### help
`python3 train.py -h`

`python3 train.py -h`