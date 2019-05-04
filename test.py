#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import argparse
import os

import data
import util
from colorize_model import ColorizationModel


if __name__ == '__main__':

    # get the params
    parser = argparse.ArgumentParser(description='test the model')
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--load_epoch', type=int, default=-1, help='specify the epoch # to load, default load the latest poch')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer, must be the same as training')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer, must be the same as training')
    parser.add_argument('--gpu', action='store_true', help='whether to use gpu')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    args = parser.parse_args()

    args.max_dataset_size = float('inf')
    args.batch_size = 1    # test code only supports batch_size = 1
    args.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    args.serial_batches = True  # no shuffle for data loader
    args.verbose = False  # do not print the whole network structures
    args.isTrain = False

    # create the data loader
    dataset = data.create_dataset(args)  # create a dataset
    print('The number of testing images = %d' % len(dataset))

    # create folder for test results
    util.mkdir('test_result')

    # create model
    model = ColorizationModel(args)
    model.setup(args)  # print networks; create schedulers

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()    # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        assert len(img_path) == 1
        img_filename = os.path.basename(img_path[0]).rsplit('.')[0]  # get original file name
        model.save_current_results(visuals, isTrain=False, prefix=img_filename)
        if i % 10 == 0:
            print('processing', i)
