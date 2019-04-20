#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import argparse
import time

import data
import util
from colorize_model import ColorizationModel

if __name__ == '__main__':

    # get the params
    parser = argparse.ArgumentParser(description='train the model')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size for training')
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--epoch_start', default=0, type=int, help='the epoch number to start training, if >0, will try to load corresponding saved model')
    parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective, [vanilla| lsgan | wgangp]')
    parser.add_argument('--gpu', action='store_true', help='whether to use gpu')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--epoch_fixed_num', default=50, type=int, help='# of epoch at fixed learning rate')
    parser.add_argument('--epoch_dacay_num', default=50, type=int, help='# of epoch to linearly decay learning rate to zero')
    parser.add_argument('--print_every', default=100, type=int, help='frequency of showing training results on console')
    parser.add_argument('--save_epoch_every', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--load_epoch', type=int, default=0, help='load epoch # to continue training, default not load')    
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    args = parser.parse_args()
    args.isTrain = True  # force isTrain to True

    if args.gpu:
        print('='*30, 'GPU MODE', '='*30)
    else:
        print('='*30, 'CPU MODE', '='*30)

    # create the data loader
    dataset = data.create_dataset(args)  # create a dataset
    print('The number of training images = %d' % len(dataset))

    # create folder for intermediate training results
    util.mkdir('train_result')

    # create model
    model = ColorizationModel(args)
    model.setup(args)  # print networks; create schedulers

    total_iters = 0  # the total number of training iterations
    epoch_end = args.epoch_fixed_num + args.epoch_dacay_num  # contains 2 parts, fixed and decay lr
    # begin training
    for epoch in range(args.epoch_start, epoch_end):
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0  # the number of training iterations in current epoch

        for data in dataset:
            epoch_iter += args.batch_size
            total_iters += args.batch_size

            model.set_input(data)
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % args.print_every == 0:  # output current training info
                model.compute_visuals()  # compute rgb images
                model.save_current_results(model.get_current_visuals(), epoch)
                print('training images saved, you may have a look')

                losses = model.get_current_losses()
                print('epoch', epoch, ', iter', epoch_iter, ', loss:', losses)

        if epoch % args.save_epoch_every == 0:  # save results every x epochs
            print('saving the model at the end of epoch %d, total iter %d' % (epoch, total_iters))
            # model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, epoch_end, time.time() - epoch_start_time))
        model.update_learning_rate()   # update learning rates at the end of every epoch.
