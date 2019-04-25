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
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--gpu', action='store_true', help='whether to use gpu')

    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')#scale_width seems have problem?
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size for training')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    
    parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective, [vanilla| lsgan]')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--epoch_start', type=int, default=0, help='staring epoch # to train, if positive, try to load saved models')    
    parser.add_argument('--epoch_fixed_num', default=10, type=int, help='# of epoch at fixed learning rate')
    parser.add_argument('--epoch_dacay_num', default=10, type=int, help='# of epoch to linearly decay learning rate to zero')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

    parser.add_argument('--print_every_iters', default=500, type=int, help='frequency of showing training on console and saving training images')
    parser.add_argument('--save_every_epoch', type=int, default=2, help='frequency of saving checkpoints at the end of epochs')
    args = parser.parse_args()
    args.isTrain = True  # force isTrain to True

    if args.gpu:
        print('='*30, 'GPU MODE', '='*30)
    else:
        print('='*30, 'CPU MODE', '='*30)

    # create the data loader
    dataset = data.create_dataset(args)  # create a dataset
    print('The number of training images = %d' % len(dataset))

    # create folder for intermediate training results, and model dumping
    util.mkdir('train_result')
    util.mkdir('checkpoint')
    with open('./train_result/loss_history.csv', 'w') as f:
        f.write('iter,G_GAN,G_L1,D_real,D_fake\n')
    with open('./checkpoint/train_params.txt', 'w') as f:  # record time and parameters
        f.write('%s\n' % time.ctime())
        f.write('%s\n' % args)

    # create model
    model = ColorizationModel(args)
    model.setup(args)  # print networks; create schedulers

    total_iters = 0  # the total number of training iterations
    epoch_end = args.epoch_fixed_num + args.epoch_dacay_num  # contains 2 parts, fixed and decay lr
    # begin training
    for epoch in range(args.epoch_start, epoch_end):
        print('Starting epoch %d' % epoch)
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0  # the number of training iterations in current epoch

        for data in dataset:
            epoch_iter += args.batch_size
            total_iters += args.batch_size

            model.set_input(data)
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % args.print_every_iters == 0:  # output current training info
                model.compute_visuals()  # compute rgb images
                model.save_current_results(model.get_current_visuals(), epoch)
                print('training images saved, you may have a look')

                losses = model.get_current_losses()
                print('epoch', epoch, ', iter', epoch_iter, ', loss:', losses)
                with open('./train_result/loss_history.csv', 'a') as f:  # save the loss data to file
                    f.write('%s,' % total_iters)
                    f.write('%(G_GAN)s,%(G_L1)s,%(D_real)s,%(D_fake)s\n' % losses)

        if (epoch > 0) and epoch % args.save_every_epoch == 0:  # save results every x epochs
            print('saving the model at the end of epoch %d, total iters: %d' % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, epoch_end, time.time() - epoch_start_time))
        model.update_learning_rate()   # update learning rates at the end of every epoch.
