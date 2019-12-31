#! /usr/bin/python3 
import tensorflow as tf 
tf.enable_eager_execution()
import numpy as np 
from util import save2img
import imageio
import sys, os, time, argparse, shutil, scipy, h5py, glob
from data_processor import bkgdGen, gen_train_batch_bg, get1batch4test

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt

import math
import csv
import pdb
import cv2

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-resfolder', type=str, default="all_plots", help='place within videos folder where the files will be stored')
parser.add_argument('-lmse', type=float, default=0.5, help='lambda mse')
parser.add_argument('-lperc', type=float, default=2.0, help='lambda perceptual')
parser.add_argument('-ladv', type=float, default=20, help='lambda adv')
parser.add_argument('-lunet', type=int, default=3, help='Unet layers')
parser.add_argument('-depth', type=int, default=1, help='input depth')
parser.add_argument('-itg', type=int, default=1, help='iterations for G')
parser.add_argument('-itd', type=int, default=2, help='iterations for D')
parser.add_argument('-inputData', type=str, default='dataset/fullframe/frames_1024/', help='directory of input frames')
parser.add_argument('-inputdim', type=int, default=512, help='used to define the size of images when computing the SSIM and PSNR values')

args, unparsed = parser.parse_known_args()

mb_size = 16
img_size = 256
in_depth = args.depth
disc_iters, gene_iters = args.itd, args.itg
lambda_mse, lambda_adv, lambda_perc = args.lmse, args.ladv, args.lperc

samples = ['s1','s2','s3']
p_noisy = ['10','15','20','25'] # # 0% is GT (clean) and the rest are noisy inputs
frame_count = 279 # number of frames per case

counter = 0
for si, s in enumerate(samples):
    for p in p_noisy:
        for i in range(frame_count): #frame_count):
            counter = counter + 1
            cstr = str(counter).zfill(4)
            gt_file_name = args.inputData+s+'_0_'+str(i+1).zfill(3)+'_1024x1024.jpg'
            file_name = args.inputData+s+'_'+p+'_'+str(i+1).zfill(3)+'_1024x1024.jpg'
            gt_im = imageio.imread(gt_file_name)
            im = imageio.imread(file_name)
            new_size = args.inputdim
            np_gt_im = np.array(gt_im)[:,:,0]
            np_im = np.expand_dims(np.array(im)[:,:,0],axis=2)
            np_im = np.expand_dims(np_im,axis = 0)
            np_im = np_im[0,:,:,0]
            np_im = cv2.resize(np_im,(new_size,new_size)) # doing this resize to match the ground truth results based with other models
            np_gt_im = cv2.resize(np_gt_im,(new_size,new_size)) # doing this resize to match the ground truth results based with other models
            np_gt_im = (np_gt_im - np.min(np_gt_im)) / (np.max(np_gt_im) - np.min(np_gt_im))
            np_im = (np_im - np.min(np_im)) / (np.max(np_im) - np.min(np_im))
            #print (gt_file_name.replace('1024x1024.jpg','target.npy'), file_name.replace('1024x1024.jpg','input.npy'),np_im.shape, np_gt_im.shape, np_im.min(), np_im.max(), np_gt_im.min(), np_gt_im.max())
            target_file = gt_file_name.replace(args.inputData,'dataset/fullframe/frames_512/'+cstr+'_').replace('1024x1024.jpg','target.npy')
            input_file = file_name.replace(args.inputData,'dataset/fullframe/frames_512/'+cstr+'_').replace('1024x1024.jpg','input.npy')
            print(target_file,input_file)
            np.save(target_file,np_gt_im)
            np.save(input_file,np_im)
            
