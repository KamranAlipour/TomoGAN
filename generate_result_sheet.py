#! /usr/bin/python3 
import tensorflow as tf 
tf.enable_eager_execution()
import numpy as np 
from util import save2img
import imageio
import sys, os, time, argparse, shutil, scipy, h5py, glob
from models import tomogan_disc as make_discriminator_model  # import a disc model
from models import unet as make_generator_model           # import a generator model
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
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')
#parser.add_argument('-resFolder', type=str, required=True, help='result folder')
parser.add_argument('-weights', type=str, default="test-last-model.h5", help='.h5 file that carries the trained model weights')
parser.add_argument('-resfolder', type=str, default="all_plots", help='place within videos folder where the files will be stored')
parser.add_argument('-lmse', type=float, default=0.5, help='lambda mse')
parser.add_argument('-lperc', type=float, default=2.0, help='lambda perceptual')
parser.add_argument('-ladv', type=float, default=20, help='lambda adv')
parser.add_argument('-lunet', type=int, default=3, help='Unet layers')
parser.add_argument('-depth', type=int, default=1, help='input depth')
parser.add_argument('-itg', type=int, default=1, help='iterations for G')
parser.add_argument('-itd', type=int, default=2, help='iterations for D')
parser.add_argument('-inputData', type=str, default='dataset/fullframe/frames_1024/', help='directory of input frames')
parser.add_argument('-inputdim', type=int, default=1024, help='used to define the size of images when computing the SSIM and PSNR values')
#parser.add_argument('-test', type=str, required=True, help='file name for testing')

args, unparsed = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config = config)
tf.keras.backend.set_session(sess)

mb_size = 16
img_size = 256
in_depth = args.depth
disc_iters, gene_iters = args.itd, args.itg
lambda_mse, lambda_adv, lambda_perc = args.lmse, args.ladv, args.lperc
print(args.weights)
res_dir = os.path.join(args.resfolder,args.weights.split('/')[0])

if not os.path.exists(res_dir):
    os.makedirs(res_dir)


generator = make_generator_model(input_shape=(None, None, in_depth), nlayers=args.lunet ) 
discriminator = make_discriminator_model(input_shape=(img_size, img_size, 1))

# input range should be [0, 255]
feature_extractor_vgg = tf.keras.applications.VGG19(\
                        weights='vgg19_weights_notop.h5', \
                        include_top=False)

time_dit_st = time.time()

generator.load_weights(args.weights)
samples = ['s1','s2','s3']
p_noisy = ['0','10','15','20','25'] # # 0% is GT (clean) and the rest are noisy inputs
frame_count = 279 # number of frames per case
denoise_frames = []
noisy_frames = []
gt_frames = []

labels = []
input_avgs = []
result_avgs = []
filter_legends = []

#data_lbls = ['PSNR','SSIM']
data_lbls = ['PSNR']

for data_lbl in data_lbls:
  labels = []
  input_avgs = []
  result_avgs = []
  filter_legends = []
  for si, s in enumerate(samples):
      for p in p_noisy:
          input_np = []
          result_np = []
          for i in range(frame_count): #frame_count):
                gt_file_name = args.inputData+s+'_0_'+str(i+1).zfill(3)+'_1024x1024.jpg'
                file_name = args.inputData+s+'_'+p+'_'+str(i+1).zfill(3)+'_1024x1024.jpg'
                gt_im = imageio.imread(gt_file_name)
                im = imageio.imread(file_name)
                np_gt_im = np.array(gt_im)[:,:,0]
                np_im = np.expand_dims(np.array(im)[:,:,0],axis=2)
                np_im = np.expand_dims(np_im,axis = 0)
                pred_img = generator.predict(np_im)
                denoised = pred_img[0,:,:,0]
                new_size = args.inputdim
                np_gt_im = cv2.resize(np_gt_im,(new_size,new_size)) # doing this resize to match the ground truth results based with other models
                np_gt_im = (np_gt_im - np.min(np_gt_im)) / (np.max(np_gt_im) - np.min(np_gt_im))
                if np.max(denoised) != np.min(denoised):
                    denoised = (denoised - np.min(denoised)) / (np.max(denoised) - np.min(denoised))
                denoised = cv2.resize(denoised,(new_size,new_size)) # doing this resize to match the ground truth results based with other models
                np_im = np_im[0,:,:,0]
                np_im = cv2.resize(np_im,(new_size,new_size)) # doing this resize to match the ground truth results based with other models
                np_im = (np_im - np.min(np_im)) / (np.max(np_im) - np.min(np_im))
                if data_lbl == 'SSIM':
                    input_np.append(ssim(np_gt_im, np_im,  data_range=np.max(np_im) - np.min(np_im)))
                    if np.max(denoised) != np.min(denoised):
                        result_np.append(ssim(np_gt_im, denoised,  data_range=np.max(denoised) - np.min(denoised)))
                elif data_lbl == 'PSNR':
                    input_diff = np_gt_im - np_im
                    if input_diff.min() == 0.0 and input_diff.max() == 0:
                        input_np.append(-1.0)
                    else:
                        input_np.append(-10 * math.log10(np.mean(input_diff ** 2)))
                    result_diff = np_gt_im - denoised
                    result_np.append(-10 * math.log10(np.mean(result_diff ** 2)))
                    #print(gt_file_name,file_name,'PSNR Input: ',-10 * math.log10(np.mean(input_diff ** 2)),' Model: ',-10 * math.log10(np.mean(result_diff ** 2)))
          sample = s
          filter_lbl = p + '%'
          labels.append(sample)
          filter_legends.append(filter_lbl)
          input_avgs.append(np.mean(input_np))
          result_avgs.append(np.mean(result_np))
  print(input_avgs)        
  print(result_avgs)
  print(filter_legends)
  print(labels)
  csv_dir = os.path.join(res_dir,data_lbl)
  if not os.path.exists(csv_dir):
      os.makedirs(csv_dir)
  print(csv_dir)
  data_file = open(os.path.join(csv_dir,'TomoGAN_'+data_lbl+'_Results_sample'+str(si+1)+'.csv'), mode='w')
  data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  for sample,filter,input_val,model_val in zip(labels,filter_legends,input_avgs,result_avgs):
      data_writer.writerow([sample,filter,input_val,model_val])
