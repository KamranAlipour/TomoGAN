#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import timeit
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt


# In[2]:


args = {}
args['gpus'] = "0"
args['lmse'] =0.5
args['lperc'] =2.0
args['ladv'] =20
args['lunet'] =3
args['depth'] =3
args['itg'] =1
args['itd'] =2


# In[3]:


os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus'] #args.gpus
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR


# In[4]:


all_weights = [
               "train_results_samples_2_3_noise10/test-last-model.h5",
               "train_results_samples_2_3_noise15/test_s2_s3_n15-last-model.h5",
               "train_results_samples_2_3_noise20/test_s2_s3_n20-last-model.h5",
               "train_results_samples_2_3_noise25/test_s2_s3_n25-last-model.h5",
               "train_results_n15_scratch/test-last-model.h5",
               "train_results_n20_scratch/test-last-model.h5",
               "train_results_n25_scratch/test-last-model.h5",
               "train_cropped_results_n15_scratch/test-last-model.h5",
               "train_cropped_results_n20_scratch/test-last-model.h5",
               "train_cropped_results_n25_scratch/test-last-model.h5"
              ]
weight_labels = [ "F10",
                  "F10_15",
                  "F10_15_20",
                  "F10_15_20_25",
                  "F15",
                  "F20",
                  "F25",
                  "F15_C",
                  "F20_C",
                  "F25_C"]


# In[ ]:


frame_count = 279 # number of frames per case
samples = ['s1','s2','s3']
p_noisy = ['10','15','20','25'] # # 0% is GT (clean) and the rest are noisy inputs

#denoise_mse = {} 
#denoise_mse_avg = {}
#noisy_mse = {} 
#noisy_mse_avg = {}
#gt_snr = {}
#denoise_snr = {}
#noisy_snr = {} 
mse_vals = {}
filter_vals = {}
model_labels = {}

for s in samples:
    mse_vals[s] = []
    filter_vals[s] = []
    model_labels[s] = []

    fig_mse = plt.figure()
    fig_mse = plt.figure(figsize=(10, 15))
    plt.title('MSE results on Sample '+s)
    """
    for n,p in enumerate(p_noisy):
        noisy_mse = []
        for i in range(frame_count):
            gt_file = 'dataset/fullframe/frames_1024/'+s+'_0_'+str(i+1).zfill(3)+'_1024x1024.jpg'
            file_name = 'dataset/fullframe/frames_1024/'+s+'_'+p+'_'+str(i+1).zfill(3)+'_1024x1024.jpg'
            gt_im = imageio.imread(gt_file)
            np_gt = np.array(gt_im)
            np_gt = np_gt[:,:,0]
            np_gt = (np_gt - np.min(np_gt)) / (np.max(np_gt) - np.min(np_gt))
            im = imageio.imread(file_name)
            np_im = np.array(im)
            np_im = np_im[:,:,0]  
            np_im = (np_im - np.min(np_im)) / (np.max(np_im) - np.min(np_im))
            n_mse = ((np_gt - np_im)**2).mean(axis=None)
            noisy_mse.append(n_mse)
        mse_vals[s].append(np.mean(noisy_mse))
        filter_vals[s].append(p)
        model_labels[s].append('F'+p)
    """
    for nw,wght in enumerate(all_weights):
        #print(nw)
        #args['weights'] =wght
        #args['resfolder'] = wght.split('/')[0]
   
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.99
        sess = tf.Session(config = config)
        tf.keras.backend.set_session(sess)
        mb_size = 16
        img_size = 256
        if wght.startswith('train_cropped_results'):
            in_depth = 1 #args.depth
        else:
            in_depth = args['depth'] #args.depth
        disc_iters, gene_iters = args['itd'], args['itg'] #args.itd, args.itg
        lambda_mse, lambda_adv, lambda_perc = args['lmse'], args['ladv'], args['lperc'] #args.lmse, args.ladv, args.lperc
        #res_dir = 'graphs/'+ args['resfolder'] #args.resfolder
        generator = make_generator_model(input_shape=(None, None, in_depth), nlayers=args['lunet']) #nlayers=args.lunet ) 
        discriminator = make_discriminator_model(input_shape=(img_size, img_size, 1))
        feature_extractor_vgg = tf.keras.applications.VGG19(weights='vgg19_weights_notop.h5', include_top=False)
        generator.load_weights(wght)

        #line_styles = ['-','--','-.',':','o']
        #labels = list(map(lambda x : 'Filter '+x+'%', p_noisy))

        for n,p in enumerate(p_noisy):
            denoise_mse = []
            start = timeit.default_timer()
            for i in range(frame_count):
                gt_file = 'dataset/fullframe/frames_1024/'+s+'_0_'+str(i+1).zfill(3)+'_1024x1024.jpg'
                file_name = 'dataset/fullframe/frames_1024/'+s+'_'+p+'_'+str(i+1).zfill(3)+'_1024x1024.jpg'
                gt_im = imageio.imread(gt_file)
                np_gt = np.array(gt_im)
                np_gt = np_gt[:,:,0]
                np_gt = (np_gt - np.min(np_gt)) / (np.max(np_gt) - np.min(np_gt))
                im = imageio.imread(file_name)
                np_im = np.array(im)
                if wght.startswith('train_cropped_results'):
                    np_im = np_im[:,:,0]
                    np_im = np.expand_dims(np_im,axis = 0)
                    np_im = np.expand_dims(np_im,axis = 3)
                    #print(np_im.shape)
                else:
                    np_im = np.expand_dims(np_im,axis = 0)
                pred_img = generator.predict(np_im)
                denoised = pred_img[0,:,:,0]
                denoised = (denoised - np.min(denoised)) / (np.max(denoised) - np.min(denoised))
                d_mse = ((np_gt - denoised)**2).mean(axis=None)
                denoise_mse.append(d_mse)
            mse_vals[s].append(np.mean(denoise_mse))
            filter_vals[s].append(p)
            model_labels[s].append('D_'+weight_labels[nw])
            #plt.plot(noisy_mse[s][p],'r'+line_styles[n],label='Filter '+p+'%')
            #plt.plot(denoise_mse[s][p],'b'+line_styles[n],label='Filter '+p+'% Denoised')
            stop = timeit.default_timer()
            print('Sample '+s+' Filter '+p+' Time '+str(stop - start))
    #print('plotting ..')
    plt.scatter(filter_vals[s],mse_vals[s])
    for i, txt in enumerate(model_labels[s]):
        plt.annotate(txt, (filter_vals[s][i], mse_vals[s][i]))
    plt.savefig(s+'_performances.png', dpi=600)
    #plt.legend()


# In[ ]:


mse_vals


# In[ ]:




