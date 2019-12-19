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

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description='encode sinogram image.')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')
#parser.add_argument('-resFolder', type=str, required=True, help='result folder')
parser.add_argument('-weights', type=str, default="test-last-model.h5", help='.h5 file that carries the trained model weights')
parser.add_argument('-resfolder', type=str, default="test-last-model.h5", help='place within videos folder where the files will be stored')
parser.add_argument('-lmse', type=float, default=0.5, help='lambda mse')
parser.add_argument('-lperc', type=float, default=2.0, help='lambda perceptual')
parser.add_argument('-ladv', type=float, default=20, help='lambda adv')
parser.add_argument('-lunet', type=int, default=3, help='Unet layers')
parser.add_argument('-depth', type=int, default=3, help='input depth')
parser.add_argument('-itg', type=int, default=1, help='iterations for G')
parser.add_argument('-itd', type=int, default=2, help='iterations for D')
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

res_dir = 'videos/'+args.resfolder

generator = make_generator_model(input_shape=(None, None, in_depth), nlayers=args.lunet ) 
discriminator = make_discriminator_model(input_shape=(img_size, img_size, 1))

# input range should be [0, 255]
feature_extractor_vgg = tf.keras.applications.VGG19(\
                        weights='vgg19_weights_notop.h5', \
                        include_top=False)

time_dit_st = time.time()

generator.load_weights(args.weights)
samples = ['s1','s2','s3']
p_noisy = ['10','15','20','25'] # # 0% is GT (clean) and the rest are noisy inputs
frame_count = 279 # number of frames per case
denoise_frames = []
noisy_frames = []
gt_frames = []
  
for s in samples:
    #gt_writer = imageio.get_writer(res_dir+'/'+s+'_gt.mp4', fps=50)
    #for i in range(frame_count):
    #    gt_file_name = 'dataset/frames_1024/'+s+'_0_'+str(i+1).zfill(3)+'_1024x1024.jpg'
    #    gt_im = imageio.imread(gt_file_name)
    #    np_gt_im = np.array(gt_im)
    #    np_gt_im = np_gt_im[:,:,0]
    #    np_gt_im = (np_gt_im - np.min(np_gt_im)) / (np.max(np_gt_im) - np.min(np_gt_im))
    #    gt_writer.append_data(np_gt_im)
    #gt_writer.close()
    for p in p_noisy:
        denoised_writer = imageio.get_writer(res_dir+'/'+s+'_'+p+'_denoised.mp4', fps=50)
        noisy_writer = imageio.get_writer(res_dir+'/'+s+'_'+p+'_noisy.mp4', fps=50)
        for i in range(frame_count):
            file_name = 'dataset/frames_1024/'+s+'_'+p+'_'+str(i+1).zfill(3)+'_1024x1024.jpg'
            im = imageio.imread(file_name)
            np_im = np.array(im)
            np_im = np.expand_dims(np_im,axis = 0)
            pred_img = generator.predict(np_im)
            denoised = pred_img[0,:,:,0]
            #np_gt_im = (np_gt_im - np.min(np_gt_im)) / (np.max(np_gt_im) - np.min(np_gt_im))
            denoised = (denoised - np.min(denoised)) / (np.max(denoised) - np.min(denoised))
            #np_im = np_im[0,:,:,0]
            #np_im = (np_im - np.min(np_im)) / (np.max(np_im) - np.min(np_im))
            #denoise_frames.append(denoised)
            #noisy_frames.append(np_im)
            #gt_frames.append(np_gt_im)
            denoised_writer.append_data(denoised)
            #noisy_writer.append_data(np_im)
            #save2img(denoised, '%s/frames/denoised_%03d.png' % (res_dir,i))
            #save2img(y222[0,:,:,0], '%s/frames/gtruth_%03d.png' % (res_dir,i))
            #save2img(X222[0,:,:,in_depth//2], '%s/frames/noisy_%03d.png' % (res_dir,i))
        denoised_writer.close()
        #noisy_writer.close()
        #imageio.mimsave(res_dir+'/'+s+'_gt.gif', gt_frames)
        #imageio.mimsave(res_dir+'/'+s+'_'+p+'_denoised.gif', denoise_frames)
        #imageio.mimsave(res_dir+'/'+s+'_'+p+'_noisy.gif', noisy_frames)
