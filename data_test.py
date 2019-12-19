import numpy as np 
import h5py, threading
import queue as Queue
import h5py, glob
from util import scale2uint8

x_fn = 'dataset/cropped/npy_frames_split_1024_train/input_*'
y_fn = 'dataset/cropped/npy_frames_split_1024_train/target_*'
mb_size = 16
in_depth = 1
img_size = 256

x_list = np.array(glob.glob(x_fn))
y_list = np.array(glob.glob(y_fn))
sample_img = np.load(x_list[0])
frame_size = sample_img.shape[0]
data_size = x_list.size
while True:
    print('data size {}'.format(data_size))
    print('frame size {}'.format(frame_size))
    idx = np.random.randint(0, data_size, mb_size)
    crop_idx = np.random.randint(0, frame_size-img_size)
    print(x_list[idx])
    batch_X = np.expand_dims([np.load(x_list[s_idx])[:,:,0] for s_idx in idx], 3)
    batch_X = batch_X[:, crop_idx:(crop_idx+img_size), crop_idx:(crop_idx+img_size), :]
    batch_Y = np.expand_dims([np.load(y_list[s_idx])[:,:,0] for s_idx in idx], 3)
    batch_Y = batch_Y[:, crop_idx:(crop_idx+img_size), crop_idx:(crop_idx+img_size), :]
    print('BATCH X: ',batch_X.shape)
    print('BATCH Y: ',batch_Y.shape)
    #yield batch_X, batch_Y

