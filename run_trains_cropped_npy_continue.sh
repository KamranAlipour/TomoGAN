python3 main-gan.py -gpu 0 -resfolder train_shuffle_cropped_mse50_continued60k_results -expName test -ite 60001 -xtrain dataset/cropped/npy_frames_split_1024_train/input_* \
                                                                                  -ytrain dataset/cropped/npy_frames_split_1024_train/target_* \
										  -xtest dataset/cropped/npy_frames_split_1024_test/input_* \
										  -ytest dataset/cropped/npy_frames_split_1024_test/target_* \
										  -pretrain train_shuffle_cropped_mse50_100k_results/test-last-model.h5

#python3 main-gan.py -gpu 0 -lmse 0.75 -resfolder train_shuffle_cropped_mse75_100k_results -expName test -ite 100001 -xtrain dataset/cropped/npy_frames_split_1024_train/input_* \
#	                                                                                     -ytrain dataset/cropped/npy_frames_split_1024_train/target_* \
#											     -xtest dataset/cropped/npy_frames_split_1024_test/input_* \
#											     -ytest dataset/cropped/npy_frames_split_1024_test/target_*

#python3 main-gan.py -gpu 0 -lunet 5 -resfolder train_shuffle_cropped_mse50_layer5_results -expName test -xtrain dataset/cropped/npy_frames_split_1024_train/input_* \
#	                                                                                         -ytrain dataset/cropped/npy_frames_split_1024_train/target_* \
#												 -xtest dataset/cropped/npy_frames_split_1024_test/input_* \
#												 -ytest dataset/cropped/npy_frames_split_1024_test/target_*

#python3 main-gan.py -gpu 0 -lunet 5 -lmse 0.75 -resfolder train_shuffle_cropped_mse75_layer5_results -expName test -xtrain dataset/cropped/npy_frames_split_1024_train/input_* \
#                                                                                                 -ytrain dataset/cropped/npy_frames_split_1024_train/target_* \
#												  -xtest dataset/cropped/npy_frames_split_1024_test/input_* \
#												  -ytest dataset/cropped/npy_frames_split_1024_test/target_*
