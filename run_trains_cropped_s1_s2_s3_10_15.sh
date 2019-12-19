python3 main-gan.py -gpu 0 -resfolder train_cropped_s1_s2_s3_n10 -expName test -xtrain dataset/cropped/h5/noisy4train_n10_s1_s2_s3.h5 \
	                                                                       -ytrain dataset/cropped/h5/clean4train_s1_s2_s3.h5 \
                                                                               -xtest dataset/cropped/h5/noisy4test_n10_s1_s2_s3.h5 \
                                                                               -ytest dataset/cropped/h5/clean4test_s1_s2_s3.h5 >> train_cropped_s1_s2_s3_n10_log.txt 2>> train_cropped_s1_s2_s3_n10_error.txt
cp -r train_cropped_s1_s2_s3_n10 train_cropped_s1_s2_s3_n10_backup
python3 main-gan.py -gpu 0 -resfolder train_cropped_s1_s2_s3_n10_n15 -expName test -xtrain dataset/cropped/h5/noisy4train_n15_s1_s2_s3.h5 \
                                                                                   -ytrain dataset/cropped/h5/clean4train_s1_s2_s3.h5 \
										   -xtest dataset/cropped/h5/noisy4test_n15_s1_s2_s3.h5 \
										   -ytest dataset/cropped/h5/clean4test_s1_s2_s3.h5 \
										   -pretrain train_cropped_s1_s2_s3_n10/test-last-model.h5 >> train_cropped_s1_s2_s3_n10_n15_log.txt 2>>  train_cropped_s1_s2_s3_n10_n15_error.txt
