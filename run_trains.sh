python3 main-gan.py -gpu 0 -resfolder train_results_n15_scratch -expName test -xtrain dataset/noisy4train_n15_s2_s3_80.h5 -ytrain dataset/clean4train_s2_s3_80.h5 -xtest dataset/noisy4test_n15_s2_s3_20.h5 -ytest dataset/clean4test_s2_s3_20.h5
python3 main-gan.py -gpu 0 -resfolder train_results_n20_scratch -expName test -xtrain dataset/noisy4train_n20_s2_s3_80.h5 -ytrain dataset/clean4train_s2_s3_80.h5 -xtest dataset/noisy4test_n20_s2_s3_20.h5 -ytest dataset/clean4test_s2_s3_20.h5
#python3 main-gan.py -gpu 0 -resfolder train_results_n25_scratch -expName test -xtrain dataset/noisy4train_n25_s2_s3_80.h5 -ytrain dataset/clean4train_s2_s3_80.h5 -xtest dataset/noisy4test_n25_s2_s3_20.h5 -ytest dataset/clean4test_s2_s3_20.h5