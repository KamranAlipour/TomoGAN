pip install imageio-ffmpeg
python3 generate_video.py -gpu 0 -resfolder results_n15_scratch --weights train_results_n15_scratch/test_s2_s3_n25-last-model.h5
python3 generate_video.py -gpu 0 -resfolder results_n20_scratch --weights train_results_n20_scratch/test_s2_s3_n25-last-model.h5
python3 generate_video.py -gpu 0 -resfolder results_n25_scratch --weights train_results_n25_scratch/test_s2_s3_n25-last-model.h5
