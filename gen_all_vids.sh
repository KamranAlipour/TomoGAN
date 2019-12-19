pip install imageio-ffmpeg
python3 generate_video.py -gpu 0 -resfolder results_n10 --weights model_trained_n10
python3 generate_video.py -gpu 0 -resfolder results_n15 --weights model_trained_n15
python3 generate_video.py -gpu 0 -resfolder results_n20 --weights model_trained_n20
python3 generate_video.py -gpu 0 -resfolder results_n25 --weights model_trained_n25
