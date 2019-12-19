pip install imageio-ffmpeg
python3 generate_plots.py -gpu 0 -resfolder all_plots -weights train_shuffle_cropped_mse50_100k_results/test-last-model.h5
python3 generate_plots.py -gpu 0 -resfolder all_plots -weights train_shuffle_cropped_mse75_100k_results/test-last-model.h5
python3 generate_plots.py -gpu 0 -resfolder all_plots -weights train_shuffle_cropped_mse50_results/test-last-model.h5
python3 generate_plots.py -gpu 0 -resfolder all_plots -weights train_cropped_results_n15_scratch/test-last-model.h5
