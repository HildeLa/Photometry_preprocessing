import matplotlib.pyplot as plt
from preprocess_functions import preprocess
import numpy as np
import os


'''cut_paths = ['/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/B3M7_Training_day1/2024_08_13-18_23_03',
            '/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/B3M8_Training_day5/2024_08_18-13_55_14',
            '/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/B3M7_Training_day5/2024_08_18-14_22_52',
            '/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/B3M7_MMclosed&open_day1/2024_08_19-15_39_15',
              ]'''





rootdir = '/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/MM_closed-and-regular_day1'

paths=[]

for dirpath, subdirs, files in os.walk(rootdir):
    for x in files:
        if x == 'Fluorescence.csv':
            print(dirpath)
            paths.append(dirpath)

for path in paths: 
    sensors = {'470':'g5-HT3', '560':'g5-HT3', '410':'g5-HT3'} #{'470':'G8m', '560':'G8m', '410':'G8m'} #{'470':'g5-HT3', '560':'g5-HT3', '410':'g5-HT3'}
    processed = preprocess(path, sensors)
    processed.Info = processed.get_info()
    processed.rawdata, processed.data, processed.data_seconds, processed.signals, processed.save_path = processed.create_basic(path_save ='/Volumes/RanczLab/20240730_Mismatch_Experiment/GRAB_MMclosed-and-Regular_220824/')
    processed.events = processed.extract_events()
    processed.filtered = processed.low_pass_filt()
    processed.data_detrended, processed.exp_fits = processed.detrend()
    processed.motion_corr = processed.movement_correct()
    processed.zscored = processed.z_score(motion = False)
    processed.deltaF_F = processed.deltaF_F(motion = False)
    processed.crucial_info = processed.add_crucial_info()

    processed.info_csv = processed.write_info_csv()

    processed.data_csv = processed.write_preprocessed_csv(Onix_align = True)



