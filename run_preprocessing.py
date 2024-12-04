import matplotlib.pyplot as plt
from preprocess_functions import preprocess
import numpy as np
import os


cut_paths = ['/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/B3M7_Training_day1/2024_08_13-18_23_03',
            '/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/B3M8_Training_day5/2024_08_18-13_55_14',
            '/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/B3M7_Training_day5/2024_08_18-14_22_52',
            '/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/B3M7_MMclosed&open_day1/2024_08_19-15_39_15',
            ]

rootdir = '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test'


#rootdir = '/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/'
rootdir = '/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_G8m/'

rootdir = '/Volumes/RanczLab/GRAB_agonist_test/'
paths=[]

for dirpath, subdirs, files in os.walk(rootdir):
    for x in files:
        if x == 'Fluorescence.csv':
            #if (dirpath.split('/')[-2].split('_')[-2] != 'Training') & (len(dirpath.split('/')[-2].split('_'))<4):#&(dirpath not in cut_paths):
             #   print(dirpath)
             #   paths.append(dirpath)
            if 'wakeup' in dirpath:
                print(dirpath)
                paths.append(dirpath)
paths = ['/Volumes/RanczLab/GRAB_agonist_test/B3M6_30mgkg_wakeup/2024_07_10-13_26_18']

for path in paths: 
    sensors = {'470':'g5-HT3', '560':'g5-HT3', '410':'g5-HT3'}
    processed = preprocess(path, sensors)
    processed.Info = processed.get_info()
    processed.rawdata, processed.data, processed.data_seconds, processed.signals, processed.save_path = processed.create_basic()
    processed.events = processed.extract_events()
    processed.filtered = processed.low_pass_filt()
    processed.data_detrended, processed.exp_fits = processed.detrend()
    processed.motion_corr = processed.movement_correct()
    processed.zscored = processed.z_score(motion = False)
    processed.deltaF_F = processed.deltaF_F(motion = False)
    processed.crucial_info = processed.add_crucial_info()

    processed.info_csv = processed.write_info_csv()

    processed.data_csv = processed.write_preprocessed_csv(motion_correct = False, Onix_align = False)



