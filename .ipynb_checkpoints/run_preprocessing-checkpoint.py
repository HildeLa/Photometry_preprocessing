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
paths=[]

for dirpath, subdirs, files in os.walk(rootdir):
    for x in files:
        if x == 'Fluorescence.csv':
            if (dirpath.split('/')[-2].split('_')[-2] != 'Training') & (len(dirpath.split('/')[-2].split('_'))<4):#&(dirpath not in cut_paths):
                print(dirpath)
                paths.append(dirpath)


for path in paths:
    sensors = ['G8m']#['G8m'] #[1500]#['g5-HT3']#['G8m'] # provide sensor info, for now it can be: 'G8m', 'g5-HT3', or provide half decay time in ms
    processed = preprocess(path, sensors)
    processed.Info = processed.get_info()
    processed.rawdata, processed.data, processed.data_seconds, processed.signals, processed.save_path = processed.create_basic_470()
    try:
        processed.events = processed.extract_events()
        events = True
    except AttributeError:
        print('No events')
        events = False
    #processed.filtered = processed.low_pass_filt()
    processed.data_detrended, processed.exp_fits = processed.detrend()
    #processed.motion_corr = processed.movement_correct()
    processed.zscored = processed.z_score(motion = False)
    processed.deltaF_F = processed.deltaF_F(motion = False)
    processed.crucial_info = processed.add_crucial_info()
    processed.info_csv = processed.write_info_csv()
    processed.data_csv = processed.write_preprocessed_csv(Events = events, motion_correct = False)#Events = True) #optional: Events = True, motion_correct = True
    plt.close()



