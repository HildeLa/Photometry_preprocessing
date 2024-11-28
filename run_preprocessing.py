import matplotlib.pyplot as plt
from preprocess_functions import preprocess
import numpy as np
import os

# 'provide path of fluorescence.csv file'
#path = '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B2M6/2024_06_07-17_27_34/Fluorescence.csv'
#path = '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B2M6_wakeup/2024_06_07-18_07_38/Fluorescence.csv'
#path = '/Users/hildeteigen/Documents/Photometry_data/spatio-temporal-sens/B2M1_vers1/2024_06_14-17_23_01/Fluorescence.csv'

rootdir = '/Users/hildeteigen/Documents/Photometry_data/spatio-temporal-sens'
#paths = ['/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M5_30mgkg/2024_07_10-18_01_05/Fluorescence.csv',
 #        '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M5_30mgkg_wakeup/2024_07_10-18_27_44/Fluorescence.csv',
  #       '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M6_30mgkg/2024_07_10-12_58_24/Fluorescence.csv',
   #      '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M6_30mgkg_wakeup/2024_07_10-13_26_18/Fluorescence.csv']
paths=[]
for dirpath, subdirs, files in os.walk(rootdir):
    for x in files:
       if x == 'Fluorescence.csv':
            paths.append(dirpath)

#paths = ['/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B2M6/2024_06_07-17_27_34']
#paths = ['/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B2M6_wakeup/2024_06_07-18_07_38']
#paths = ['/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M7_30mgkg/2024_07_04-14_44_38',
# '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M7_30mgkg_wakeup/2024_07_04-15_17_10',
# '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B2M6_30mgkg/2024_07_03-13_28_15v',
# '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B2M6_30mgkg_wakeup/2024_07_03-14_00_56']
'''
paths = ['/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M8_30mgkg/2024_07_09-17_11_52',
         '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M8_30mgkg_wakeup/2024_07_09-17_37_29',
         '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M5_noinject/2024_07_08-15_37_58',
         '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M5_noinject_wakeup/2024_07_08-16_04_05']
'''
#paths = ['/Users/hildeteigen/Documents/Photometry_data/spatio-temporal-sens/B3M2_vers1/2024_07_18-16_13_24',
         #'/Users/hildeteigen/Documents/Photometry_data/spatio-temporal-sens/B3M3_vers1/2024_07_18-17_06_55']

#paths = ['/Users/hildeteigen/Documents/Photometry_data/test_data/B3M5_noinject_wakeup_copy/2024_07_08-16_04_05']

paths =['/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M7_30mgkg_wakeup/2024_07_04-15_17_10',
        '/Users/hildeteigen/Documents/Photometry_data/GRAB_agonist_test/B3M6_30mgkg_wakeup/2024_07_10-13_26_18']

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



#redo_list = ['/B3M1_vers1/2024_07_18-16_41_12', '/B2M5_vers1/2024_07_18-15_46_29', '/B2M1_vers1/2024_06_14-17_23_01']
#paths = [rootdir + element for element in redo_list]

#paths = ['/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_GRAB/B3M6_MMclosed&Regular_day1/2024_08_22-15_16_40']
paths = ['/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_G8m/B3M3_MMclosed&Regular_day1/2024_08_12-18_57_17']
#paths = ['/Volumes/RanczLab/Photometry_recordings/August_Mismatch_Experiment_G8m/B2M4_MMclosed&open_day1/2024_08_07-12_57_09']
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



