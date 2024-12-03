import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress, zscore
from scipy.optimize import curve_fit, minimize
from datetime import datetime, timedelta
import os
import matplotlib as mpl
import time

class preprocess:
    def __init__(self, path, sensors):
        if path[-1]== '/':
            self.path = path
        else:
            self.path = path+'/'
        self.sensors = sensors
        #self.colors = mpl.colormaps['tab10'].colors
        self.colors = ['mediumseagreen', 'indianred', 'mediumpurple', 'steelblue']

    def get_info(self):
        with open(self.path+'Fluorescence.csv', newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)
        Info = str(row1[0]).replace(";", ",")
        Info = Info.replace('true', 'True')
        Info = eval(Info.replace("false", "False"))
        fps = Info['Fps']
        Light_info = Info['Light']
        if Light_info['Led560Enable'] == True:
            red_light = Light_info['Led560Value']
        if Light_info['Led410Enable'] == True:
            blue_light = Light_info['Led410Value']
        if Light_info['Led470Enable'] == True:
            green_light = Light_info['Led470Value']
        Excitation = Info['Excitation']
        mode = Excitation['mode']
        discontious = Excitation['discontinuous']
        interval_time = Excitation['interval_time']
        continous_time = Excitation['continuous_time']
        return(Info)

    def create_basic(self, cutend = False, path_save = None):
        '''
        This is the same as create basic except that it uses Events.csv and Fluorescence-unaligned.csv.
        1) aligns events and all fluorescence to the 470 nm channel in one dataframe
        2) removed first 15 seconds and makes a new TimeStamp column in seconds rather than ms
        3) look at info to see which nm signals were recorded, and makes a df with the data from those
        4) use the path to make a new path to the output data from the preprocessing.
        A folder should already have been made called 'preprocessed'.
        If they don't exist, the mouse ID, date and time, and experiment as indicated from the path is used to make a
        path to the output files. This means that the path should be of the format:
            ***location**/Experiment_name/[four characted mouse ID]_[other-info]/[YYY_MM_DD-hh_mm_ss]
            [four characted mouse ID]: is the name of the recording, specified in the photometry software
            [YYY_MM_DD-hh_mm_ss]: automatically generated folder as a recording is initiated in the photometry software
            Experiment_name: all recording in the same experiment should be in the same folder

        :param: cutend: default set to False, set it to true if you know that you forgot to turn off the equipment before removing fiber
        :return: rawdata, data, data_seconds, signals, save_path
        'signals' is the most important dataframe containing all the data
        'save_path' will indicate where the data should be saved to
        '''
        Info = self.Info
        event_path = self.path + 'Events.csv'  # events with precise timestamps
        fluorescence_path = self.path + 'Fluorescence-unaligned.csv'  # fluorescence with precise timestamps

        #reading the csv files into pandas dataframes
        events = pd.read_csv(event_path)
        fluorescence = pd.read_csv(fluorescence_path)
        # Create separate dataframes for each light
        df_470 = fluorescence[fluorescence['Lights'] == 470][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '470'})
        df_410 = fluorescence[fluorescence['Lights'] == 410][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '410'})
        df_560 = fluorescence[fluorescence['Lights'] == 560][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '560'})

        # Merge the '410' and '560' dataframes with the '470' dataframe based on nearest timestamps
        # All will be shifted to match the 470 nm signal
        df_final = pd.merge_asof(df_470, df_410, on='TimeStamp', direction='backward')
        df_final = pd.merge_asof(df_final, df_560, on='TimeStamp', direction='backward')

        # Fill nan values or handle missing data,
        # first forward filling and then backwards filling in case of nans at the end of the columns
        df_final['410'] = df_final['410'].ffill()
        df_final['560'] = df_final['560'].ffill()
        df_final['410'] = df_final['410'].bfill()
        df_final['560'] = df_final['560'].bfill()

        #rename the 'Name' column in events to 'Event'
        events.rename(columns={'Name': 'Event'}, inplace=True)
        fluorescence = pd.read_csv(fluorescence_path)

        # Create separate dataframes for each light
        df_470 = fluorescence[fluorescence['Lights'] == 470][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '470'})
        df_410 = fluorescence[fluorescence['Lights'] == 410][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '410'})
        df_560 = fluorescence[fluorescence['Lights'] == 560][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '560'})

        # Merge the '410' and '560' dataframes with the '470' dataframe based on nearest timestamps
        # All will be shifted to match the 470 nm signal
        df_final = pd.merge_asof(df_470, df_410, on='TimeStamp', direction='backward')
        df_final = pd.merge_asof(df_final, df_560, on='TimeStamp', direction='backward')

        # Fill nan values or handle missing data,
        # first forward filling and then backwards filling in case of nans at the end of the columns
        df_final['410'] = df_final['410'].ffill()
        df_final['560'] = df_final['560'].ffill()
        df_final['410'] = df_final['410'].bfill()
        df_final['560'] = df_final['560'].bfill()

        #merge events and fluorescence
        #is merged to match the 470 signal timestamps as closely as possible,
        # meaning that the event timestamps may be slightly shifted
        if len(events) < 1:
            rawdata = df_final
            rawdata = rawdata.loc[:, ~rawdata.columns.str.contains('^Unnamed')]
            data = rawdata[rawdata["TimeStamp"] > 30] 
        else:
            rawdata = pd.merge_asof(df_final, events[['TimeStamp', 'Event', 'State']], on='TimeStamp', direction='backward')
            rawdata = rawdata.loc[:,~rawdata.columns.str.contains('^Unnamed')]  # sometimes an Unnamed column has appeared...
            # removing first 15 seconds because of bleaching
            data = rawdata[rawdata["TimeStamp"] > 15000]  # removing first 15 seconds because of bleaching
        if cutend == True:
            data = data.drop(data.tail(300).index) #This can be done if fiber was by mistake removed before 
        data_seconds = pd.DataFrame([second for second in data['TimeStamp'] / 1000], columns=['TimeStamp'])
        signals = pd.DataFrame()
        if Info['Light']['Led470Enable'] == True:
            signals['470'] = data['470']
        if Info['Light']['Led560Enable'] == True:
            signals['560'] = data['560']
        if Info['Light']['Led410Enable'] == True:
            signals['410'] = data['410']

        recording_time = self.path.split('/')[-2][:]  # use same file name as folder
        mousename = self.path.split('/')[-3]#[:4]
        mousename_recordtime = f'{mousename}_{recording_time}'
        print(f'\n\n \033[1m Preprocessing data for {mousename} at {recording_time} ...\033[0m \n')
        experiments = self.path.split('/')[-4][:]  #
        session = self.path.split('/')[-3][5:]
        if '&' in session:
            session = session.replace('&', '-')


        if path_save is not None:
            found_mouse_dir = False  # Flag to track if mousename was found
            for dirpath, subdirs, files in os.walk(path_save):
                # Check if `mousename` is in the last part of the directory path
                if mousename in dirpath.split('/')[-1]:
                    found_mouse_dir = True
                    photometry_dir = os.path.join(dirpath, 'photometry')
                    
                    # Create the 'photometry' directory if it doesn't exist
                    if not os.path.exists(photometry_dir):
                        os.makedirs(photometry_dir)
                        print(f"Directory created: {photometry_dir}")
                    
                    save_path = photometry_dir
                    print("Save path set to:", save_path)
                    break  # Exit loop once the save path is set
        
            # If `mousename` is not found, handle fallback save path creation
            if not found_mouse_dir:
                print("Mousename not found in path_save. Creating a new directory for the save path.")
                
                if not os.path.exists(path_save):
                    os.makedirs(path_save)  # Ensure `path_save` exists
                    print(f"Root path created: {path_save}")
                
                # Create a directory in the root of `path_save`
                new_dir = f"{mousename}_{recording_time}_photometry"
                save_path = os.path.join(path_save, new_dir)
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    print(f"Directory created: {save_path}")
                else:
                    print(f"Directory already exists: {save_path}")
        
        if path_save is None:
            mainfolder = 'Processed'
            # Check if the directory already exists
            if not os.path.exists(f'{mainfolder}/{experiments}'):
                # Create the directory
                os.makedirs(f'{mainfolder}/{experiments}')
                print("Directory created")
            if not os.path.exists(f'{mainfolder}/{experiments}/{session}'):
                # Create the directory
                os.makedirs(f'{mainfolder}/{experiments}/{session}')
                print("Directory created")
            else:
                print(f"Directory {mainfolder}/{experiments}/{session}/ already exists")
        
            if not os.path.exists(f'{mainfolder}/{experiments}/{session}/{mousename_recordtime}'):
                # Create the directory
                os.makedirs(f'{mainfolder}/{experiments}/{session}/{mousename_recordtime}')
                print("Directory created")
            else:
                print(f"{mainfolder}/{experiments}/{session}/{mousename_recordtime} already exists")
        
            save_path = f'{mainfolder}/{experiments}/{session}/{mousename_recordtime}'

    
        self.mousename = mousename
                        
        return(rawdata, data, data_seconds, signals, save_path)
        
 
    def extract_events(self): 
        """
        Assigns a boolean data column for each unique event type indicating 
        whether the event occurred at each time point (True/False). 
        Removes 'Event' and 'State' columns after processing.
        Saves the updated data DataFrame to self.data.
        
        Returns:
            DataFrame with 'Event' and 'State' columns (original values) before removal.
        """
        # Copy the data for processing
        data = self.data.copy()
    
        # List to hold unique event names for reference (optional)
        events = []
    
        # Check if the Event column exists and is not empty
        if 'Event' not in data.columns or data['Event'].isna().all():
            print("There are no recorded events.")
            
        else:
            # Initialize event columns to False
            for event_name in data['Event'].dropna().unique():
                data[f'{event_name}_event'] = False
                events.append(event_name)
            
            # Iterate through unique event names in the Event column
            for event_name in data['Event'].dropna().unique():
                # Identify rows where the Event matches the event_name
                matching_rows = data['Event'] == event_name
                # Set the event-specific column to True for those rows
                data.loc[matching_rows, f'{event_name}_event'] = True
    
        # Extract 'Event' and 'State' columns to return before dropping them
        event_state_df = data[['Event', 'State']]
    
        # Drop 'Event' and 'State' columns from the main data
        data = data.drop(columns=['Event', 'State'], errors='ignore')
    
        # Save the updated data to self.data
        self.data = data
    
        # Return the original 'Event' and 'State' columns
        return event_state_df



    def low_pass_filt(self, plot=False):
        start_time = time.time()
        
        signals = self.signals
        sensors = self.sensors
        filtered = pd.DataFrame(index=self.signals.index)  # Ensure filtered DataFrame has the same index as signals
    
        fps = self.Info['Fps'] / len(sensors)
        
        for signal in signals:
            sensor = sensors[signal]
            print(f'Filtering {signal} with sensor {sensor}')
            
            # Determine cutoff frequency and filter order
            if sensor == 'G8m':
                Wn = 25
            elif sensor == 'g5-HT3':
                Wn = 3
            elif isinstance(sensor, int):
                Wn = 100 / (sensor / 3)
            else:
                Wn = 1000 / (int(input("Enter sensor half decay time: ")) / 3)
            
            try:
                # Design the filter
                b, a = butter(2, Wn, btype='low', fs=fps)
                # Apply the filter
                filtered[f'filtered_{signal}'] = filtfilt(b, a, signals[signal])
                print(f"Filtering of {signal} completed. Time taken:", time.time() - start_time)
            
            except ValueError:
                print(f'Wn is set to {Wn} for the sensor {sensor}, while the frame rate is set to {fps} fps')
                print(f'Digital filter critical frequencies must be 0 < Wn < fs/2 (fs={fps} -> fs/2={fps/2})')
                print(f'The signal {signal} is not filtered and will be returned as-is.')
                # Copy the unfiltered signal to the filtered DataFrame
                filtered[f'filtered_{signal}'] = signals[signal]
        
        return filtered

    
        # Plot the signals
        if plot:
            num_signals = len(signals)
            fig, axs = plt.subplots(num_signals, figsize=(12, 8), sharex=True, sharey=False)
            fig.subplots_adjust(hspace=0.5)
        
            if num_signals == 1:
                axs = [axs]  # Ensure axs is iterable for a single signal
        
            for i, signal in enumerate(signals):
                ax = axs[i]
        
                # Plot raw signal
                ax.plot(self.data_seconds, signals[signal], alpha=0.3, label=f'Raw {signal}')
                # Plot filtered signal
                ax.plot(self.data_seconds, filtered[f'filtered_{signal}'], alpha=1, label=f'Filtered {signal}')
                ax.set_title(f'{signal} Filtered')
                ax.legend(loc='upper right')
                ax.grid()
        
            axs[-1].set_xlabel('Time (seconds)')
            fig.suptitle('Low-pass Filtered Signals')
        
            # Save plot to file
            plt.savefig(self.save_path + f'/low-pass_filtered_{self.mousename}.png', dpi=150)  # Lower DPI to save time
            plt.close()
        
            print(f"Plotting completed. Total time taken: {time.time() - start_time:.2f} seconds.")
        
    



    def detrend(self):
        try:
            traces = self.filtered
        except:
            print('No filtered signal was found')
            traces = self.signals
        data_detrended = pd.DataFrame()
        exp_fits = pd.DataFrame()
        def double_exponential(time, amp_const, amp_fast, amp_slow, tau_slow, tau_multiplier):
            '''
            Based on: https://github.com/ThomasAkam/photometry_preprocessing
            Compute a double exponential function with constant offset.
            Parameters:
            t       : Time vector in seconds.
            const   : Amplitude of the constant offset.
            amp_fast: Amplitude of the fast component.
            amp_slow: Amplitude of the slow component.
            tau_slow: Time constant of slow component in seconds.
            tau_multiplier: Time constant of fast component relative to slow.
            '''
            tau_fast = tau_slow * tau_multiplier
            return amp_const + amp_slow * np.exp(-time / tau_slow) + amp_fast * np.exp(-time / tau_fast)

        # Fit curve to signal.
        ### Find out if initial parameters should be set differently
        for trace in traces:
            max_sig = np.max(traces[trace])
            inital_params = [max_sig / 2, max_sig / 4, max_sig / 4, 3600, 0.1]  # Why 3600, why 4, why 0.1
            bounds = ([0, 0, 0, 1000, 0],
                  [max_sig, max_sig, max_sig, 36000, 1])
            try:
                signal_parms, parm_cov = curve_fit(double_exponential, self.data_seconds['TimeStamp'], traces[trace], p0=inital_params, bounds=bounds, maxfev=1000)
            except RuntimeError:
                print('Could not fit exponential fit for: \n', self.path)
                pass
            signal_expfit = double_exponential(self.data_seconds['TimeStamp'], *signal_parms)
            print(f'Parameters used for detrending {trace}: ', signal_parms)
            signal_detrended = traces[trace].reset_index(drop=True) - signal_expfit.reset_index(drop=True)
            data_detrended[f'detrend{trace[-4:]}'] = signal_detrended
            exp_fits[f'expfit{trace[-4:]}'] = signal_expfit

        #Plotting the detrended data
        fig, axs = plt.subplots(len(data_detrended.columns), figsize = (15, 10), sharex=True)
        color_count = 0
        for column, ax in zip(data_detrended.columns, axs):
            ax.plot(self.data_seconds, data_detrended[column], c=self.colors[color_count], label=column)
            ax.set(ylabel='data detrended')
            color_count += 1
            ax.legend()
        axs[-1].set(xlabel='seconds')
        fig.suptitle(f'detrended data {self.mousename}')
        plt.savefig(self.save_path + f'/Detrended_data_{self.mousename}.png', dpi=300)

        # Plotting the filtered data with the exponential fit
        fig, axs = plt.subplots(len(traces.columns),figsize = (15, 10), sharex=True)
        color_count = 0
        for trace, exp, ax in zip(traces.columns, exp_fits.columns, axs):
            line1 = ax.plot(self.data_seconds, traces[trace], c=self.colors[color_count], label=trace)
            color_count += 1
            ax2 = ax.twinx()
            line2 = ax2.plot(self.data_seconds, exp_fits[exp], c=self.colors[color_count], alpha =0.5, label=exp)
            ax.set(ylabel='fluoresence')
            ax2.set(ylabel='exponential fit')
            lns = line1 + line2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc=0)

        axs[-1].set(xlabel = 'seconds')
        fig.suptitle(f'exponential fit {self.mousename}')
        plt.savefig(self.save_path + f'/exp-fit_{self.mousename}.png', dpi=300)

        return data_detrended, exp_fits

    def movement_correct(self):
        data = self.data_detrended
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x=data['detrend_410'], y=data['detrend_470'])
            print(f'The slope of the linear regression between the main signal and the control is: ', slope)
            fig, ax = plt.subplots(figsize=(15, 10))
            plt.scatter(data['detrend_410'][::5], data['detrend_470'][::5], alpha=0.1, marker='.')
            x = np.array(ax.get_xlim())
            ax.plot(x, intercept + slope * x)
            ax.set_xlabel('410')
            ax.set_ylabel('470')
            ax.set_title(f'410 nm - 470 nm correlation {self.mousename}.')
            xlim = ax.get_xlim()[1]
            ylim = ax.get_ylim()[0]
            #ax.text(xlim-2, ylim+2,'Slope    : {:.3f}'.format(slope))
            #ax.text(xlim-2, ylim+1,'R-squared: {:.3f}'.format(r_value ** 2))
            print('Slope    : {:.3f}'.format(slope))
            print('R-squared: {:.3f}'.format(r_value ** 2))
            plt.rcParams.update({'font.size': 18})
            plt.savefig(self.save_path + f'/motion_correlation_{self.mousename}.png', dpi=300)
            if slope > 0:
                control_corr = intercept + slope * data['detrend_410']
                signal_corrected = data['detrend_470'] - control_corr
                return signal_corrected
            else:
                print('signal could not be motion corrected')
                print(intercept, slope)
                return data['detrend_470']
        except KeyError:
            print('signal could not be motion corrected, the original data is returned')
            print(intercept, slope)
            return data['detrend_470']

    def z_score(self, motion = False):
        '''
        Z-scoring of signal traces
        Gets relative values of signal, equivalent to delt F/ F
        Calculates median signal value and standard deviation
        Gives the signal strenght in terms of standard deviation units
        Does not take into account signal reduction due to bleaching
        '''
        zscored_data = pd.DataFrame()

        if motion == True:
            signal = self.motion_corr
            zscored_data[f'z_470'] = (signal - np.median(signal)) / np.std(signal)
        if motion == False:
            signals = self.data_detrended
            for signal in signals:
                    signal_corrected = signals[signal]
                    zscored_data[f'z_{signal[-3:]}'] = (signal_corrected - np.median(signal_corrected)) / np.std(signal_corrected)

        zscored_data = zscored_data.reset_index(drop = True)

        fig, axs = plt.subplots(len(zscored_data.columns),figsize = (15, 10), sharex=True)
        color_count = 0
        if len(zscored_data.columns) > 1:
            for column, ax in zip(zscored_data.columns, axs):
                ax.plot(self.data_seconds, zscored_data[column], c = self.colors[color_count], label = column)
                ax.set(xlabel='seconds', ylabel='z-scored fluorescence')
                ax.legend()
                color_count += 1
        else:
            axs.plot(self.data_seconds, zscored_data, c=self.colors[2])
            axs.set(xlabel='seconds', ylabel='z-scored fluorescence')
            axs.legend()

        fig.suptitle(f'zscored_data {self.mousename}')
        plt.savefig(self.save_path+f'/zscored_figure_{self.mousename}.png', dpi = 300)

        return zscored_data

    def deltaF_F(self, motion = False):
        '''
        Input:
        - the detrended signal as delta F
        - The decay curve estimate as baseline fluorescence
        Calculates 100 * deltaF / F (ie. % change)
        :returns
        - dF/F (signals relative to baseline)
        '''
        dF_F = pd.DataFrame()
        if motion == False:
            main_data = self.data_detrended
            for signal, fit in zip(main_data, self.exp_fits):
                F = self.exp_fits[fit]
                deltaF = main_data[signal]
                signal_dF_F = 100 * deltaF / F
                dF_F[f'{signal[-3:]}_dfF'] = signal_dF_F
        if motion == True:
            deltaF = self.motion_corr
            F = self.exp_fits['expfit_470']
            signal_dF_F = 100 * deltaF / F
            dF_F['470_dfF'] = signal_dF_F


        fig, axs = plt.subplots(len(dF_F.columns),figsize = (15, 10), sharex=True)
        color_count = 0
        if len(dF_F.columns) >1:
            for column, ax in zip(dF_F.columns, axs):
                ax.plot(self.data_seconds, dF_F[column], c = self.colors[color_count], label = column)
                ax.set(xlabel='seconds', ylabel='Delta F fluorescence')
                ax.legend()
                color_count += 1
        else:
            axs.plot(self.data_seconds, dF_F, c=self.colors[2])
            axs.set(xlabel='seconds', ylabel='Delta F fluorescence')


        fig.suptitle(f'Delta F / F to exponential fit {self.mousename}')
        plt.savefig(self.save_path+f'/deltaf-F_figure_{self.mousename}.png', dpi = 300)
        return dF_F

    def write_info_csv(self):
        '''
        Writes the available info into a csv file that can be read into a dictionary
        # To read it back:
        with open(f'{filename}_info.csv') as csv_file:
            reader = csv.reader(csv_file)
            Info = dict(reader)
        '''
        path = self.save_path
        Info = self.Info
        with open(f'{path}/info.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in Info.items():
                writer.writerow([key, value])

    def add_crucial_info(self, additional = None):
        '''
        :param additional: optional, can take dict where keys are set as column names and
         items will repeat through rows. This must be necessary info for later slicing if data is pooled !!NOT IMPLEMENTED
        :return: a df with mouse ID, timedelta, recording date, and potential additional info
        *IMPORTANT: This must be according to a path where the folder name two levels up from the
         fluorencence.csv file is such that the 4 first letters gives mouse ID
        '''
        # concat dataframes but make sure to add columns specifying the mouse ID, the data, and so on
        # will need to backtrack mouse ID and recording data etc. though file names
        info_columns = pd.DataFrame()
        date_format = '%Y_%m_%d-%H_%M_%S'
        date_obj = datetime.strptime(self.path.split('/')[-2][:], date_format)
        actualtime = [date_obj + timedelta(0, time) for time in self.data_seconds['TimeStamp']]
        info_columns['Time'] = actualtime
        mouseID = self.path.split('/')[-3]#[:4] # if mouse ID annotation is changed to include more or less letters, the number 4 must be changed
        print(f'Please ensure that {mouseID} is the correct mouse ID \n '
              f'If not, changes must be made to either add_crucial_info fucntion or file naming')
        info_columns['mouseID'] = [mouseID for i in range(len(info_columns))]
        print(f'Mouse {mouseID}')
        Area = input('Add the location of fluorescent protein: ')
        info_columns['Area'] = [Area for i in range(len(info_columns))]
        Sex = input('Add the sex of the mouse: ')
        info_columns['Sex'] = [Sex for i in range(len(info_columns))]
        return info_columns



    def write_preprocessed_csv(self, motion_correct=False):
        """
        Writes the processed traces into a CSV file containing:
        - Corrected and z-scored traces
        - Optionally motion-corrected signal
        - If self.events exists, it also writes the events DataFrame to a separate file
    
        :param motion_correct: Default False, set to True if a motion-corrected signal should be added.
        """
        # Prepare the base filename
        filename = self.path.split('/')[-2][:]
    
        # Combine the base data
        final_df = pd.concat([self.data_seconds, self.deltaF_F, self.zscored, self.crucial_info], axis=1)
        final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]
    
        # Handle motion correction
        if motion_correct:
            print('Motion correction added')
            final_df = pd.concat([final_df, self.motion_corr], axis=1)
            final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]
    
        # Save the main fluorescence file
        final_df.to_csv(self.save_path + '/Processed_fluorescence.csv', index=False)
        print('Processed_fluorescence.csv file saved to ', self.save_path)
    
        # Handle events if self.events exists
        if hasattr(self, 'events') and isinstance(self.events, pd.DataFrame):
            print('Events detected and saved.')
            # Save the events DataFrame separately
            self.events.to_csv(self.save_path + '/Events.csv', index=False)

        mpl.pyplot.close()


