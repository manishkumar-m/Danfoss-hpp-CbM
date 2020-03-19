# -*- coding: utf-8 -*-
"""
@author: manishkumar-m

Objective: Process vibration waveform data
 1. Read waveform data files
 2. Extract features from filenames. E.g. DMY/HMS, pump name, sensor name.
 3. Append features to create waveform dataset
 4. Consolidate waveform file to dataset for given path
 5. Convert waveform to acc
"""

import os
os.chdir('C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\EDA')

# Load modules
import pandas as pd
import datetime
import pickle
from vib_functions_v2 import *

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
#%% **Do put slash at the end of path**
data_path_Pump4_PumpAxial = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Vibration data\\Pump4_PumpAxial\\"
data_path_Pump4_PumpRadial = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Vibration data\\Pump4_PumpRadial\\"
data_path_Pump5_PumpAxial = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Vibration data\\Pump5_PumpAxial\\"
data_path_Pump5_PumpRadial = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Vibration data\\Pump5_PumpRadial\\"
data_path_Pump6_PumpAxial = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Vibration data\\Pump6_PumpAxial\\"
data_path_Pump6_PumpRadial = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Vibration data\\Pump6_PumpRadial\\"

data_path = data_path_Pump4_PumpAxial
#%%
#Check available data files

def fn_get_filenames(path):
    return(os.listdir(data_path))

file_names = fn_get_filenames(data_path)
print("Available file count: ", len(file_names))
print(file_names[0])

#%%
print(file_names[0].split('_'))

"""
# List available features in filename
[0] pump name
[1] sensor name
[2] data type
[3] date
[4] time
[5] rpm
[6] ? ignore

We here need to include 0,1,3,4,5 in dataset
"""

#%%  1. Read waveform data files

# Function to read waveform data from a file a file and return the data frame
file_path = data_path + file_names[0]

def fn_read_wf_data_from_filename(file_path):
    wf_data = pd.read_csv(file_path)
    return(wf_data)

fn_read_wf_data_from_filename(file_path).head()
#%%  2. Extract features from filenames.
## Write function to return pump_name, sensor_name, date, time, rpm
#def fn_get_features_from_filename(filename):
#    features = filename.split('_')
#    return([features[0], features[1], features[3], features[4], features[5]])

#pump_name, sensor_name, date, time, rpm = fn_get_features_from_filename(file_names[0])

def fn_get_features_list_from_filename(filename):
    features = filename.split('_')
    feat_dict = dict()
    feat_dict['pump_name'] = features[0] 
    feat_dict['sensor_name'] = features[1] 
    feat_dict['date'] = features[3] 
    feat_dict['time'] = features[4] 
    feat_dict['rpm'] = features[5] 
    
#    datetime.datetime.strptime('2019-11-24_08-15-58', '%Y-%m-%d_%H-%M-%S')
    feat_dict['datetime'] = datetime.datetime.strptime(features[3] + '_' + features[4], '%Y-%m-%d_%H-%M-%S')
    
    return(feat_dict)

fn_get_features_list_from_filename(file_names[0])
#%%
## Get features from file names as list and convert to dataframe
#wf_features = fn_get_features_list_from_filename(file_names[0])
#pd.DataFrame(wf_features, index=[0])


#%% 3. Append features to data
## Read and process data for given filename
#file_path = data_path + file_names[0]
#
## read data
#wf_data = fn_read_wf_data_from_filename(file_path)
#print(wf_data.head())
#
## get asset features from filename
#wf_features = fn_get_features_list_from_filename(file_names[0])
#
## Create waveform dataset with all features
#wf_data_df = pd.DataFrame({'x': wf_data['x'],
#                           'y': wf_data['y'],
#                           'pump_name': wf_features['pump_name'],
#                           'sensor_name': wf_features['sensor_name'],
#                           'date': wf_features['date'],
#                           'time': wf_features['time'],
#                           'rpm': wf_features['rpm'],
#                           'datetime': wf_features['datetime']
#                           })
#
#print(wf_data_df.head())

#%% 3. Function to append features & create dataframe

def fn_process_wf_file_single(folder_path_with_slash_suffix, file_name):
    # Read and process data for given filename
    file_path = folder_path_with_slash_suffix + file_name

    wf_data = fn_read_wf_data_from_filename(file_path)
    # print(wf_data.head())

    # get asset features from filename
    wf_features = fn_get_features_list_from_filename(file_name)

    # Create waveform dataset with all features
#    wf_data_df = pd.DataFrame({'x': wf_data['x'],
#                               'y': wf_data['y'],
#                               'pump_name': wf_features['pump_name'],
#                               'sensor_name': wf_features['sensor_name'],
#                               'date': wf_features['date'],
#                               'time': wf_features['time'],
#                               'rpm': wf_features['rpm'],
#                               'datetime': wf_features['datetime']
#                               })
    
    # Append asset features
    wf_data['pump_name'] = wf_features['pump_name']
    wf_data['sensor_name'] = wf_features['sensor_name']
    wf_data['rpm'] = wf_features['rpm']
    wf_data['measurement_type'] = 'accl'
#    wf_data['date'] = wf_features['date']
#    wf_data['time'] = wf_features['time']
    wf_data['datetime'] = wf_features['datetime']
    
    return(wf_data)
    

print(fn_process_wf_file_single(data_path, file_names[0]).head())

#%% 3.1 Function to process spectrum data, append features & create dataframe

def fn_get_vfd_status(pump_name):
    if((pump_name == 'g-pump-1') | (pump_name == 'g-pump-4')):
        return(True)
    else:
        return(False)
#print(fn_get_vfd_status(pump_name = 'g-pump-4'))

def fn_process_wf_file_to_spec_single(folder_path_with_slash_suffix, file_name):
    # Read and process data for given filename
    file_path = folder_path_with_slash_suffix + file_name

    # get asset features from filename
    wf_features = fn_get_features_list_from_filename(file_name)
    
    isVFD = fn_get_vfd_status(wf_features['pump_name'])
    
    df_spec, fundamental_freq, unit = process_waveform_file(file_path,
                                        isVFD = isVFD)
    # print(spec_data.head())

    # Append asset & spectrum features
    df_spec['fundamental_freq'] = fundamental_freq
    df_spec['unit'] = unit
    df_spec['pump_name'] = wf_features['pump_name']
    df_spec['sensor_name'] = wf_features['sensor_name']
    df_spec['rpm'] = wf_features['rpm']
#    df_spec['date'] = wf_features['date']
#    df_spec['time'] = wf_features['time']
    df_spec['datetime'] = wf_features['datetime']
    
    return(df_spec)
    

print(fn_process_wf_file_to_spec_single(data_path, file_names[930]).head())
#%% 4. Consolidate waveform data
# Below function processes accl waveform data and create dataframe with asset indentifiers
# Additional task to be added to this function:
# - Convert accl wf to accl spec
# - Convert accl spec to velocity spec
# - Extract features for 1X, 2X, 3X & 9X amplitudes 
# - Save datasets seperately: wf_accl, spec_accl, spc_vel
# - Save dataset for harmonic features + datetime + asset identifiers

def fn_process_wf_data_from_folder(folder_path, no_of_file_to_process = -1):
    # get available waveform file names in the folder
    file_names = fn_get_filenames(folder_path)
    print(len(file_names), "files found.")
    
    if(no_of_file_to_process != -1):
        file_names = file_names[0:no_of_file_to_process]
    
    # Create empty dataframe to store the data
    wf_consolidate = pd.DataFrame()
    spec_consolidate = pd.DataFrame()
    
    # Process waveform files one by one and append to wf_consolidate
    print('Processing', len(file_names), "files.")
    for file in file_names:
        df_wf_temp = fn_process_wf_file_single(folder_path, file)
        wf_consolidate = wf_consolidate.append(df_wf_temp, sort=False)    
        
        df_spec_temp = fn_process_wf_file_to_spec_single(folder_path, file)
        spec_consolidate = spec_consolidate.append(df_spec_temp, sort=False)

    # reset index return dataframes
    wf_consolidate.reset_index(drop=True, inplace=True)
    spec_consolidate.reset_index(drop=True, inplace=True)
    
    return(wf_consolidate, spec_consolidate)


#%%
##############################################################################
#############  Master Function to Call and Process Waveform Data Files #######
##############################################################################

# record start time
dt_start = datetime.datetime.now()
    
# pass -1 to run process all files
wf_df, spec_df = fn_process_wf_data_from_folder(data_path, no_of_file_to_process=-1)

# Drop duplicate records from waveform and spectrum dataset
wf_df = wf_df.drop_duplicates()
spec_df = spec_df.drop_duplicates()

# Calculate total time taken in execution
dt_end = datetime.datetime.now()
print("Completed in ", dt_end - dt_start)


#%%    
##############################################################################
#############  Save Processed Data to CSVs        ############################
##############################################################################

#file_save_path = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Processed data\\"
#wf_df.to_csv(file_save_path + 'waveform_data_processed' + '_' + 
#             str(datetime.datetime.timestamp(datetime.datetime.now())) + '.csv', index=False)
#spec_df.to_csv(file_save_path + 'spectrum_data_processed' + '_' + 
#             str(datetime.datetime.timestamp(datetime.datetime.now())) + '.csv', index=False)

#%%
##############################################################################
#############  Save Processed Data to Pickles        #########################
##############################################################################

file_save_path = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Processed data\\"

# Save datasets as pickle/CSV
wf_file_name_pkl = file_save_path + 'wf_df.pickle'
file_out = open(wf_file_name_pkl,"wb")
pickle.dump(wf_df, file_out)
file_out.close()
##%%
# Write spectrum data to pickle
spec_file_name_pkl = file_save_path + 'spec_df.pickle'
file_out = open(spec_file_name_pkl,"wb")
pickle.dump(spec_df, file_out)
file_out.close()

#%%

# Load pickle files again in memory
file_in = open(wf_file_name_pkl,"rb")
wf_df = pickle.load(file_in)
file_in.close()


#file_in = open(file_save_path+'spec_df_pump4_axial_full_NEW.pickle',"rb")
file_in = open(spec_file_name_pkl,"rb")
spec_df = pickle.load(file_in)
file_in.close()


