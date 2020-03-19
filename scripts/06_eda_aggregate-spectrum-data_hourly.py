# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:18:23 2020

@author: manishkumar-m

Objective: Aggregate spectrum data (Hourly)
 1. Read spectrum data processed by "01_eda_process-vibration-data.py"
 2. Extract spectrum features, e.g. 1X, 2X, 3X & 9X to start with
 3. Trend spectrum features
 4. Aggregate spectrum dataset for selected features - (hourly)
 5. Create function to process each task
 6. save dataset to CSV/pickle
 
"""

# Load modules
import numpy as np
import pandas as pd
import pickle
#import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import os
import datetime
from scipy.stats import kurtosis as ss_kurtosis

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

#%% - FUNCTIONS
#
## 1. Function to plot spectrum
#def fn_plot_spectrum(dataset, x_var, y_var, x_lab, y_lab, plot_title):
#    spec_plot = go.Figure()
#    # Add traces
#    spec_plot.add_trace(go.Scatter(x=dataset[x_var], y=dataset[y_var],
#                                   mode='lines', name="Spectrum"))
#    spec_plot.update_layout(template='none', 
#                            title={ 'text': plot_title, 'y':0.95, 'x':0.1, 'xanchor': 'left', 'yanchor': 'top'},
#                            xaxis_title = x_lab, yaxis_title = y_lab, xaxis = dict( range=[0,1000]))
#    return(spec_plot)
#
#
## 2. Function to create plot_title based on input features
#def fn_get_spectrum_plot_title(dataset, prefix, var_list):
#    dataset = dataset[0:1]
#    suffix = ''
#    
#    for var in var_list:
#        if(suffix!=''):
#            suffix = suffix + ' | ' + str(list(dataset[var])[0])
#        else:
#            suffix = list(dataset[var])[0]
#    
#    plot_title = prefix + ': ' + suffix
#    return(plot_title)
#    
## fn_get_spectrum_plot_title(dataset=spec1, prefix='Spectrum for', var_list = ['pump_name', 'sensor_name', 'datetime'])
#
#
## 3. Function to plot spectrum with plot title
#def fn_plot_spectrum_auto(dataset, x_var, y_var, x_lab, y_lab):
#    plot_title_new = fn_get_spectrum_plot_title(dataset, prefix='Spectrum for', var_list = ['pump_name', 'sensor_name', 'datetime'])
#    
#    spec_plot = fn_plot_spectrum(dataset = dataset, x_var = x_var, y_var = y_var, 
#                             x_lab = x_lab, y_lab = y_lab,
#                             plot_title = plot_title_new)
#    return(spec_plot)
#
#
## Function to get filtered order data only
## Return dataset
#def fn_get_order_data(dataset, order_to_trend_list = [1], order_var = 'order'):
#    return(dataset[dataset[order_var].isin(order_to_trend_list)])
#
#
## 4. Function to get order trend plots for given harmonic
## Return dataset & plot both
#def fn_get_order_trend_plot(dataset, order_to_trend = 1):
#    spec_df_x = dataset[dataset['order'] == order_to_trend]
#    
#    # Visualize trend
#    plot_title = 'Spectrum ' + str(order_to_trend) + 'X trend ' + '| ' + list(spec_df_x['pump_name'].head(1))[0] + ' | ' + list(spec_df_x['sensor_name'].head(1))[0] 
#
#    spec_plot = go.Figure()
#    # Add traces
#    spec_plot.add_trace(go.Scatter(x=spec_df_x['datetime'], y=spec_df_x['amp_velocity'],
#                                    mode='lines', name="Spectrum"))
#    spec_plot.update_layout(template='none', title={ 'text': plot_title, 'y':0.95, 'x':0.1, 'xanchor': 'left', 'yanchor': 'top'},
#                             xaxis_title = 'Datetime', yaxis_title='Amplitude')
#    # spec_plot.show()
#    return(spec_df_x, spec_plot)
#


#%%
# 5. Read processed spectrum data from pickle file
# Inputs: i) path, ii) pump name

# Load spectrum data
file_save_path = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Processed data\\"
#spec_file_name = 'spec_df_pump5_axial_full.pickle'

def fn_load_spectrum_data_pkl(file_path, pump_name = 'pump5', sensor_name = 'axial'):
    spec_file_name = 'spec_df_' + pump_name + '_' + sensor_name + '_full.pickle'
    file_path = file_path + spec_file_name
    
    if(os.path.exists(file_path)):
        print('File exists.')
        file_in = open(file_path,"rb")
        spec_df = pickle.load(file_in)
        file_in.close()
        return(spec_df)
    else:
        print('File does not exist.')
        return(pd.DataFrame())        
    

#spec_df_pump_axial = fn_load_spectrum_data_pkl(file_path = file_save_path, pump_name = 'pump5', sensor_name='axial')
#spec_df_pump_axial = fn_load_spectrum_data_pkl(file_path = file_save_path, pump_name = 'pump51', sensor_name='axial')


# 6. Read processed waveform data from pickle file
# Inputs: i) path, ii) pump name

# Load waveform data
def fn_load_waveform_data_pkl(file_path, pump_name = 'pump5', sensor_name = 'axial'):
    wf_file_name = 'wf_df_' + pump_name + '_' + sensor_name + '_full.pickle'
    file_path = file_path + wf_file_name
    
    if(os.path.exists(file_path)):
        print('File exists.')
        file_in = open(file_path,"rb")
        wf_df = pickle.load(file_in)
        file_in.close()
        return(wf_df)
    else:
        print('File does not exist.')
        return(pd.DataFrame())  

#wf_df_pump_axial = fn_load_waveform_data_pkl(file_path = file_save_path, pump_name = 'pump5', sensor_name='axial')
##wf_df_pump_axial = fn_load_waveform_data_pkl(file_path = file_save_path, pump_name = 'pumpxx', sensor_name='axial')
#wf_df_pump_axial = wf_df_pump_axial[wf_df_pump_axial['datetime'] == wf_df_pump_axial['datetime'][0]]

#%%
# 7. function to get binwidth from spectrum data
#    - Take a spectrum as input
#    - Sort data by timestamp & frequency
#    - Binwidth is difference of two frequencies. (f2 - f1)
def fn_get_binwidth_from_spectrum(spec_df, freq_var = 'freq', timestamp_var = 'datetime'):
    spec_df.sort_values(by=[timestamp_var, freq_var], inplace=True)
    return(spec_df[freq_var][1] - spec_df[freq_var][0])


spec_df_pump_axial = fn_load_spectrum_data_pkl(file_path = file_save_path, pump_name = 'pump5', sensor_name='axial')
#spec_df_pump_radial = fn_load_spectrum_data_pkl(file_path = file_save_path, pump_name = 'pump5', sensor_name='radial')
#spec_df_pump_axial = spec_df_pump_axial[spec_df_pump_axial['datetime'] == spec_df_pump_axial['datetime'][0]]
spec_df_pump_axial = spec_df_pump_axial[spec_df_pump_axial['datetime'].isin(spec_df_pump_axial['datetime'].unique()[0:2])]

#fn_get_binwidth_from_spectrum(spec_df_pump_axial)

#%%
# 8. Function to get amplitude of given order, considering binwidth for single spectrum
def fn_get_amplitude_by_order_binwidth_single_spectrum(spec_df, order_to_fetch=1, binwidth_to_check=3, 
                                                       freq_var='freq', amp_var='amp_velocity', 
                                                       timestamp_var='datetime', order_var='order'):
    # Assumption: Given dataset has only one spectrum
    # Still filtering data for first timestamp to avoid any descrepency
#    spec_df = spec_df[spec_df[timestamp_var]==spec_df[timestamp_var].head()[0]]
    spec_df.reset_index(drop = True, inplace = True)
    spec_df = spec_df[spec_df[timestamp_var]==spec_df[timestamp_var].head()[0]]
 
    # get frequency for given order
    order_freq = float(spec_df[spec_df[order_var]==order_to_fetch][freq_var])
    
    # get binwidth
    binwidth = fn_get_binwidth_from_spectrum(spec_df, freq_var=freq_var, timestamp_var=timestamp_var)
    
    # multiply binwidth by 'binwidth_to_check'
    freq_low = order_freq - (binwidth * binwidth_to_check)
    freq_high = order_freq + (binwidth * binwidth_to_check)
 
    # get frequency range by adding/deducting binwidth range to order frequency
    # get all frequencies in this range
    # get amplitudes for these frequencies
    # get maximum amplitude among these and return
    freq_peaks = spec_df[spec_df[freq_var].between(freq_low, freq_high)][amp_var]

    return(max(freq_peaks))

#fn_get_amplitude_by_order_binwidth_single_spectrum(spec_df_pump_axial, order_to_fetch=9)
#%%
# 9. Dummy function to get asset_id by asset_name as this is currently not available in data file
def fn_get_asset_id(asset_name):
    return('aid-' + asset_name)

# Dummy function to get metric_type
def fn_get_measurement_type_by_unit(metric_unit):
    if(metric_unit.lower() == 'mms' or metric_unit.lower() == 'ips' or metric_unit.lower() == 'inps'):
        return('velocity')
    elif(metric_unit.lower() == 'ms2' or metric_unit.lower() == 'g'):
        return('acceleration')
    else:
        return('unknown')

#fn_get_measurement_type_by_unit('IPS')
#%%
# 10. Function to create dataset for spectrum orders- single spectrum - Velocity/acceleration
# Take specturm - single
# Take orders to process
# Run for each order:
#   get amplitude by order considering binwidth
#   add relevant features to the dataset
def fn_create_spectrum_feature_dataset_single(spec_df, order_to_process_list=[1,9], 
                                              binwidth_to_check=3, sensor_type = 'spectrum', 
                                              amp_var='amp_velocity', feature_unit='mms', 
                                              asset_var='pump_name', sensor_info_var = 'sensor_name',
                                              timestamp_var='datetime', freq_var='freq', 
                                              order_var='order'):
    # Assumption: Given dataset has only one spectrum
    # Still filtering data for first timestamp to avoid any descrepency
    spec_df.reset_index(drop = True, inplace = True)
    spec_df = spec_df[spec_df[timestamp_var]==spec_df[timestamp_var].head()[0]]
    
    asset_id = fn_get_asset_id(asset_name = spec_df.head()[asset_var][0])
    
    #Get metric_type/measurement_type
    measurement_type = fn_get_measurement_type_by_unit(feature_unit)
    
    df_spec_features = pd.DataFrame()
    for order in order_to_process_list:
        order_amp = fn_get_amplitude_by_order_binwidth_single_spectrum(spec_df, order_to_fetch=order, 
                                                                       amp_var=amp_var)
        df_temp = pd.DataFrame({'asset_id': asset_id,
                                'asset_name': spec_df[asset_var][0],
                                'sensor_type': sensor_type,
                                'measurement_type': measurement_type,
                                'feature_name': 'spectrum_' + str(int(order)) + 'x',
                                'feature_value': [order_amp],
                                'feature_unit': feature_unit,
                                'sensor_info': spec_df[sensor_info_var][0],
                                'datetime': spec_df[timestamp_var][0]                                
                                })
        df_spec_features = df_spec_features.append(df_temp)
    
    df_spec_features.reset_index(drop=True, inplace=True)
    return(df_spec_features)

#fn_create_spectrum_feature_dataset_single(spec_df_pump_axial, order_to_process_list=[1,9,18,27],
#                                          sensor_type = 'spectrum', amp_var='amp_velocity',
#                                          feature_unit='mms')
#fn_create_spectrum_feature_dataset_single(spec_df_pump_axial, direction='axial', order_to_process_list=[1,9,18,27])
#fn_create_spectrum_feature_dataset_single(spec_temp, direction='axial', order_to_process_list=[1,9,18,27])
#%%
# 11. Function to create hourly aggregated spectrum feature dataset - Velocity/acceleration

# Take spectrums dataset
# Process each spectrum one by one
# Consolidate spectrum features
# Aggregate spectrum features on hours - mean amplitude
# return full dataset with all spectrum features

def fn_create_spectrum_feature_dataset_hourly_agg(spec_df, order_to_process_list=[1], 
                                                  binwidth_to_check=3, sensor_type = 'spectrum', 
                                                  amp_var='amp_velocity', feature_unit='mms',
                                                  asset_var='pump_name', sensor_info_var = 'sensor_name',
                                                  timestamp_var='datetime', freq_var='freq',  
                                                  order_var='order', feature_val_var = 'feature_value'):
    # get unique timestamps
    unique_spec_datetime = spec_df[timestamp_var].unique()
#    print(len(unique_spec_datetime), 'Unique spectrums')
  
    # What spectrum orders to process
#    order_list_to_process = [1,2,3,9,18,27,36,45,54]
    
    # Create empty dataframe
    df_spec_consolidate = pd.DataFrame()
#    i=0
    # Process each spectrum
#    for t in unique_spec_datetime[0:20]:
    for t in unique_spec_datetime:
        spec_temp = spec_df[spec_df[timestamp_var]==t]
#        print(t)
        df_temp = fn_create_spectrum_feature_dataset_single(spec_df=spec_temp,
                            order_to_process_list=order_to_process_list, binwidth_to_check=binwidth_to_check,  
                            sensor_type = sensor_type, amp_var = amp_var, feature_unit = feature_unit, 
                            asset_var=asset_var, sensor_info_var = sensor_info_var, 
                            timestamp_var=timestamp_var, freq_var=freq_var, 
                            order_var=order_var)
        
        df_spec_consolidate = df_spec_consolidate.append(df_temp)

#    df_spec_consolidate.reset_index(drop = True, inplace = True)
    
    # Aggregate spectrum features on hours - mean amplitude
    df_spec_consolidate[timestamp_var] = df_spec_consolidate[timestamp_var].dt.floor('H')
    col_names_to_groupby = df_spec_consolidate.columns.to_list()
    col_names_to_groupby.remove(feature_val_var)
    df_spec_consolidate = df_spec_consolidate.groupby(col_names_to_groupby).mean()[feature_val_var].reset_index()
    return(df_spec_consolidate)


#%%
## record start time
#dt_start = datetime.datetime.now()
#
#spec_feature_df = fn_create_spectrum_feature_dataset_hourly_agg(spec_df_pump_axial,  
#                    order_to_process_list=[1,2,3,9,18,27,36,45,54], sensor_type = 'spectrum',
#                    amp_var='amp_velocity', feature_unit ='mms')
##                    amp_var='amp_acceleration', feature_unit='ms2')
#
#print(spec_feature_df.head())
#print(spec_feature_df.shape)
#
## Calculate total time taken in execution
#dt_end = datetime.datetime.now()
#print("Completed in ", dt_end - dt_start)

#%%
# 12. Function to create dataset for waveform features- single waveform - acceleration
# Take waveform - single
# Take metric list to process
# Process each metric over given waveform
# Create dataset
# Add relevant features to the dataset

#def fn_create_waveform_feature_dataset_single(wf_df, direction,  
#                                                  asset_var='pump_name', param_var = 'sensor_name',
#                                                  timestamp_var='datetime', time_var='x', amp_var='y', 
#                                                  metric_unit='ms2'):
def fn_create_waveform_feature_dataset_single(wf_df, sensor_type = 'waveform',  
                                              amp_var='y', feature_unit='ms2',
                                              asset_var='pump_name', sensor_info_var = 'sensor_name',
                                              timestamp_var='datetime', time_var='x',
                                              feature_name_var = 'feature_name', 
                                              feature_value_var = 'feature_value'):
    # Assumption: Given dataset has only one spectrum
    # Still filtering data for first timestamp to avoid any descrepency
    wf_df.reset_index(drop = True, inplace = True)
    wf_df = wf_df[wf_df[timestamp_var]==wf_df[timestamp_var].head()[0]]
    
    asset_id = fn_get_asset_id(asset_name = wf_df.head()[asset_var][0])
    
    #Get feature_type
    measurement_type = fn_get_measurement_type_by_unit(feature_unit)
    
    # What features to group on
    col_to_group_on  = [asset_var, sensor_info_var, timestamp_var]
    col_to_group_on  = timestamp_var
    
    # Define metric functions 
    def RMS(x):
        return np.sqrt(np.mean(np.square(x)))
    def mean(x):
        return np.mean(x)
    def peak(x):
        return np.max(np.abs(x))
    def pk_pk(x):
        return (np.max(x) - np.min(x))
    def crest_factor(x):
        return np.max(np.abs(x))/np.sqrt(np.mean(np.square(x)))
    def kurtosis(x):
        return ss_kurtosis(x, bias=False)
    
    # Define aggregation rule here
    ops1 = [RMS, mean, peak, pk_pk, crest_factor, kurtosis] # features to calculate on waveform amplitude
    aggregations = {}
    aggregations = dict.fromkeys(['y'], ops1)
    #aggregations.update(dict.fromkeys(['col_name1'], ops2))
    
    # Aggregate data and rename the columns.
    agg_df = pd.DataFrame() # Define a dataframe
    agg_df = wf_df.groupby(col_to_group_on, as_index=False).agg(aggregations)
    agg_df.columns = [x[1] if x[1]!='' else x[0] for x in agg_df.columns.ravel()]
    agg_df = pd.melt(agg_df, id_vars=col_to_group_on, 
                     value_vars=['RMS', 'mean', 'peak', 'pk_pk', 'crest_factor', 'kurtosis'],
                     var_name = feature_name_var, value_name = feature_value_var)
    agg_df['asset_id'] = asset_id
    agg_df['asset_name'] = wf_df[asset_var][0]
    agg_df['sensor_type'] = sensor_type
    agg_df['measurement_type'] = measurement_type
    agg_df['feature_unit'] = feature_unit
    agg_df['sensor_info'] = wf_df[sensor_info_var][0]

    agg_df = agg_df[['asset_id', 'asset_name', 'sensor_type', 'measurement_type', feature_name_var, 
                     feature_value_var, 'feature_unit', 'sensor_info', timestamp_var]]

    return agg_df

#fn_create_waveform_feature_dataset_single(wf_df_pump_axial, sensor_type = 'waveform',  
#                                              amp_var='y', feature_unit='ms2',
#                                              asset_var='pump_name', sensor_info_var = 'sensor_name',
#                                              timestamp_var='datetime', time_var='x',
#                                              feature_name_var = 'feature_name', 
#                                              feature_value_var = 'feature_value')

#%%
# 13. Function to create hourly aggregated waveform feature dataset - acceleration

# Take waveform dataset
# Process each waveform one by one
# Consolidate waveform features
# Aggregate waveform features on hours - mean of metric value if there is more than one value
# return full dataset with all waveform features

#def fn_create_waveform_feature_dataset_hourly_agg(wf_df, direction,
#                                                  asset_var='pump_name', param_var = 'sensor_name',
#                                                  timestamp_var='datetime', time_var='x', amp_var='y', 
#                                                  metric_unit='ms2'):
def fn_create_waveform_feature_dataset_hourly_agg(wf_df, sensor_type = 'waveform',  
                                              amp_var='y', feature_unit='ms2',
                                              asset_var='pump_name', sensor_info_var = 'sensor_name',
                                              timestamp_var='datetime', time_var='x',
                                              feature_name_var = 'feature_name', 
                                              feature_value_var = 'feature_value'):
    # get unique timestamps
    unique_wf_datetime = wf_df[timestamp_var].unique()
    # print(len(unique_wf_datetime), 'Unique spectrums')
      
    # Create empty dataframe
    df_wf_consolidate = pd.DataFrame()

    # i=0
    # Process each waveform
#    for t in unique_wf_datetime[0:20]:
    for t in unique_wf_datetime:
        wf_temp = wf_df[wf_df[timestamp_var]==t]
#        print(t)
        df_temp = fn_create_waveform_feature_dataset_single(wf_temp, sensor_type = sensor_type,  
                                              amp_var=amp_var, feature_unit=feature_unit,
                                              asset_var=asset_var, sensor_info_var = sensor_info_var,
                                              timestamp_var=timestamp_var, time_var=time_var,
                                              feature_name_var = feature_name_var, 
                                              feature_value_var = feature_value_var)
        df_wf_consolidate = df_wf_consolidate.append(df_temp)

#    df_wf_consolidate.reset_index(drop = True, inplace = True)
    
    # Aggregate waveform features on hours - mean amplitude of features
    df_wf_consolidate[timestamp_var] = df_wf_consolidate[timestamp_var].dt.floor('H')
    col_names_to_groupby = df_wf_consolidate.columns.to_list()
    col_names_to_groupby.remove(feature_value_var)
    df_wf_consolidate = df_wf_consolidate.groupby(col_names_to_groupby).mean()[feature_value_var].reset_index()
    return(df_wf_consolidate)


#fn_create_waveform_feature_dataset_hourly_agg(wf_df_pump_axial, sensor_type = 'waveform',  
#                                              amp_var='y', feature_unit='ms2')

#%%
# 14. Function to create hourly aggregated spectrum & waveform feature dataset - Velocity/acceleration
# This will be for all spectrum features & waveform features
# Get features for velocity spectrum, acceleration spectrum & accleratio waveform one by one
#   & consolidate them together. The consolidated data should be hourly aggregated.
def fn_process_spec_wf_to_hourly_aggregated_feature_dataset(pump_name = 'pump5'):
    
    # Load spectrum & waveform data (This process can be replaced by data collection method for given pump)
    file_save_path = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Processed data\\"
    
    spec_df_axial = fn_load_spectrum_data_pkl(file_path = file_save_path, pump_name = pump_name, 
                                                   sensor_name = 'axial')    
    spec_df_radial = fn_load_spectrum_data_pkl(file_path = file_save_path, pump_name = pump_name, 
                                                   sensor_name = 'radial')
    
    wf_df_axial = fn_load_waveform_data_pkl(file_path = file_save_path, pump_name = pump_name, 
                                                   sensor_name = 'axial')
    wf_df_radial = fn_load_waveform_data_pkl(file_path = file_save_path, pump_name = pump_name, 
                                                   sensor_name = 'radial')

    # Process spectrum features - axial
    #------------------------------------------------
    spec_feature_axial_vel = fn_create_spectrum_feature_dataset_hourly_agg(spec_df_axial, 
                        order_to_process_list=[1,2,3,9,18,27,36,45,54], sensor_type = 'spectrum',
                        amp_var='amp_velocity', feature_unit='mms')
    
    spec_feature_axial_accl = fn_create_spectrum_feature_dataset_hourly_agg(spec_df_axial, 
                        order_to_process_list=[1,2,3,9,18,27,36,45,54], sensor_type = 'spectrum',
                        amp_var='amp_acceleration', feature_unit='ms2')


    # Process spectrum features - radial
    #------------------------------------------------
    spec_feature_radial_vel = fn_create_spectrum_feature_dataset_hourly_agg(spec_df_radial, 
                        order_to_process_list=[1,2,3,9,18,27,36,45,54], sensor_type = 'spectrum',
                        amp_var='amp_velocity', feature_unit='mms')


    spec_feature_radial_accl = fn_create_spectrum_feature_dataset_hourly_agg(spec_df_radial,
                        order_to_process_list=[1,2,3,9,18,27,36,45,54], 
                        amp_var='amp_acceleration', feature_unit='ms2')    

    # Process velocity waveform features
    # We don't have velocity waveforms yet
    
    # Process waveform features - radial 
    #------------------------------------------------
    # We only have waveforms in acceleration
    wf_feature_axial_accl = fn_create_waveform_feature_dataset_hourly_agg(wf_df_axial, 
                                    sensor_type = 'waveform', amp_var='y', feature_unit='ms2',
                                    time_var='x')

    
    # Process waveform features - radial 
    #------------------------------------------------
    # We only have waveforms in acceleration
    wf_feature_radial_accl = fn_create_waveform_feature_dataset_hourly_agg(wf_df_radial, 
                                    sensor_type = 'waveform', amp_var='y', feature_unit='ms2',
                                    time_var='x')
    
    # Consolidate all features and return dataset
    
    df_feature_consolidate = pd.DataFrame()
    df_feature_consolidate = df_feature_consolidate.append(spec_feature_axial_vel).append(spec_feature_axial_accl)
    df_feature_consolidate = df_feature_consolidate.append(spec_feature_radial_vel).append(spec_feature_radial_accl)
    df_feature_consolidate = df_feature_consolidate.append(wf_feature_axial_accl).append(wf_feature_radial_accl)
    
    df_feature_consolidate.reset_index(drop=True, inplace=True)
    return(df_feature_consolidate)


#%%
# record start time
dt_start = datetime.datetime.now()

vib_feature_df = fn_process_spec_wf_to_hourly_aggregated_feature_dataset(pump_name = 'pump6')

print(vib_feature_df.head())
print(vib_feature_df.shape)

# Calculate total time taken in execution
dt_end = datetime.datetime.now()
print("Completed in ", dt_end - dt_start)

#%%
# 13. Function to save aggregated hourly dataset for vibration features to CSV/Pickle
##############################################################################
#############  Save Processed Data to CSV/Pickles        #####################
##############################################################################

target_path = "C:\\Users\\manishkumar-m\\OneDrive - HCL Technologies Ltd\\Desktop_old\\Clients\\Danfoss\\Data\\New Data Dump\\Processed data\\hourly data\\"

# Save datasets as pickle/CSV
vib_file_name_pkl = target_path + 'vib_feature_df.pickle'
file_out = open(vib_file_name_pkl ,"wb")
pickle.dump(vib_feature_df, file_out)
file_out.close()


vib_feature_df.to_csv(target_path + 'vib_feature_df' + '_' + 
             str(datetime.datetime.timestamp(datetime.datetime.now())) + '.csv', index=False)




#%%

#%%