# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 13:27:34 2020
@author: ritesh_si
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_spec_from_waveform(df_wf, y_var_name = "amp_acceleration", plot = False):
    """
    get spectrum data from waveform, df_wf should have x(time) and y(amplitude) columns
    """
    N = df_wf.shape[0]; # N # Number of Sample Points
    spacing = df_wf['x'][N-1]/N
    Y    = np.fft.fft(df_wf['y'])
    freq = np.fft.fftfreq(N, spacing)
    df_spectrum = pd.DataFrame( freq, Y)
    #d = {'col1': [1, 2], 'col2': [3, 4]}
    d = {'freq':  freq[:N//2], y_var_name: 2.0/N * np.abs(Y[:N//2])}
    #pd.DataFrame( freq[:N//2], 2.0/N * np.abs(Y[:N//2]))
    #df_filtered_spectrum = pd.DataFrame( freq[:N//2], 2.0/N * np.abs(Y[:N//2]))
    df_filtered_spectrum = pd.DataFrame( data = d)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(freq[:N//2], 2.0/N * np.abs(Y[:N//2]))
        plt.show()
    return(df_filtered_spectrum, df_spectrum)

def accln_to_velocity_old(A_in_grms, F_in_hertz ):
    """
    V = 5217 A divided by F- This is used here.
    V = D times F Square divided by 9.958 X 10 power 7. V is represented in in/sec pk.
    F = Frequency in CPM.
    A = g in RMS. (Need to check if g provided to us is in RMS)
    """
    V_in_inps = 0
    if (F_in_hertz != 0):
        V_in_inps = (5217 * A_in_grms)/ (F_in_hertz * 60)
    return(V_in_inps)

def accln_to_velocity(A_in_m_per_sec2, F_in_hertz):
    """
    V = 5217 A divided by F- This is used here.
    V = D times F Square divided by 9.958 X 10 power 7. V is represented in in/sec pk.
    F = Frequency in CPM.
    A = g in RMS. (Need to check if g provided to us is in RMS)
    """
    A_in_grms = A_in_m_per_sec2 * 0.101972
    V_in_inps = 0
    if (F_in_hertz != 0):
        V_in_inps = (5217 * A_in_grms)/ (F_in_hertz * 60)
    return(V_in_inps)

def accln_to_velocity_v2(amp, freq, G=1):
    """
    freq = Frequency in Hz.
    amp = amplitude in acceleration spectrum.
         (If we convert from RMS then the resulting velocity amplitude will be in RMS.)
    G = G in these formulas is not the acceleration of gravity. 
        It is a constant for calculation within different systems. 
        For metric, G is 9.80665 m/s². For Imperial, G is 386.0885827 in/s² For SI, G is 1 m/s²
    
    * In below formula, multilying nominator with 1000 to convert acceleration amplitude from m/s2 to mm/s2 
    """
    try:
        amp_velocity = (G * amp * 1000)/(2 * np.pi * freq)
        return(amp_velocity)
    except ZeroDivisionError:
        return(0)

def get_fundamental_frequency(df_spec, linefreq= 1780/60, x_var = 'freq', y_var = 'amp_velocity', scan_percent = 10):
    # look at 5% plus and minus range.
    plus = (100 + scan_percent)/ 100 * linefreq
    minus = (100 - scan_percent)/ 100 * linefreq
    flt = df_spec[(df_spec['freq'] < plus) & (df_spec['freq'] > minus)]
    ff = flt[flt['amp_velocity'] == max(flt['amp_velocity'])]
    return(ff.reset_index().at[0,'freq'], 'Hertz')
    #print(ff['freq'].astype(float)* 60)

# Function to find running/fundamental frequency in VFD
def fn_get_fundamental_frequency_vfd(df_spec, lowest_rpm = 700, highest_rpm = 1780,
                   look_percentage = 20, freq_col = 'freq', amp_col = 'amp_velocity',
                   amp_1x_threshold = 1.5 # Test 1x threshold in mm/sec
                   ):
    '''Inputs:
    df_spec- A dataframe for Velocity/acceleration spectrum
    lowest_rpm = Lower RPM limit for the VFD machine
    highest_rpm = Higher RPM limit for the VFD machine
    look_percentage = Possible deviation in the limits
    freq_col = Frequency column name in the data
    amp_col = Amplitude columns name in the data
    amp_1x_threshold = Test 1x threshold in mm/sec - 1.5 mm/sec
    '''
    
    '''Output (a number): Running frequency of the VFD machine'''
    
    # Find lowest and highest frequency limits
    lfl = np.floor((lowest_rpm * (1-look_percentage/100))/ 60)
    hfl = np.ceil((highest_rpm * (1+look_percentage/100))/ 60)
    
    # Filter spectrum data in the lowest and highest frequency range 
    df_ = df_spec[((df_spec[freq_col] >= lfl) & (df_spec[freq_col] <= hfl))].reset_index(drop=True).copy(deep=False)
    df_.sort_values(by=[freq_col], inplace=True)
    
    # Find the frequency in the filtered data corresponding with amplitude. : First label
    freq1 = df_.loc[df_[amp_col].idxmax(), freq_col]
    
    # Look for a shorter frequency range (first 1/3rd range) of the provided range and if there are any higher
    #   amiplitudes than the provided amplitude 1x minimum threshold, that should be the 1x, else the frequency with
    #   highest amplitude in the rpm range is running frequency.
    lfl2 = np.ceil(lfl + (hfl-lfl)/3)
    df__ = df_[((df_[freq_col] >= lfl) & (df_[freq_col] <= lfl2))].reset_index(drop=True).copy(deep=False)
    try: idx2 = df__[df__[amp_col].gt(amp_1x_threshold)].index[0]
    except: idx2 = None

    if not (idx2 is None):
        freq2 = df__.loc[df__[amp_col].idxmax(), freq_col]
        running_freq = freq2 
    else: 
        running_freq = freq1
        
    # Take minimum frequency from the above frequencies. That should be the running frequency.
    #running_freq = np.min([freq1, freq2])
    
    # return running_freq 
    return(running_freq, 'Hertz')  

        
def process_waveform_file(file_path, mode = "aceeleration", linefreq= 1780/60, isVFD = False, 
                          vfd_lowest_rpm=700, vfd_highest_rpm=1780, vfd_look_percentage = 20,
                          freq_var = 'freq', vfd_amp_1x_threshold='1.5'):
    """
    takes a file path which contains waveform data and 
    returns processed spectrum data along with orders and 
    velocity spectrum data plus fundamental frequency
    """
    df_wv = pd.read_csv(file_path); df_wv.head()
    df_spec, _ = get_spec_from_waveform(df_wv, plot = False)
#    df_spec['v_in_inps_old'] = list(map(accln_to_velocity_old, df_spec['a_in_grms'], df_spec['freq'])); #df_spec.head()
#    df_spec['v_in_inps'] = list(map(accln_to_velocity, df_spec['a_in_grms'], df_spec['freq'])); #df_spec.head()
    df_spec['amp_velocity'] = list(map(accln_to_velocity_v2, df_spec['amp_acceleration'], df_spec['freq'])); #df_spec.head()
        
    # Get fundamental frequency
    if(isVFD):
#        print('VFD Machine identified, processing FF detection..')
        fundamental_freq, unit = fn_get_fundamental_frequency_vfd(df_spec, lowest_rpm = vfd_lowest_rpm, 
                                    highest_rpm = vfd_highest_rpm, look_percentage = vfd_look_percentage, 
                                    freq_col = freq_var, amp_col = 'amp_velocity', 
                                    amp_1x_threshold = vfd_amp_1x_threshold # Test 1x threshold in mm/sec
                                    )
    else:
#        print('VNon-FD Machine identified, processing FF detection..')
        fundamental_freq, unit = get_fundamental_frequency(df_spec, linefreq=linefreq)
    
    df_spec['order'] =  df_spec['freq']/(fundamental_freq); 
    return(df_spec, fundamental_freq, unit)






