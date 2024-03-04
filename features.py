import numpy as np
import math
import spectrum as sp
import loader
import pandas as pd

samp_frequency = 100 # Hz - defined in the paper

# Window size (time) = 250 ms
window_size = 250 

def mean(window):
    return np.mean(window)

def max(window):
    return np.max(window)

def min(window):
    return np.min(window)

def rms(window):
    rms = 0
    for x in window:
        rms = rms + x**2

    rms = rms/len(window)
    rms = math.sqrt(rms)
    return rms

def std(window):
    return np.std(window)

def mav(window):
    '''
    Returns the mean absolute value of the window
    '''
    return np.mean(np.absolute(window))

def zc(window):
    '''
    Returns the zero crossings for the given window 
    '''
    zc = 0
    for i in np.arange(len(window) - 1):
        if window[i] > 0 and window[i+1] < 0:
            if abs(window[i] - window[i+1]) >= 0.01:
                zc += 1

        elif window[i] < 0 and window[i+1] > 0:
            if abs(window[i] - window[i+1]) >= 0.01:
                zc += 1

    return zc

def ssc(x):
    '''
    Slope sign changes
    '''
    ssc = 0
    for i in range(len(x) - 2):
        if abs(x[i] - x[i+1]) > 0.01 or abs(x[i+1] - x[i+2]) > 0.01:
            if x[i+1] > x[i] and x[i+1] > x[i+2]:
                ssc += 1

            elif x[i+1] < x[i] and x[i+1] < x[i+2]:
                ssc += 1

    return ssc 

def wfl(window):
    wfl = 0
    for i in range(len(window) - 1):
        wfl += abs(window[i] - window[i+1])

    return wfl

def levinson(window):
    '''Coefficients of the Levinson-Durbin Recursion'''
    levinson_coeff = sp.LEVINSON(window, order=6, allow_singularity=True)
    coeff = levinson_coeff[0]
    coeff = coeff.tolist()
    return coeff

def getFeaturesFromWindow(window: np.array):
    '''
    Returns features for each EMG signal window
    Inputs: EMG signal data window  (1D np array)
    Returns: 2D array with features for each signal
    '''
    features = [ssc(window),
                zc(window),
                mean(window),
                rms(window), 
                min(window), 
                max(window),
                wfl(window),
                mav(window), 
                std(window)] + levinson(window)
    
    features = np.array(features)
    return features

def getFeatures(signals, torques):
    features = np.array([])
    flag = False
    for i in range(np.shape(torques)[0]):
        if flag == True:
            flag = False
            continue
        else:
            flag = True
        if torques[i][0] - signals[0][0] >= 0.250: # check if a window of 250ms exists
            index = np.where(signals[:, 0]  == torques[i][0])[0][0]
            x = np.array(signals[index-250:index, 1:])
            n_cols = np.shape(x)[1]
            for j in range(n_cols):
                if j == 0:
                    extracted_features_from_window = getFeaturesFromWindow(x[:, j])
                else: 
                    extracted_features_from_window = np.concatenate((np.squeeze(extracted_features_from_window), np.squeeze(getFeaturesFromWindow(x[:, j]))))

            extracted_features_from_window = np.concatenate((np.squeeze(extracted_features_from_window), np.squeeze(torques[i][1:])))
            
            if np.shape(features)[0] == 0:
                features = extracted_features_from_window
            else:
                features = np.vstack((features, extracted_features_from_window))
    features = np.array(features)
    
    return features

if __name__ == "__main__":
    df = loader.loadData("C://Users/kodey/Documents/546_Dataset/10_09_18/levelground/emg//levelground_cw_normal_04_01.csv")
    df = pd.DataFrame(df.values)
    df = df.to_numpy()
    
    df2 = loader.loadData("C://Users/kodey/Documents/546_Dataset/10_09_18/levelground/id//levelground_cw_normal_04_01.csv", cols = ['time', 'hip_flexion_r_moment', 'knee_angle_r_moment', 'ankle_angle_r_moment'])
    df2 = pd.DataFrame(df2.values)
    df2 = df2.to_numpy()

    features = getFeatures(df, df2)
    df3 = pd.DataFrame(features)
    df3.to_csv("C://Users/kodey/Documents/546_Dataset/10_09_18/levelground/emg//levelground_cw_normal_04_01_features.csv", index=False)
