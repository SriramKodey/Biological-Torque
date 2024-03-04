import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Go to location and load data
# The data loader function will return a pandas dataframe

def loadData(path, cols = None, header = True):
    '''
    Inputs:
        Path, List of columns to load 
    Returns:
        A pandas datefram with the loaded data
    '''
    if cols == None:
        df = pd.read_csv(path)
        print(df.head())
    elif header == False:
        df = pd.read_csv(path, header=None)
    else:
        df = pd.read_csv(path, usecols=cols)
        print(df.head())

    return df  


if __name__ == "__main__":
    loadData("C://Users/kodey/Documents/546_Dataset/10_09_18/levelground/emg//levelground_cw_normal_05_01.csv")