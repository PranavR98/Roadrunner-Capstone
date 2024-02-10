#!/usr/bin/env python
# coding: utf-8

# In[1]:

import cv2
import os
import numpy as np
import pandas as pd

def check_blurry(directory, file_num): #, var_blur, ff_mean_blur, var_dark, ff_mean_dark, blurry_truth, dark_truth):
     
    '''
    Parameters:
    directory: Your directory to the images. Do not include the file number
    file_num: the file number, str or int
    
    Returns:
    A dataframe that has the laplacian variance and fft of each image
    '''
    
    file_num = str(file_num)
    
    files = np.empty((0,0))
    variances = np.empty((0,0))
    ff_means = np.empty((0,0))

    # Loop through each image in the directory
    for file_name in os.listdir(directory + file_num):
        # Check if the file is an image
        if file_name.endswith(".jpg") or file_name.endswith(".png"):

            files = np.append(files, file_name)
            # Load the image and convert it to grayscale
            image = cv2.imread(os.path.join(directory, file_num, file_name))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate the variance of Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()

            variances = np.append(variances, variance)
            
            # Create the fast fourier transform
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            # Calculate the mean of the magnitude spectrum
            mean = np.mean(magnitude_spectrum)

            ff_means = np.append(ff_means, mean)
           
    #df = pd.DataFrame({'Files':files, 'Variances':variances, 'FFMeans':ff_means, 'Blurry_Truth':blurry_truth, 'Dark_Truth':dark_truth})
    df = pd.DataFrame({'Files':files, 'Variances':variances, 'FFMeans':ff_means})
   # df['Blurry_Test'] = df.apply(lambda row: check_blur(row,var_blur, ff_mean_blur), axis=1)
   # df['Dark_Test'] = df.apply(lambda row: check_dark(row, var_dark, ff_mean_dark), axis=1)
    return df

