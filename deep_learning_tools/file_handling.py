#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:21:17 2021

@author: matthew
"""

import pdb

#%%



#%%

def file_merger(files, variables, object_array = False): 
    """Given a list of npz files of tensors ready for use with Keras, open them and merge into one array.  
    Note that other objects (e.g. lists and dicts) can be in the npz files, but can't be opened.  
    Inputs:
        files | list | list of paths to the .npz files
        variables
        object_array | 
    Returns
        batch_data_merged | dict 
    History:
        2020/10/?? | MEG | Written
        2020/11/11 | MEG | Update to remove various input arguments
        2021_06_15 | MEG | Add option to check that files is a list.  
    
    """
    import numpy as np
    
    def open_synthetic_data_npz(name_with_path, variables = ['X', 'Y'], object_array = False):
        """Open a file data file """  
        data = np.load(name_with_path)
        data_open = {}
        for variable in variables:
            data_open[variable] = data[variable]    
        return data_open

    # 1: Check inputs are correct:
    if type(files) != list:
        raise Exception(f"'files' must be a list of either strings or paths to each file, and not a {type(files)}.  Exiting.  ")

    n_files = len(files)
    
    batch_data_merged = {}                                                                      # initiate to store all the variables from all the files in.  
    for i, file in enumerate(files):                                                            # loop through each of the files to be merged.  
        batch_data = open_synthetic_data_npz(file, variables, object_array)                     # open the npz file and return a dict containing the variables that are numpy tensors
        #
        for variable in variables:
            if i == 0:                                                                                  # if the first iteration of the loop, need to get the size of the 
                batch_data_shape = list(batch_data[variable].shape)
                n_data_per_file = batch_data_shape[0]
                batch_data_shape[0] *= n_files
                batch_data_merged[variable] = np.zeros(tuple(batch_data_shape))
                
            batch_data_merged[variable][i*n_data_per_file:(i*n_data_per_file)+n_data_per_file,] = batch_data[variable]
                          
    return batch_data_merged



#%%


def file_list_divider(file_list, n_files_train, n_files_validate, n_files_test):
    """ Given a list of files, divide it up into training, validating, and testing lists.  
    Inputs
        file_list | list | list of files
        n_files_train | int | Number of files to be used for training
        n_files_validate | int | Number of files to be used for validation (during training)
        n_files_test | int | Number of files to be used for testing
    Returns:
        file_list_train | list | list of training files
        file_list_validate | list | list of validation files
        file_list_test | list | list of testing files
    History:
        2019/??/?? | MEG | Written
        2020/11/02 | MEG | Write docs
        2021_05_25 | MEG | Move a copy to deep learning tools 
        """
    file_list_train = file_list[:n_files_train]
    file_list_validate = file_list[n_files_train:(n_files_train+n_files_validate)]
    file_list_test = file_list[(n_files_train+n_files_validate) : (n_files_train+n_files_validate+n_files_test)]
    return file_list_train, file_list_validate, file_list_test


#%% plot_data_and_reconstructions




def plot_data_and_reconstructions(X, Y, n_datas=10, outpath = None):
    """Given 4d tensors of vae inputs and outputs, plot them.  
    Inputs:
        X | rank 4 numpy array | images in standard keras format.  Function assumes working with 1 channel data
        Y | rank 4 numpy array | images in standard keras format.  Function assumes working with 1 channel data
        n_data | int | number of iamges to show (as columns of the plot)
    Returns:
        figure
    History:
        2021_05_18 | MEG | Written
        
    """
    import matplotlib.pyplot as plt
    
    f, axes = plt.subplots(2, n_datas)
    for n_data in range(n_datas):
        axes[0, n_data].imshow(X[n_data, :,:,0])
        axes[1, n_data].imshow(Y[n_data, :,:,0])
    if outpath != None:
        f.savefig(outpath)
