#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:21:17 2021

@author: matthew
"""

import pdb

#%%



def shuffle_data_pkls(unshuffled_files, file_batch_size, outdir):
    """ Given a direcotry of unshuffled data that is split into files that cannot all be loaded into RAM at once, shuffle by opening 
    a subset of them at a time and shuffling them.  
    
    For a more complete shuffle, this function can be called multiple times on the results of itself.  
    
    Inputs:
        unshuffld_files | list of strings | paths to unshuffled files.  
        file_batch_size | int | number of files in one batch.  Bigger is better, but will eventually max out machine RAM.  
        outdir | pathlib Path | our directory for shuffled files.  
        
    Returns:
        shuffled files.  
        
    History:
        2022_10_11 | MEG | Written.  
        
    
    
    """
    import random
    import numpy as np
    import numpy.ma as ma
    import pickle
    import os

    
    random.shuffle(unshuffled_files)                                                            # shuffle the items in the list (this doesn't actually shuffle the data in the files though, so still called unshuffled)

    if len(unshuffled_files) % file_batch_size != 0:
        raise Exception(f"This function has not been tested with batch sizes that do not divide exactly into the number of files.  "
                        f"({len(unshuffled_files)} were detected, batches were {file_batch_size}, producing a {int(len(unshuffled_files) / file_batch_size)} batches and a remainder of {len(unshuffled_files) % file_batch_size})  "
                        f"Exiting.")
    else:
        n_batches = int(len(unshuffled_files) / file_batch_size)
    
    # get some info about the data by opening the first file.  
    with open(unshuffled_files[0], 'rb') as f:                                                    # open the file
        X = pickle.load(f)
    n_per_file, ny, nx, _ = X.shape
    
    # loop through opening file_batch_size files at once. (e.g. 10 files are opened and the contents shuffled)
    file_n = 0
    for i in range(n_batches):
        unshuffled_batch_files = unshuffled_files[i * file_batch_size: (i+1)*file_batch_size]                           # 
    
        X = ma.zeros((file_batch_size * n_per_file, ny, nx, 1))               # initialise, rank 4 ready for Tensorflow, last dimension is being used for different crops.  
        Y_class = np.zeros((file_batch_size * n_per_file, 3))                                                                             # initialise, doesn't need another dim as label is the same regardless of the crop.  
        Y_loc = np.zeros((file_batch_size * n_per_file, 4))                                                                            # initialise
        
        for file_n_batch, unshuffled_batch_file in enumerate(unshuffled_batch_files):
            with open(unshuffled_batch_file, 'rb') as f:                                                    # open the file
                print(f"Opening file {unshuffled_batch_file}.")
                X[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ]  = pickle.load(f)                                                              # and extract data (X) and labels (Y)
                Y_class[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ] = pickle.load(f)
                Y_loc[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ] = pickle.load(f)
            os.remove(unshuffled_batch_file)
        
        # do the shuffling        
        args = np.arange(0, X.shape[0])
        random.shuffle(args)
        X = X[args,]
        Y_class = Y_class[args,]
        Y_loc = Y_loc[args,]
        
        # save parts of the large shuffled array into separate files.  
        for file_n_batch in range(file_batch_size):
            print(f"    Saving shuffled file {file_n}")
            with open(outdir / f"data_file_shuffled_{file_n:05d}.pkl", 'wb') as f:                     # save the output as a pickle
                pickle.dump(X[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ], f)
                pickle.dump(Y_class[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ], f)
                pickle.dump(Y_loc[file_n_batch * n_per_file : (file_n_batch+1) * n_per_file, ], f)
            file_n += 1
           


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
