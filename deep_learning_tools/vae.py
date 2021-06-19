#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:20:34 2021

@author: matthew
"""

#%%
from tensorflow import keras


class CustomSaver(keras.callbacks.Callback):
    
    """Ideally model_output_dir could be set as a class variable in __init__, but I haven't worked out how to modify __init__
    and all the methods that it calls, without just copying the whole thing here.  """     
    
    def on_epoch_end(self, epoch, logs={}):                                                                 # overwrite the on_epoch_end default metho in the callback class.  
        self.model.encoder.save(str(self.model_output_dir / f"encoder_epoch_{epoch:03d}"))                  # model_output_dir will have to have been set.  
        self.model.decoder.save(str(self.model_output_dir / f"decoder_epoch_{epoch:03d}"))

#%% train a vae from a directoty full of numpy files.  
import tensorflow

class numpy_sequence(tensorflow.keras.utils.Sequence):                                                                                  # inheritance not tested like ths.  
        
    def __init__(self, file_list, batch_size):                                          # constructor
        """
        Inputs:
            file_list | list of strings or paths | locations of numpy files of data.  
            batch_size | int | number of data for each batch.  Note tested if larger than the number of data in a single file.  
        """
        self.file_list = file_list
        self.batch_size = batch_size

    def __len__(self):                                                      # number of batches in an epoch
        """As one large file (e.g. 1000 data) can't be used as a batch on a GPU (but maybe on a CPU?), then 
        this Sequence will break each file into a batch.  """
        
        import numpy as np
        n_files = len(self.file_list)                                                           # get the number of datat files.  
        n_data_per_file = np.load(self.file_list[0])['X'].shape[0]                              # get the number of data in a file (assumed to be the same for all files)
        n_batches_per_file = int(np.ceil(n_data_per_file / self.batch_size))                    # the number of batches required to cover every data in the file.  
        n_batches = n_files * n_batches_per_file
        return n_batches

    def __getitem__(self, idx):                                             # iterates over the data and returns a complete batch, index is a number upto the number of batches set by __len__, with each number being used once but in a random order.  
        
        import numpy as np
        # repeat of __len__ to get info about batch sizes etc, probably a better way to do this.  
        n_files = len(self.file_list)                                                      # get the number of data files.  
        n_data_per_file = np.load(self.file_list[0])['X'].shape[0]                         # get the number of data in a file (assumed to be the same for all files)
        n_batches_per_file = int(np.ceil(n_data_per_file / self.batch_size))               # the number of batches required to cover every data in the file.  
        n_batches = n_files * n_batches_per_file
        
        # deal with files and batches (convert idx to a file number and batch number).  
        n_file, n_batch = divmod(idx, n_batches_per_file)                                   # idx tells us which batch, but that needs mapping to a file, and to which batch in that file.  
        data = np.load(self.file_list[n_file])                                              # load the correct numpy file.  
        X = data['X']                                                                       # extract X from this (which is needed for the vae)
        
        # Open the correct file and batch, which depends on if it's the last batch for a file (as care needed to make sure it's the same size as all batches).  
        if n_batch == (n_batches_per_file - 1):                                             # the last batch may not have enough data for it, so is more complex to prepare
            X_unused = np.copy(X[n_batch * self.batch_size :, ])                            # get all the data not used in  batches so far, and make part of a batch with it
            n_required = self.batch_size - X_unused.shape[0]                                # find out how short of a batch we are.  
            extra_data_args = np.arange(n_batch * self.batch_size)                          # get the index of all the data that has been used in the previous batches.  
            np.random.shuffle(extra_data_args)                                              # shuffle it
            X_repeated = np.copy(X[extra_data_args[:n_required]], )                         # and make part of a batch with data that the network has already seen this epoch (ie repeated)
            return np.concatenate((X_unused, X_repeated), axis = 0)                         # merge the data that make an incomplete batch with some of the repeated data to make the final batch the right size.  
    
        else:
            return X[n_batch*self.batch_size : (n_batch+1) *self.batch_size, ]              # return part of X of size batch_size
            
      
        
      
############################################################################################################################################
############################################################################################################################################
###################################                                                                      ###################################
###################################                         Plotting                                     ###################################
###################################                                                                      ###################################
############################################################################################################################################
############################################################################################################################################      

#%%

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

#%%

def plot_latent_space(vae_decoder, n=30, figsize=15, z0_limits = [-2, 2], z1_limits = [-2, 2], digit_size = 28, outpath = None):
    """
    WARNING - WHAT ABOUT THE VERSION OF THIS IN 03_VAE_TESTER.PY
    
    Inputs:
        vae | keras model | model to decode spaces in the 2D latent space into images.  
        n | int | number of images in each direction in the plot
        figsize | int or float | figures is a square of side figsize inches.  
        zmax | float | latent space will be samplied from -zmax to zmax in both directions.  
        digit_size | int | number of pixels of square side of image created by vae_decoder
        outpath | string or path | path and filename to save .png to.  
    Returns:
        Figure
    History:
        2021_05_XX | MEG | Written
        2021_05_25 | MEG | Update to set digit_size, z0 and z1 limits,  and write docs.  
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    figure = np.zeros((digit_size * n, digit_size * n))                         # create a giant array of zeros to hold all the one channel images.  
    grid_x = np.linspace(z0_limits[0], z0_limits[1], n)                                      # same number of points as images, but spaced between -scale and scale
    grid_y = np.linspace(z1_limits[0], z1_limits[1], n)[::-1]

    for i, yi in enumerate(grid_y):                                             # i is image number, yi is z value
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])                                     # 1x2 of point in latent space
            x_decoded = vae_decoder.predict(z_sample)                           # keras rank 4 tensor of the image decoded from those z values.  
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size,
                   j * digit_size : (j + 1) * digit_size,] = digit              # write the digit to the giant figure of all the latent space decodings.  

    f = plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)                                      # plot the giant figure of all the latent space decodings.  
    plt.show()
    if outpath != None:
        f.savefig(outpath)


#%%

def plot_data_in_latent_space(Z_xy, Y = None, Y_labels= None, outpath = None):
    """Given the 2d latent space positions of data that has been passed through the encoder of a VAE, plot them.  
    If labels are provided (Y), colour the points by label.  
    
    Inputs:
        Z_xy | rank 2 array | nx2, where n is the number of ponts.  
        Y | rank 1 or 2 | n or nx1, labels of the points, not one hot (so e.g. [0,1,2,1,2,4]), though should correct them if this is the case
    Returns:
        Figure
    History:
        2021_05_18 | MEG | Written
        2021_05_25 | MEG | Update to include a legend rather than a colourbar.  
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    #0: Check that Y is not one hot, and convert if so
    if Y is not None:                                                                           # Y may be set to None if it doesn't exist.  
        if len(Y.shape) > 1:                                                                    # if it's a rank1, it shouldn't be one hot (categorical) encoding
            if Y.shape[1] > 1:                                                                  # but if it, it could still be nx1 and make sence, but if the number of columns is >1 then it's probably one hot
                print(f"'Y' has shape {Y.shape} which appears to be one hot encoding and is being reversed.  ")
                Y = np.argmax(Y, axis = 1)
                print(f"'Y' now has shape {Y.shape}. ")
        n_classes = np.max(Y)                                                                       # assume that there are as many classes as different labels.  
        
                    
    #1: Begin the figure
    f, ax = plt.subplots(1,1, figsize = (12,6))
    if Y is None:
        points = ax.scatter(Z_xy[:,0], Z_xy[:,1])                                                       # note that the labels can't be one hot
    else:
        for class_n in range(n_classes+1):
            class_args = np.ravel(np.argwhere(Y == class_n))                                            # find the args of which data are in that class (and make rank 1)
            points = ax.scatter(Z_xy[class_args,0], Z_xy[class_args,1])                                 # plot those points, colour changes automatically due to loop
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    if Y_labels is not None:
        plt.legend(Y_labels)                                                                           # Legend using list
    if outpath != None:
        f.savefig(outpath)