#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:20:34 2021

@author: matthew
"""



def plot_latent_space(vae_decoder, n=30, figsize=15, z0_limits = [-2, 2], z1_limits = [-2, 2], digit_size = 28, outpath = None):
    """
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