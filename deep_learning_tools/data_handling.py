#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:58:50 2021

@author: matthew
"""

#%%

def custom_range_for_CNN_mean_centered(r4_array, custom_range, mean_centre = False):
    """ Rescale a rank 4 array so that the maximum (or minimum) of each channel's image touches the edge of a custom range.  
    e.g. input with range of [-5 10] and custom_range (4) is rescaled to [-1 2], and
        input with range of [-5 5] and custom_range (4) is rescaled to [-2 2], and
         input with range of [-10 5] and custom_range (4) is rescaled to [-2 1], so the range for all the data is 4.  

    This differs from the function below as that would would stretch the input so that each data touches both the min and max of the range (which includes shifting the mean).  
    
    Inputs:
        r4_array | r4 masked array | works with masked arrays?  
        custom_range | int | range for the data to lie in.  if set to 4, data would lie between -2 and 2
        mean_centre | boolean | if True, data are mean centered.  
    
    2019/03/20 | ?
    2021_06_21 | MEG | Update the docs, add to deel_learning_tools
                
    """
    import numpy as np
    from neural_network_functions import expand_to_r4
    
    if mean_centre:
        im_channel_means = np.mean(r4_array, axis = (1,2))                                                  # get the average for each image (in all thre channels)
        im_channel_means = expand_to_r4(im_channel_means, r4_array[0,:,:,0].shape)                                                   # expand to r4 so we can do elementwise manipulation
        r4_array -= im_channel_means                                                                        # do mean centering    

    im_channel_abs_max = np.max(np.abs(r4_array), axis = (1,2))                                             # find the biggest either negative or positive number for each image and its channel (ie abs max).  Retrns someting of n_data x n_channels.  
    im_channel_abs_max = expand_to_r4(im_channel_abs_max, r4_array[0,:,:,0].shape)                          # conver the previous array to rank 4                            
    r4_array = (custom_range/2)* (r4_array/im_channel_abs_max)                                              # divide by abs max so biggest +ve or -ve is 1, then rescale bu custom_range/2 so that all data will lie in custom_range.  

    return r4_array    

#%%

def custom_range_for_CNN(r4_array, min_max):
    """ Rescale a rank 4 array so that each channel's image lies in custom range
    e.g. input with range of (-5, 15) is rescaled to (-125 125) or (-1 1) for use with VGG16.  
    Designed for use with masked arrays.  
    Inputs:
        r4_array | r4 masked array | works with masked arrays?  
        min_max | dict | 'min' and 'max' of range desired as a dictionary.  
    Returns:
        r4_array | rank 4 numpy array | masked items are set to zero, rescaled so that each channel for each image lies between min_max limits.  
    History:
        2019/03/20 | now includes mean centering so doesn't stretch data to custom range.  
                    Instead only stretches until either min or max touches, whilst mean is kept at 0
        2020/11/02 | MEG | Update so range can have a min and max, and not just a range
        2021/01/06 | MEG | Upate to work with masked arrays.  Not test with normal arrays.
        2021_06_08 | MEG | Make a copy from VUDL-Net_21 repo.  
        2021_06_21 | MEG | Remove the pointless mean_centre argument.  
    """
    import numpy as np
    import numpy.ma as ma
    
    def expand_to_r4(r2_array, shape = (224,224)):
        """
        Calcaulte something for every image and channel in rank 4 data (e.g. 100x224x224x3 to get 100x3)
        Expand new rank 2 to size of original rank 4 for elemtiwise operations
        """
        import numpy as np
        
        r4_array = r2_array[:, np.newaxis, np.newaxis, :]
        r4_array = np.repeat(r4_array, shape[0], axis = 1)
        r4_array = np.repeat(r4_array, shape[1], axis = 2)
        return r4_array

    im_channel_min = ma.min(r4_array, axis = (1,2))                                         # get the minimum of each image and each of its channels
    im_channel_min = expand_to_r4(im_channel_min, r4_array[0,:,:,0].shape)                  # exapnd to rank 4 for elementwise applications
    r4_array -= im_channel_min                                                              # set so lowest channel for each image is 0
    
    im_channel_max = ma.max(r4_array, axis = (1,2))                                         # get the maximum of each image and each of its channels after the previous operation.  
    im_channel_max = expand_to_r4(im_channel_max, r4_array[0,:,:,0].shape)                  # expand to rank 4.  
    r4_array /= im_channel_max                                                              # should now be in range [0, 1]
    
    r4_array *= (min_max['max'] - min_max['min'])                                           # should now be in range [0, new max-min], note that min is noramlly negative so n_max - min is normally > new_max
    r4_array += min_max['min']                                                              # and now in range [new min, new max]        
    r4_nparray = r4_array.filled(fill_value = 0)                                            # convert to numpy array, maksed incoherent areas are set to zero.  
    
    return r4_nparray  


#%%