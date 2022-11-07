#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:58:50 2021

@author: matthew
"""

#%%


def rescale_timeseries(displacement_r3, rescale_factor = 1.0):
    """Given a time series as a dict in the usual form, rescale it by a given factor.  Not tested for downsampling.  
    Inputs:
        displacment_r3 | dict | cumulative, lons, lats, mask, dem.  
        Rescale factor | float | factor to rescale by  
    Returns:
        displacment_r3_rescaled | dict | as above, but all rescaled.  
    History:
        2022_05_23 | MEG | Written.  
    """
    
    import numpy as np
    import numpy.ma as ma
    from skimage.transform import rescale

    #print(f"Rescaling the time series.  ")

    n_acq = displacement_r3['cumulative'].shape[0]
    displacement_r3_rescaled = {}
    
    mask = displacement_r3['cumulative'].mask[0,]                                                              # get a slice of the mask (as it's rank3, but we're using a time series with consistent pixels so it might as well be rank 2)
    displacement_r3_rescaled['mask'] = rescale(mask, rescale_factor, order = 0)                                                   # order 0 as boolean so don't want any interpolate
    try:
        displacement_r3_rescaled['dem'] = rescale(displacement_r3['dem'], rescale_factor)                          # 
    except:
        pass
    
    for ifg_n, ifg in enumerate(displacement_r3['cumulative']):
        ifg_rescaled = rescale(ifg, rescale_factor)
        if ifg_n == 0:
            displacement_r3_rescaled['cumulative'] = ma.zeros((n_acq, ifg_rescaled.shape[0],  ifg_rescaled.shape[1]))
        displacement_r3_rescaled['cumulative'] [ifg_n,] = ma.array(ifg_rescaled, mask = displacement_r3_rescaled['mask'])
        
    _, ny_rescaled, nx_rescaled = displacement_r3_rescaled['cumulative'].shape
    
    lons_mg, lats_mg = np.meshgrid(np.linspace(displacement_r3['lons'][0,0], displacement_r3['lons'][0,-1], nx_rescaled),           # x 
                                   np.linspace(displacement_r3['lats'][0,0], displacement_r3['lats'][-1,0], ny_rescaled))           # then y
    
    displacement_r3_rescaled['lons'] = lons_mg
    displacement_r3_rescaled['lats'] = lats_mg
    
    return displacement_r3_rescaled
        



#%%


def random_cropping(ifg, out_resolution = 224, def_loc = None):
    """ Given an input ifg that is larger than the crop required, make 9 crops of the ifg to the new scale.  
    If no labels (def_loc) are provided, then these are quasi random to sample top left/ middle left, bottom left etc (i.e.9)
    If labels are provided, it's random so that all of the label is retained.  
    
    Inputs:
        ifg | rank 2 masked array | interferogram.  
        out_resolution | int | output side length (output is square)
        def_loc | rank 2 array | closed polygon of deformation, x then y, in pixels.  e.g.:
                                                                                            array([[168, 183],
                                                                                                   [243, 183],
                                                                                                   [243, 233],
                                                                                                   [168, 233],
                                                                                                   [168, 183]])
                                                                                            
    Returns:
        ifg_cropped | rank 3 array | out_resolution x out_resolution x 9
        Y_loc | rank 2 array | 9 x 4, x centre y centre x half width, y half width.  
        
    History:
        2022_05_16 | MEG | Written
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    
    bands = [0, 1/3, 2/3, 3/3]
    
    # # inputs sanity check
    # f, ax = plt.subplots()
    # ax.imshow(ifg)
    # ax.plot(def_loc[:,0], def_loc[:,1])
    
    ifg_cropped = ma.zeros((out_resolution, out_resolution, 9))             # initialise
    
    #pdb.set_trace()                
    
    ny, nx = ifg.shape
    y_padding = ny - out_resolution
    x_padding = nx - out_resolution
    
    if def_loc is None:                                                                                                 # if no deformaiton label, cropping is easy as can take any part of the data.  
        data_n = 0
        for yband in np.arange(3):
            for xband in np.arange(3):
                y_start = np.random.randint(low = bands[yband] * y_padding , high = bands[yband + 1] * y_padding )
                x_start = np.random.randint(low = bands[xband] * x_padding , high = bands[xband + 1] * x_padding )
                ifg_cropped[:,:,data_n] = ifg[y_start: y_start + out_resolution, x_start: x_start + out_resolution]
                data_n += 1
        return ifg_cropped
                
    else:                                                                                                               # but, if we do have location information, cropping is harder, and completely random.  
        Y_loc = np.zeros((9, 4))                                                                                        # initialise to store locations for each crop.  
        def_x_start = np.min(def_loc[:,0])
        def_x_stop = np.max(def_loc[:,0])
        def_x_centre = int(np.mean([def_x_start, def_x_stop]))                                                          # 
        def_x_half_width = (def_x_stop - def_x_start)/2                                                                 # this won't be changed by cropping
        def_y_start = np.min(def_loc[:,1])
        def_y_stop = np.max(def_loc[:,1])
        def_y_centre = int(np.mean([def_y_start, def_y_stop]))                                              
        def_y_half_width = (def_y_stop - def_y_start)/2                                                                 # also won't be changed by cropping
        
        x_low = def_x_stop - out_resolution                                         # the lowest that x can start at and the max x deformation still be in the image
        if x_low < 0:
            x_low = 0 
        x_high = def_x_start
        if x_high > (nx - out_resolution):
            x_high = nx - out_resolution    
        
        
        y_low = def_y_stop - out_resolution                                         # the lowest that x can start at and the max x deformation still be in the image
        if y_low < 0:
            y_low = 0 
        y_high = def_y_start
        if y_high > (ny - out_resolution):
            y_high = ny - out_resolution
        
            
        data_n = 0 
        for i in range(9):
            #pdb.set_trace()
            x_start = np.random.randint(x_low, x_high)            # x_high is just the lowest value for the deformaiton.  
            y_start = np.random.randint(y_low, y_high)            #
            
            ifg_cropped[:,:, data_n] = ifg[y_start : y_start + out_resolution, x_start : x_start + out_resolution]
            Y_loc[data_n, :] = np.array([def_x_centre- x_start, def_y_centre - y_start, def_x_half_width, def_y_half_width])
            data_n +=1 
        return ifg_cropped, Y_loc
        


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