#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:31:41 2022

@author: matthew
"""

#%%

def moving_average(ts, window = 11):
    """ A simple moving average function that reduces the window size at either edge of the 1d data.  
        Therefore the smoothed data is the same size as the original.  
        Can be tested with something like this:
            #smooth, valid =  moving_average(np.arange(20), window = 7)                                # testing fuctnion
    Inputs:
        ts | rank 1 data | the 1d time series to be smoothed.  
        window | int | odd number that sets the window size.  If 3, one data point either side of the data will be used for the smoothing.  
    
    Returns:
        ts_smooth | rank 1 data | smoothed version of the time series.  
        valid | rank 1 data | 0 if average for that point has edge effects (i.e. the ful window could not be used for averaging), 1 if full window was used.  
    History:
        2022_01_?? | MEG | Written
        2022_02_17 | MEG | add "valid" return.  
        
    """
    import numpy as np
    
    
    if window % 2 == 0:
        raise Exception(f"'window' must be odd, but is even.  Exiting.")
    half_window = int((window-1)/2)
    
    n_points = ts.shape[0]
    ts_smooth = np.zeros(n_points)
    valid = np.ones(n_points)
    
    for point_n in range(n_points):
        window_stop_adjusted = False                                                    # set (or reset)
#        pdb.set_trace()
        window_start = point_n - half_window
        if window_start < 0:
            window_start = 0
            window_stop = point_n + (point_n - window_start) + 1
            window_stop_adjusted = True
            valid[point_n] = 0
        
        if not window_stop_adjusted:
            window_stop = point_n + half_window +1
            if window_stop > n_points:
                window_stop = n_points                                          # if there are 20 points, this will also be 20 so can be indexced up to
                window_start = point_n - (window_stop - point_n) + 1
                valid[point_n] = 0

        ts_smooth[point_n] = np.mean(ts[window_start : window_stop])

        #print(f"for {ts[point_n]}: {ts[window_start : window_stop]}")               # debug/test
    return ts_smooth, valid

#%%