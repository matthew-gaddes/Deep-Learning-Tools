#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:31:41 2022

@author: matthew
"""



#%%

def plot_all_metrics(batch_metrics, epoch_metrics, metrics = None, title = 'Training metrics', 
                     two_column = False, out_path = None, y_epoch_start = 0, no_window = False,
                     n_batch_average = 51):
    """ Given a dict of metrics for every batch and for every epoch, plot the combination of the two.  
    
    Inputs:
        batch_metrics | dist of lists | names of metrics are keys and list of metrics values are items.  There are as many entries as n_epochs * n _batches (i.e. for every batch that was used in training)
        epoch_metrics | dict of lists | standard metrics saved each epoch by Keras.                       There are as many entries as n_epochs  
        metrics | None or list | names of metrics to plot (i.e. so we don't have to plot all fo them).  If None, just plot all.
        title | string | figure and window title.  
        two_column | boolean | If True, plots are in two columns.  
        y_epoch_start | float | epoch number to start y values of plot on (i.e. to crop out very different values in first epoch).  Can be fractions of an epoch.  
    Returns:
        Figure
    History:
        2021_11_11 | MEG | Written
        2022_11_01 | MEG | Update so that y limits are adjusted to avoid first epoch values that are very different to rest
        2022_11_08 | MEG | Add moving average that only plots where valid.  
        
    """
    import numpy.ma as ma
    
    from deep_learning_tools.plotting import moving_average

    
    import matplotlib.pyplot as plt    
    import numpy as np
        
    if no_window:
        plt.switch_backend('agg')                                                                       # with this backend, no windows are created during figure creation.  
    
    if metrics is None:                                                                                   # if no specific metrics are requested
        metrics = list(batch_metrics.keys())                                                              # just get them all from the batch metrics
    
    n_losses_total = len(batch_metrics[list(batch_metrics.keys())[0]])
    n_epochs = len(epoch_metrics[list(epoch_metrics.keys())[0]])                                            # get an item from the validation dict and see how long it is.  
    n_batches = int(n_losses_total / n_epochs)                                                            # get the number of entries in the first item of the batch_metrics (which is epochs x n_batches), then divided by epochs to get n_batches
    n_metrics = len(metrics)
    
    if two_column == False:
        fig1, axes = plt.subplots(1, n_metrics, figsize = (28,7))                                           # many rows, one column
    else:
        fig1, axes = plt.subplots(int(np.ceil(n_metrics/2)), 2, figsize = (14,7))                           # seems to wor, first bit works out how many rows we need.  
    fig1.canvas.manager.set_window_title(title)
    
    xvals_batch = np.arange(0, n_losses_total)                                                           # for every batch in every epoch, the xvalue to plot it at
    xvals_epoch = xvals_batch[::-1][::n_batches][::-1]                                                      # for every epoch, the x value to plot it at (i.e. every n_epochs)
    
    
    for plot_n, metric in enumerate(metrics):
        ax = np.ravel(axes)[plot_n]                                                                     # get the ax to plot on
        ax.scatter(xvals_batch, batch_metrics[metric], c = 'k', marker = '.', alpha = 0.5)              # plot for each batch
        ts_smoothed, valid = moving_average(np.array(batch_metrics[metric]), window = n_batch_average)                   # calculate the moving average, valid is 1 where the full window was available.  
        ax.plot(ma.array(xvals_batch, mask = (1 - valid)), ma.array(ts_smoothed, mask = (1 - valid)))                    # plot the moving average, but only where valid (rank 1 masked arrays)
        ax.scatter(xvals_epoch, epoch_metrics[f"val_{metric}"], c = 'r', marker = 'o')                           # plot for each epoch
        ax.plot(xvals_epoch, epoch_metrics[f"val_{metric}"], c = 'r', marker = 'o')                           # plot for each epoch as a line
        ax.set_ylabel(metric)
        ax.grid(True)

        if 'accuracy' in metric:                                                                                        # accuracy should increase and can't get higher than 1.  Adjust lower limit
            if (y_epoch_start != 0) or (y_epoch_start != 0.):
                #ax.set_ylim(bottom = batch_metrics[metric][xvals_batch[int(y_epoch_start * n_batches)]], top = 1)      # set y limits, note that upper can be the value after a certain number of epochs (i.e. so can crop out the first high values)
                ax.set_ylim(bottom = ts_smoothed[xvals_batch[int(y_epoch_start * n_batches)]], top = 1)      # set y limits, note that upper can be the value after a certain number of epochs (i.e. so can crop out the first high values)
            else:
                ax.set_ylim(top = 1)      # set y limits, note that upper can be the value after a certain number of epochs (i.e. so can crop out the first high values)
        else:                                                                                                           # loss should decrease and can't get lower than 0.  Adjust upper limi.t  
            if (y_epoch_start != 0) or (y_epoch_start != 0.):
                #ax.set_ylim(bottom = 0, top = (batch_metrics[metric][xvals_batch[int(y_epoch_start * n_batches)]]))      # set y limits, note that upper can be the value after a certain number of epochs (i.e. so can crop out the first high values)
                ax.set_ylim(bottom = 0, top = (ts_smoothed[xvals_batch[int(y_epoch_start * n_batches)]]))      # set y limits, note that upper can be the value after a certain number of epochs (i.e. so can crop out the first high values)
            else:
                ax.set_ylim(bottom = 0 )      # set y limits, note that upper can be the value after a certain number of epochs (i.e. so can crop out the first high values)
                
        ax.set_xlim(left = 0)
        
        if 'accuracy' in metric:                                                              # if accuracy is used in the metric, assume it's an accuracy and therefore
            ax.set_ylim(top = 1)                                                              # maxes out at 1
            
        ax.set_xticks(xvals_epoch)                                                            # change so a tick only after each epoch (and not each file)
        ax.set_xticklabels(np.arange(1,n_epochs+1, 1))                                          # number ticks
        ax.set_xlabel('Epoch number')
    
    if two_column and (not (n_metrics/2).is_integer()):                                        # if its two column and we didn't have an even number of metrics, delete the bottom right (left over one)
        np.ravel(axes)[-1].set_visible(False)
        
    if out_path is not None:
        fig1.savefig(out_path)                                                                   # 


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